import copy
import json
from types import SimpleNamespace

import pytest

from pipeline.continuation import prepare_continuation, ContinuationUnavailableError


def scenario(tmp_path):
    tail = tmp_path / 'tail.jpg'
    tail.write_bytes(b'accepted-frame')
    previous = {'shot_id': 1, 'scene_id': 'roof', 'observed_end_state': {
        'location': 'roof', 'subject': 'White and Black',
        'action_phase': 'Black has already lowered his sword; White crouches',
        'prop_state': 'both hold one sword', 'open_motion': 'low blade presses inward'}}
    shot = {'shot_id': 2, 'duration': 5, 'scene_id': 'roof',
        'continuity_from_previous': 'seamless', 'primary_action': 'Black raises his sword again',
        'characters': ['White', 'Black'], 'camera': {'primary_movement': 'tracking'},
        'interaction_geometry': {'effect_phase': 'active', 'actor': 'White', 'target': 'Black'},
        'action_beats': [
            {'phase': 'trigger', 'actor': 'White', 'target': 'Black', 'action': 'Replay overhead wind-up', 'visible_result': 'wind-up'},
            {'phase': 'peak', 'actor': 'White', 'target': 'Black', 'action': 'White counters', 'visible_result': 'Black drops sword'},
            {'phase': 'aftermath', 'actor': 'White', 'target': 'Black', 'action': 'Black falls', 'visible_result': 'Black remains down'}],
        'end_state': {'action_phase': 'Black remains down'},
        'narrative_beat': {'state_before': 'raised sword', 'state_after': 'Black remains down'}}
    return shot, previous, tail


class Planner:
    def __init__(self, **changes):
        self.calls = []
        self.result = {'compatible': True, 'entry_action': 'White slips the already-low blade and rises into the counter',
                       'entry_visible_result': 'The incoming blade misses', 'reason': 'The downward cut already happened', **changes}
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(
            content=json.dumps(self.result), refusal=None))])


def test_adaptation_preserves_goal_and_is_reused_after_restart(tmp_path):
    shot, previous, tail = scenario(tmp_path)
    original = copy.deepcopy(shot)
    planner = Planner()
    changed = prepare_continuation(shot, previous, str(tail), tmp_path, client=planner)
    assert shot == original
    assert changed['action_beats'][0]['action'].startswith('White slips')
    assert changed['action_beats'][1:] == shot['action_beats'][1:]
    for key in ('characters', 'camera', 'end_state', 'interaction_geometry', 'duration'):
        assert changed[key] == shot[key]
    assert 'already lowered' in changed['start_state']['action_phase']
    assert changed['narrative_beat']['state_before'] != 'raised sword'
    assert 'raises his sword again' not in changed['primary_action']
    restored = prepare_continuation(shot, previous, str(tail), tmp_path, client=planner, allow_create=False)
    assert restored == changed and len(planner.calls) == 1


@pytest.mark.parametrize('changes', [
    {'compatible': False}, {'compatible': 'true'}, {'entry_action': ''},
    {'entry_action': 'x' * 1601}, {'end_state': {'action_phase': 'resurrect'}},
])
def test_bad_or_incompatible_plan_stops_before_video(tmp_path, changes):
    shot, previous, tail = scenario(tmp_path)
    with pytest.raises(ContinuationUnavailableError):
        prepare_continuation(shot, previous, str(tail), tmp_path, client=Planner(**changes))
    assert not list((tmp_path / 'continuations').glob('*.json'))


def test_changed_reference_or_missing_resume_plan_fails_closed(tmp_path):
    shot, previous, tail = scenario(tmp_path)
    planner = Planner()
    with pytest.raises(ContinuationUnavailableError):
        prepare_continuation(shot, previous, str(tail), tmp_path, client=planner, allow_create=False)
    assert not planner.calls
    prepare_continuation(shot, previous, str(tail), tmp_path, client=planner)
    tail.write_bytes(b'different-frame')
    with pytest.raises(ContinuationUnavailableError):
        prepare_continuation(shot, previous, str(tail), tmp_path, client=planner, allow_create=False)
    assert len(planner.calls) == 1


def test_no_observation_or_different_scene_never_invents_adaptation(tmp_path):
    shot, previous, tail = scenario(tmp_path)
    planner = Planner()
    for prev in ({**previous, 'observed_end_state': {}}, {**previous, 'scene_id': 'elsewhere'}):
        assert prepare_continuation(shot, prev, str(tail), tmp_path, client=planner) == shot
    assert not planner.calls


@pytest.mark.parametrize('finish,content,refusal', [
    ('length', '{}', None), ('stop', '{broken', None), ('stop', '{}', 'refused'),
])
def test_incomplete_planner_response_is_never_persisted(tmp_path, finish, content, refusal):
    shot, previous, tail = scenario(tmp_path)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **_: SimpleNamespace(choices=[SimpleNamespace(
            finish_reason=finish, message=SimpleNamespace(content=content, refusal=refusal))]))))
    with pytest.raises(ContinuationUnavailableError):
        prepare_continuation(shot, previous, str(tail), tmp_path, client=client)
    assert not (tmp_path / 'continuations' / 'shot_002.json').exists()
    diagnostics = list((tmp_path / 'continuations' / 'responses' / 'shot_002').glob('*.json'))
    assert len(diagnostics) == 1
    assert json.loads(diagnostics[0].read_text())['response']['choices'][0]['content'] == content


@pytest.mark.asyncio
async def test_generator_pending_resume_uses_same_entry_and_task(tmp_path, monkeypatch):
    from pathlib import Path
    from pipeline.generator import VideoGenerator
    from pipeline.models import RunOptions
    from pipeline.run_state import RunWorkspace

    base = json.loads((Path(__file__).resolve().parents[1] / 'examples/contact_previs/storyboard.json').read_text())
    shot = base['shots'][0]
    shot.update(shot_id=2, duration=5, scene_id='roof', continuity_from_previous='seamless')
    shot['action_beats'].insert(0, {'phase': 'trigger', 'actor': 'attacker', 'target': 'defender',
        'action': 'raise the lead palm', 'visible_result': 'palm is raised'})
    previous = {'shot_id': 1, 'scene_id': 'roof', 'observed_end_state': {
        'location': 'roof', 'subject': 'attacker and defender', 'action_phase': 'palm already raised',
        'prop_state': 'empty hands', 'open_motion': 'forward pressure'}}
    workspace = RunWorkspace.create(tmp_path, RunOptions(request='10秒连续动作', paid_take_budget=1))
    base = workspace.save_storyboard(base)
    shot = base['shots'][0]
    tail = workspace.path / 'tail.jpg'
    tail.write_bytes(b'accepted-tail')
    planner = Planner(entry_action='attacker continues forward from the already raised palm',
                      entry_visible_result='the palm closes the remaining gap')
    monkeypatch.setattr('pipeline.continuation.OpenAI', lambda **_: planner)
    submissions, polls = [], []
    class API:
        supports_last_frame = True
        async def generate(self, **kwargs):
            submissions.append(kwargs)
            kwargs['on_submitted']('pending-original')
            return {'status': 'failed', 'error_type': 'poll_timeout',
                    'provider_task_id': 'pending-original', 'error': 'Timeout (600s)'}
        async def poll_task(self, task_id, **_):
            polls.append(task_id)
            return {'status': 'succeeded', 'provider_task_id': task_id, 'video_url': 'fake-video'}
        async def download_video(self, _url, path):
            Path(path).write_bytes(b'fake-video')
    def progress(result):
        workspace.record_shot(shot_id=result.shot_id, status=result.status,
                              provider_task_id=result.provider_task_id, prompt_profile=result.prompt_profile,
                              prompt_fingerprint=result.prompt_fingerprint,
                              compiled_contract_version=result.compiled_contract_version,
                              compiled_contract_fingerprint=result.compiled_contract_fingerprint)
    def generator(**kwargs):
        gen = VideoGenerator(str(workspace.path), on_progress=progress,
            reserve_paid_take=workspace.reserve_paid_take,
            confirm_paid_take_submission=workspace.confirm_paid_take_submission,
            reconcile_paid_take=workspace.reconcile_paid_take, **kwargs)
        gen.api = API()
        gen.semantic_reviewer = None
        return gen
    first = await generator()._generate_single_shot(copy.deepcopy(shot), str(tail), previous, base)
    assert first.status == 'running'
    assert len(submissions) == len(planner.calls) == 1
    assert 'already raised palm' in submissions[0]['prompt']
    assert submissions[0]['image_urls'] == [str(tail)]
    pending = RunWorkspace.resume(workspace.path).resumable_pending_tasks()
    assert pending[2]['task_id'] == 'pending-original'
    second = generator(resume_tasks=pending)
    monkeypatch.setattr('pipeline.generator.check_video_quality',
                        lambda _: {'pass': True, 'quality_score': 100, 'issues': []})
    monkeypatch.setattr(second, '_extract_local_tail_frame', lambda *_: str(tail))
    accepted = await second._generate_single_shot(copy.deepcopy(shot), str(tail), previous, base)
    assert accepted.status == 'success'
    assert polls == ['pending-original']
    assert len(submissions) == len(planner.calls) == 1
