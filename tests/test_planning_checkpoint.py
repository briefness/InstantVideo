"""Planning progress survives interruption without resampling completed creative work."""
import asyncio
import json
from pathlib import Path

import pytest

from pipeline.orchestrator import VideoPipeline
from test_action_results import annotated_story
from test_action_result_flow import annotated_verdict
from test_director_intent import Responses, intent_reply
from test_full_production_flow import production_services


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket
    def deny(*args, **kwargs):
        pytest.fail('planning recovery attempted a network connection')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket.socket, 'connect_ex', deny)


def stop_after_first_shot(pipeline, monkeypatch):
    save = pipeline._save_json
    def interrupted(name, value):
        save(name, value)
        if name == 'planning_lowering.json':
            raise asyncio.CancelledError()
    monkeypatch.setattr(pipeline, '_save_json', interrupted)


def clients(monkeypatch, brief):
    raw = annotated_story()
    raw['scene_entities'] = ['Door', 'Dust']
    for shot in raw['shots']:
        for unit in shot['action_plan']:
            if unit['kind'] == 'interaction':
                unit['weight'] = 3.
    author = Responses({**intent_reply(brief), 'choreography': 'flexible'}, raw)
    class Review:
        def __init__(self):
            self.chat = self.completions = self
            self.calls = []
        def create(self, **kwargs):
            story = json.loads(kwargs['messages'][-1]['content'])['storyboard']
            self.calls.append(story)
            return Responses(annotated_verdict(story)).create(**kwargs)
    reviewer = Review()
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **kw: author)
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **kw: reviewer)
    return author, reviewer


@pytest.mark.asyncio
@pytest.mark.parametrize('brief', ['15秒，三段展示双人训练', '15秒，三段展示工人交接物品'])
async def test_formal_planning_resumes_completed_prefix_without_model_replay(tmp_path, monkeypatch, brief):
    import config
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path)
    author, reviewer = clients(monkeypatch, brief)
    monkeypatch.setattr('pipeline.orchestrator.VideoGenerator', lambda **kw: pytest.fail('video stage reached'))
    pipeline = VideoPipeline(plan_only=True, paid_take_budget=0, take_policy='single_pass')
    stop_after_first_shot(pipeline, monkeypatch)
    with pytest.raises(asyncio.CancelledError):
        await pipeline.run(brief)
    assert len(author.calls) == 3 and not (pipeline.workspace / 'storyboard.json').exists()
    first = (pipeline.workspace / 'planning_shot_001_response.json').read_bytes()
    resumed = VideoPipeline.from_workspace(pipeline.workspace)
    resumed.plan_only = True
    result = await resumed.run()
    assert Path(result).is_file() and len(author.calls) == 5 and len(reviewer.calls) == 1
    assert (pipeline.workspace / 'planning_shot_001_response.json').read_bytes() == first
    assert resumed.run_workspace.manifest.stage == 'planning_ready'
    assert not resumed.run_workspace.manifest.paid_take_budget.reservations


@pytest.mark.asyncio
async def test_recovered_planning_reaches_actual_ffmpeg_export_with_offline_providers(production_services, monkeypatch):
    brief, services = production_services
    pipeline = VideoPipeline(resolution='720p', platforms=['youtube'], paid_take_budget=3,
                             take_policy='single_pass')
    stop_after_first_shot(pipeline, monkeypatch)
    with pytest.raises(asyncio.CancelledError):
        await pipeline.run(brief)
    assert len(services.author.calls) == 3 and not services.submissions
    resumed = VideoPipeline.from_workspace(pipeline.workspace)
    result = await resumed.run()
    assert Path(result).is_file() and len(services.author.calls) == 5
    assert len(services.submissions) == 3
    assert resumed.run_workspace.manifest.status.value == 'succeeded'


def workspace_at(tmp_path):
    from pipeline.models import RunOptions
    from pipeline.run_state import RunWorkspace
    return RunWorkspace.create(tmp_path, RunOptions(request='planning checkpoint test', paid_take_budget=0))


@pytest.mark.parametrize('failure', [RuntimeError('offline rejected response'), asyncio.CancelledError()])
def test_failed_or_unknown_request_is_never_reissued(tmp_path, failure):
    from pipeline.planning_checkpoint import PlanningCheckpoint
    workspace = workspace_at(tmp_path)
    checkpoint = PlanningCheckpoint(workspace, resume=False)
    calls = []
    def operation():
        calls.append(1)
        raise failure
    with pytest.raises(type(failure)):
        checkpoint.step('intent', {}, operation)
    before = checkpoint.path.read_bytes()
    for _ in range(2):
        with pytest.raises(ValueError, match='失败或请求结果不明'):
            PlanningCheckpoint(workspace, resume=True).step('intent', {}, operation)
    assert calls == [1] and checkpoint.path.read_bytes() == before


@pytest.mark.parametrize('when', ['before_request', 'after_response'])
def test_checkpoint_write_failure_never_repeats_external_work(tmp_path, monkeypatch, when):
    from pipeline.planning_checkpoint import PlanningCheckpoint
    workspace = workspace_at(tmp_path)
    checkpoint = PlanningCheckpoint(workspace, resume=False)
    calls = []
    write = workspace._atomic_write_json
    def fail_write(path, value):
        state = value['steps'][-1]['state']
        if state == ('started' if when == 'before_request' else 'returned'):
            raise OSError('offline disk failure')
        write(path, value)
    def operation():
        calls.append(1)
        return {'ok': True}
    monkeypatch.setattr(workspace, '_atomic_write_json', fail_write)
    with pytest.raises(OSError):
        checkpoint.step('intent', {}, operation)
    assert len(calls) == (0 if when == 'before_request' else 1)
    if when == 'after_response':
        monkeypatch.setattr(workspace, '_atomic_write_json', write)
        with pytest.raises(ValueError, match='结果不明'):
            PlanningCheckpoint(workspace, resume=True).step('intent', {}, operation)
        assert len(calls) == 1


@pytest.mark.parametrize('change', ['value', 'delete_step', 'duplicate_key', 'symlink'])
def test_corrupt_receipt_is_rejected_before_any_model_request(tmp_path, change):
    from pipeline.planning_checkpoint import PlanningCheckpoint
    workspace = workspace_at(tmp_path)
    checkpoint = PlanningCheckpoint(workspace, resume=False)
    checkpoint.step('intent', {'request': 'A'}, lambda: {'original': True})
    record = json.loads(checkpoint.path.read_text())
    if change == 'value':
        record['steps'][0]['value']['original'] = False
    elif change == 'delete_step':
        record['steps'] = []
    if change == 'duplicate_key':
        checkpoint.path.write_text(checkpoint.path.read_text().replace('"steps":', '"steps": [], "steps":', 1))
    elif change == 'symlink':
        outside = tmp_path / 'outside.json'
        outside.write_text(json.dumps(record))
        checkpoint.path.unlink()
        checkpoint.path.symlink_to(outside)
    else:
        checkpoint.path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='损坏|不一致'):
        PlanningCheckpoint(workspace, resume=True).step('intent', {'request': 'A'},
            lambda: pytest.fail('corrupt progress caused a repeated request'))


def test_shared_workspace_serializes_planning_requests(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from pipeline.planning_checkpoint import PlanningCheckpoint
    from pipeline.run_state import RunWorkspace
    workspace = workspace_at(tmp_path)
    first = PlanningCheckpoint(workspace, resume=False)
    second = PlanningCheckpoint(RunWorkspace.resume(workspace.path), resume=True)
    entered, release = Event(), Event()
    calls = []
    def operation():
        calls.append(1)
        entered.set()
        assert release.wait(5)
        return {'ok': True}
    with ThreadPoolExecutor(max_workers=1) as pool:
        work = pool.submit(first.step, 'intent', {}, operation)
        try:
            assert entered.wait(5)
            with pytest.raises(RuntimeError, match='正在运行'):
                second.step('intent', {}, operation)
        finally:
            release.set()
        assert work.result() == {'ok': True}
    assert second.step('intent', {}, operation) == {'ok': True} and len(calls) == 1


@pytest.mark.asyncio
async def test_changed_model_or_schema_cannot_reuse_or_regenerate_plan(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path)
    author, _ = clients(monkeypatch, '15秒，三段连续操作')
    pipeline = VideoPipeline(plan_only=True, paid_take_budget=0)
    stop_after_first_shot(pipeline, monkeypatch)
    with pytest.raises(asyncio.CancelledError):
        await pipeline.run('15秒，三段连续操作')
    with monkeypatch.context() as context:
        context.setattr(config, 'LLM_MODEL', 'different-model')
        with pytest.raises(ValueError, match='不一致'):
            await VideoPipeline.from_workspace(pipeline.workspace).run()
    import pipeline.storyboard_authoring as authoring
    schema = authoring.authored_shot_response_schema
    def changed(*args, **kwargs):
        value = schema(*args, **kwargs)
        value['description'] = 'offline changed response contract'
        return value
    monkeypatch.setattr(authoring, 'authored_shot_response_schema', changed)
    with pytest.raises(ValueError, match='输入或顺序已变化'):
        await VideoPipeline.from_workspace(pipeline.workspace).run()
    assert len(author.calls) == 3


@pytest.mark.asyncio
async def test_saved_step_still_requires_typed_compilation_and_cannot_be_promoted(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path)
    brief = '15秒，三段连续操作'
    raw = annotated_story()
    raw['scene_entities'] = ['Door', 'Dust']
    raw['shots'][0]['action_plan'][0]['actor'] = 'unknown_actor'
    author = Responses({**intent_reply(brief), 'choreography': 'flexible'}, raw)
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **kw: author)
    monkeypatch.setattr('pipeline.orchestrator.VideoGenerator', lambda **kw: pytest.fail('video stage reached'))
    pipeline = VideoPipeline(plan_only=True, paid_take_budget=0)
    with pytest.raises(ValueError, match='未注册'):
        await pipeline.run(brief)
    for _ in range(2):
        with pytest.raises(ValueError, match='未注册'):
            await VideoPipeline.from_workspace(pipeline.workspace).run()
    assert len(author.calls) == 3 and not pipeline.run_workspace.manifest.paid_take_budget.reservations


@pytest.mark.asyncio
async def test_missing_frozen_storyboard_after_planning_is_not_regenerated(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path)
    author, _ = clients(monkeypatch, '15秒，三段连续操作')
    pipeline = VideoPipeline(plan_only=True, paid_take_budget=0)
    await pipeline.run('15秒，三段连续操作')
    (pipeline.workspace / 'storyboard.json').unlink()
    with pytest.raises(ValueError, match='no validated storyboard'):
        await VideoPipeline.from_workspace(pipeline.workspace).run()
    assert len(author.calls) == 5


@pytest.mark.asyncio
async def test_recovery_and_uninterrupted_run_freeze_identical_storyboards(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path)
    brief = '15秒，三段连续操作展示'
    author, _ = clients(monkeypatch, brief)
    interrupted = VideoPipeline(plan_only=True, paid_take_budget=0)
    stop_after_first_shot(interrupted, monkeypatch)
    with pytest.raises(asyncio.CancelledError):
        await interrupted.run(brief)
    resumed = VideoPipeline.from_workspace(interrupted.workspace)
    resumed.plan_only = True
    recovered = json.loads(Path(await resumed.run()).read_text())
    assert len(author.calls) == 5
    fresh_author, _ = clients(monkeypatch, brief)
    fresh = VideoPipeline(plan_only=True, paid_take_budget=0)
    direct = json.loads(Path(await fresh.run(brief)).read_text())
    assert len(fresh_author.calls) == 5 and direct == recovered
    assert json.loads((fresh.workspace / 'planning_checkpoint.json').read_text()) == json.loads(
        (resumed.workspace / 'planning_checkpoint.json').read_text())


@pytest.mark.asyncio
async def test_real_failed_response_stays_failed_and_does_not_repeat_on_resume(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    import config
    from test_structured_response_stream import Chunks
    fixture = json.loads((Path(__file__).parent / 'fixtures/real_repeated_field_failure.json').read_text())
    sources = fixture['sources']
    responses = iter([
        NS(choices=[NS(message=NS(content=sources['planning_intent_response.json']['value']['content']),
                       finish_reason='stop')]),
        Chunks(sources['planning_identity_response.json']['value']['content']),
        Chunks(sources['planning_shot_001_response.json']['value']['content']),
    ])
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return next(responses)
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **kw: NS(chat=NS(completions=NS(create=create))))
    monkeypatch.setattr('pipeline.orchestrator.VideoGenerator', lambda **kw: pytest.fail('video stage reached'))
    pipeline = VideoPipeline(plan_only=True, paid_take_budget=0)
    # The unmodified historical identity now violates the earlier requirement
    # authority boundary. Replaying it must stop before the repeated-key shot.
    with pytest.raises(ValueError, match='用户结局'):
        await pipeline.run(sources['run_manifest.json']['value']['options']['request'])
    partial = (pipeline.workspace / 'planning_identity_response.json').read_bytes()
    record = (pipeline.workspace / 'planning_checkpoint.json').read_bytes()
    for _ in range(2):
        with pytest.raises(ValueError, match='用户结局'):
            await VideoPipeline.from_workspace(pipeline.workspace).run()
    assert len(calls) == 2
    assert not (pipeline.workspace / 'planning_shot_001_response.json').exists()
    assert (pipeline.workspace / 'planning_identity_response.json').read_bytes() == partial
    assert (pipeline.workspace / 'planning_checkpoint.json').read_bytes() == record
    assert not pipeline.run_workspace.manifest.paid_take_budget.reservations


def test_non_event_draft_checkpoint_replays_before_pure_compilation(tmp_path, monkeypatch):
    from pipeline.planning_checkpoint import PlanningCheckpoint
    from pipeline.storyboard import generate_storyboard
    from test_director_intent import draft
    workspace = workspace_at(tmp_path)
    brief = workspace.options.request
    author = Responses(intent_reply(brief, focus='balanced', pacing='atmospheric'), draft())
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **kw: author)
    snapshots = []
    def pause_after_draft(name, value):
        if name == 'planning_draft_1.json':
            snapshots.append(value)
            raise asyncio.CancelledError()
    for resume in (False, True):
        with pytest.raises(asyncio.CancelledError):
            generate_storyboard(brief, target_duration=15, save_diagnostic=pause_after_draft,
                planning_checkpoint=PlanningCheckpoint(workspace, resume=resume))
    assert len(author.calls) == 2 and snapshots[0] == snapshots[1]
