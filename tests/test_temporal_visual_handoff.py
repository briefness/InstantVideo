"""真实原片回放只验证数据流；服务替身不能证明视觉模型准确率。"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_goal_allocation import offline
from test_full_production_flow import completion

ROOT = Path(__file__).parents[1] / 'output/20261110_171057_913270'


def captured():
    story = json.loads((ROOT / 'result_admission.json').read_text())['execution_storyboard']
    record = json.loads((ROOT / 'continuations/shot_002.json').read_text())
    previous = {**story['shots'][0], 'observed_end_state': record['input']['observed_state'],
                '_accepted_video_path': str(ROOT / 'shots/shot_001.mp4')}
    shot = {**deepcopy(story['shots'][1]), 'continuity_evidence': 'temporal-window-v1'}
    return story, shot, previous, record


@pytest.mark.parametrize('phase', ['active', 'none'])
def test_formal_continuation_sees_motion_history_even_for_hold_shots(tmp_path, phase):
    from pipeline.continuation import prepare_continuation
    story, shot, previous, record = captured()
    shot['interaction_geometry']['effect_phase'] = phase
    calls = []
    def create(**request):
        calls.append(request)
        plan = deepcopy(record['plan'])
        if 'observed_motion' in request['response_format']['json_schema']['schema']['properties']:
            plan['observed_motion'] = {'summary': 'The hand moves the object downward; it remains in contact.',
                                       'sample_indices': [1, 9]}
        return completion(plan)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = prepare_continuation(shot, previous, str(ROOT / 'shots/shot_001_lastframe.jpg'),
                                  tmp_path, client=client)
    assert len(calls) == 1
    images = [part for m in calls[0]['messages'] if isinstance(m['content'], list)
              for part in m['content'] if part.get('type') == 'image_url']
    assert len(images) >= 9
    assert result['motion_handoff']['summary'].startswith('The hand moves')
    from pipeline.provider_prompt import compile_normal_provider_prompt, compile_policy_safe_prompt
    assert result['motion_handoff']['summary'] in compile_normal_provider_prompt(result, story, has_observed_start=True)
    assert result['motion_handoff']['summary'] in compile_policy_safe_prompt(
        result, storyboard=story, has_state_reference=True, image_role='reference_image', reference_count=3)
    assert prepare_continuation(shot, previous, str(ROOT / 'shots/shot_001_lastframe.jpg'),
                                tmp_path, client=client, allow_create=False) == result
    assert len(calls) == 1


def test_independent_check_receives_pixels_instead_of_only_a_positive_caption(tmp_path):
    from pipeline.semantic_review import SemanticTakeReviewer
    story, _, _, _ = captured()
    shot = {**story['shots'][0], 'continuity_evidence': 'temporal-window-v1'}
    digest = '9f22723b3d462536643db7c68d6da287fe5c17c7f73968cd614e488fd0bfd096'
    raw = json.loads((ROOT / f'semantic_reviews/{digest}.json').read_text())['observation_response']
    calls = []
    def create(**request):
        calls.append(request)
        if request['response_format']['json_schema']['name'] != 'observation_support':
            return completion(deepcopy(raw))
        content = request['messages'][-1]['content']
        payload = json.loads(content[0]['text'] if isinstance(content, list) else content)
        return completion({'decisions': {k: 'contradicted' if '/1@' in k else 'supported'
                                          for k in payload['cases']}})
    reviewer = SemanticTakeReviewer(tmp_path, model='offline',
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    first = reviewer.review(str(ROOT / 'shots/shot_001.mp4'), shot)
    assert first.accepted is False
    images = [p for m in calls[-1]['messages'] if isinstance(m['content'], list)
              for p in m['content'] if p.get('type') == 'image_url']
    assert len(images) > 1
    assert reviewer.review(str(ROOT / 'shots/shot_001.mp4'), shot).accepted is False
    assert len(calls) == 2


def test_missing_history_or_observation_stops_before_any_call(tmp_path):
    from pipeline.continuation import prepare_continuation, ContinuationUnavailableError
    _, shot, previous, _ = captured()
    def forbidden(**request):
        pytest.fail('missing evidence must stop before model call')
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=forbidden)))
    for missing in ('_accepted_video_path', 'observed_end_state'):
        broken = {k: v for k, v in previous.items() if k != missing}
        with pytest.raises(ContinuationUnavailableError):
            prepare_continuation(shot, broken, str(ROOT / 'shots/shot_001_lastframe.jpg'),
                                 tmp_path, client=client)


def test_changed_video_cannot_reuse_continuation_with_the_same_tail(tmp_path):
    from pipeline.continuation import prepare_continuation, ContinuationUnavailableError
    _, shot, previous, record = captured()
    calls = []
    def create(**request):
        calls.append(request)
        return completion({**deepcopy(record['plan']), 'observed_motion': {
            'summary': 'Offline synthetic annotation: continuing motion.', 'sample_indices': [1, 9]}})
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    tail = str(ROOT / 'shots/shot_001_lastframe.jpg')
    prepare_continuation(shot, previous, tail, tmp_path, client=client)
    previous['_accepted_video_path'] = str(ROOT / 'shots/shot_002.mp4')
    with pytest.raises(ContinuationUnavailableError, match='不匹配'):
        prepare_continuation(shot, previous, tail, tmp_path, client=client, allow_create=False)
    assert len(calls) == 1
