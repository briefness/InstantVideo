"""传输分批不截断证据；合并、恢复及拒绝路径均使用正式入口和离线服务。"""
from copy import deepcopy
import json

import jsonschema
import pytest

from pipeline.goal_allocation import requirement_key
from pipeline.planning_review import ensure_planning_semantics
from pipeline.run_state import RunWorkspace
from test_goal_allocation import evidence, offline
from test_planning_review_batches import long_story


def batch_story(goal):
    story = long_story()
    story['production_plan']['director_intent'].update(authoring_protocol='scene-units-v1',
        source_request=goal, completion_goals=[{'goal': goal, 'basis': 'user', 'evidence': goal}])
    for shot in story['shots']:
        shot['result_admission'] = 'review-required-v1'
    return story


class Reviewer:
    def __init__(self, rejected_batch=None):
        self.chat = self.completions = self
        self.calls = []
        self.rejected_batch = rejected_batch

    def create(self, **kwargs):
        from test_action_result_flow import annotated_verdict
        from planning_protocol_cases import wire_verdict, annotated_outcome
        from test_full_production_flow import completion
        payload = json.loads(kwargs['messages'][-1]['content'])
        self.calls.append(payload)
        verdict = annotated_outcome(annotated_verdict(payload['storyboard']), payload['storyboard'])
        ref, goal = next(iter(payload['evidence_sources']['goals'].items()))
        part = evidence(goal['goal'], *payload['review_targets'])
        if len(self.calls) == self.rejected_batch:
            part['relation'] = 'undetermined'
        verdict['goal_allocation'] = {ref: {'parts': [part]}}
        value = wire_verdict(verdict, kwargs)
        schema = kwargs['response_format']['json_schema']['schema']
        jsonschema.validate(value, schema)
        selected = schema['properties']['goal_allocation']['properties'][ref]['properties']['parts']['items']['properties']['units']
        # 方舟请求适配会移除 maxItems；真实目录仍发送，解码必须自行守住边界。
        assert selected['items']['enum'] == list(payload['review_targets'])
        return completion(value)


@pytest.mark.parametrize('goal', [
    'The dancers maintain their formation throughout the performance.',
    'The package stays sealed and in the courier custody throughout the journey.',
    'The rescue team keeps the rope attached throughout the rescue.',
])
def test_long_film_preserves_all_joint_evidence_and_resumes_without_new_reviews(goal, tmp_path, monkeypatch):
    story = batch_story(goal)
    original = deepcopy(story)
    sdk = Reviewer()
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **_: sdk)
    save = lambda name, data: RunWorkspace._atomic_write_json(tmp_path / name, data)
    bound = ensure_planning_semantics(story, tmp_path, save)
    assert story == original
    assert sum(len(s['action_beats']) for s in bound['shots']) > 120
    owners, groups = {}, {}
    for shot in bound['shots']:
        for index, beat in enumerate(shot['action_beats']):
            req, = beat['completion_intent']['requirements']
            key = requirement_key(req)
            owners.setdefault(key, []).append((shot['shot_id'], index))
            groups[key] = [(u['shot_id'], u['beat_index']) for u in req['scope']]
    assert owners == groups
    assert len(groups) == len(sdk.calls)
    assert all(12 < len(g) <= 24 for g in groups.values())
    expected = [(s['shot_id'], i) for s in story['shots'] for i in range(len(s['action_beats']))]
    assert [u for group in groups.values() for u in group] == expected
    count = len(sdk.calls)
    assert ensure_planning_semantics(story, tmp_path, save) == bound
    assert len(sdk.calls) == count


def test_uncertain_batch_stays_uncertain_and_does_not_restart_or_merge_as_passed(tmp_path, monkeypatch):
    goal = 'Partners maintain identity and continuity throughout all exchanges.'
    story = batch_story(goal)
    sdk = Reviewer(rejected_batch=2)
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **_: sdk)
    save = lambda name, data: RunWorkspace._atomic_write_json(tmp_path / name, data)
    for _ in range(2):
        with pytest.raises(ValueError, match='证据不足或矛盾'):
            ensure_planning_semantics(story, tmp_path, save)
    assert len(sdk.calls) == 2
    assert not (tmp_path / 'planning_semantic_review.json').exists()
    record = json.loads((tmp_path / 'planning_review_batches/002/planning_semantic_review.json').read_text())
    assert record['verdict']['goal_allocation']['/intent/completion_goals/0']['parts'][0]['relation'] == 'undetermined'
