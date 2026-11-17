"""隔离原观察污染；真实记录只回放，合成标注不冒充视频质量。"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from test_contribution_witness import frames, opposite_motion, offline
from test_support_evidence_boundary import scene
from test_director_intent import Responses

POLICY = 'grounded-requirements-v2'
ROOT = Path(__file__).parents[1]
RUN = ROOT / 'output/20261117_104852_942983'


def new_input():
    catalog, raw = scene()
    catalog['evidence_policy'] = POLICY
    return catalog, raw


def answer(status='fulfilled', issue='none', finding='物体从手中释放并落在指定支撑面，手随后离开。'):
    return {'decisions': {'g': {'requirement': status, 'issue': issue,
        'finding': finding, 'evidence': ['current_clip:1', 'current_clip:2']}}}


def test_second_observer_is_blind_to_first_claim_and_its_chosen_frames():
    from pipeline.observation_support import support_input
    catalog, raw = new_input()
    first = support_input(catalog, raw)
    raw['observations']['g'].update(observation='虚构的精确手腕角度和招式', sample_indices=[999])
    raw['end_state_context']['subject'] = '另一段未经核实的陈述'
    assert support_input(catalog, raw) == first
    assert set(first['cases']['g']) == {'requirement', 'requirement_context'}
    assert 'observed_context' not in first


@pytest.mark.parametrize('status,issue,expected', [
    ('fulfilled', 'none', 'satisfied'), ('unfulfilled', 'requirement_gap', 'violated'),
    ('unresolved', 'sampling_gap', None), ('unresolved', 'scope_ambiguity', None),
])
def test_same_call_owns_grounded_finding_and_requirement(tmp_path, frames, status, issue, expected):
    from pipeline.observation_support import verify_support, SupportJudgmentError
    catalog, raw = new_input()
    original = deepcopy(raw)
    client = Responses(answer(status, issue))
    args = dict(client=client, model='offline', cache_dir=tmp_path,
        video_hash=frames['manifest']['sources'][0]['video_sha256'], visual_evidence=frames)
    if expected is None:
        for _ in range(2):
            with pytest.raises(SupportJudgmentError):
                verify_support(catalog, raw, **args)
    else:
        checked, proof = verify_support(catalog, raw, **args)
        assert checked['observations']['g']['status'] == expected
        assert checked['observations']['g']['observation'] == answer()['decisions']['g']['finding']
        assert checked['observations']['g']['sample_indices'] == [1, 2]
        assert proof['decisions']['g']['observation_quote'] == raw['observations']['g']['observation']
    assert len(client.calls) == 1 and raw == original


@pytest.mark.parametrize('change', [
    {'finding': ''}, {'finding': ' '}, {'finding': 'x' * 481},
    {'evidence': []}, {'evidence': ['current_clip:999']},
    {'requirement': 'fulfilled', 'issue': 'sampling_gap'},
    {'requirement': 'unresolved', 'issue': 'requirement_gap'},
    {'observation': 'verified'},
])
def test_no_pass_from_missing_or_contradictory_evidence(change):
    from pipeline.support_witness import decode_witnesses
    from test_contribution_witness import payload
    task = {**payload(), 'evidence_policy': POLICY}
    value = answer()
    value['decisions']['g'].update(change)
    with pytest.raises(ValueError):
        decode_witnesses(value, task)


def test_latest_real_record_keeps_its_original_unknown():
    from pipeline.causality import compile_action_contract
    from pipeline.observation_support import support_input, apply_support, SupportJudgmentError
    saved = json.loads(next((RUN/'semantic_reviews/support').glob('*.json')).read_text())
    evidence = json.loads(next((RUN/'semantic_reviews').glob('*.evidence.json')).read_text())
    story = json.loads((RUN/'result_admission.json').read_text())['execution_storyboard']
    catalog = compile_action_contract(story['shots'][0]).review_projection['completion_contract']
    task = support_input(catalog, evidence['response'])
    assert task['cases'] == saved['input']['payload']['cases']
    manifest = saved['input']['payload']['visual_evidence']
    assert evidence['sample_timestamps'] == [f['timestamp'] for f in manifest['sources'][0]['frames']]
    with pytest.raises(SupportJudgmentError, match='关键画面含混'):
        apply_support(catalog, evidence['response'], saved['proof'], visual_manifest=manifest)
    # 新协议仅构造另一个请求，不能用旧响应冒充独立观察或写回原任务。
    catalog['evidence_policy'] = POLICY
    new = support_input(catalog, evidence['response'])
    assert new['cases'].keys() == task['cases'].keys()
    assert all(new['cases'][k]['requirement_context'] == task['cases'][k]['requirement_context'] for k in task['cases'])
    assert all('observation' not in c for c in new['cases'].values())


@pytest.mark.parametrize('count', [1, 2, 4, 8, 12])
def test_finding_schema_request_and_budget_share_one_capacity(count):
    from pipeline.support_witness import decision_schema, output_budget, finding_limit
    from test_contribution_witness import payload
    task = {**payload(), 'evidence_policy': POLICY}
    task['cases'] = {f'goal-{i}':{} for i in range(count)}
    schema = decision_schema(task)
    assert 32 <= finding_limit(task) <= 480
    assert output_budget(task) <= 8192
    assert all(c['properties']['finding']['maxLength'] == finding_limit(task)
               for c in schema['properties']['decisions']['properties'].values())


def test_latest_original_video_and_responses_replay_without_rejudgment(tmp_path):
    from types import SimpleNamespace as NS
    import hashlib
    from pipeline.semantic_review import SemanticTakeReviewer, SemanticReviewUnavailableError
    from test_full_production_flow import completion
    support_path = next((RUN/'semantic_reviews/support').glob('*.json'))
    evidence_path = next((RUN/'semantic_reviews').glob('*.evidence.json'))
    paths = [support_path, evidence_path, RUN/'result_admission.json', RUN/'shots/shot_001.mp4']
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    saved, evidence, story = [json.loads(p.read_text()) for p in paths[:3]]
    calls = []
    def create(**request):
        calls.append(request)
        user = request['messages'][-1]['content']
        if isinstance(user, list):
            return completion(evidence['response'])
        task = json.loads(user)
        assert task['cases'] == saved['input']['payload']['cases']
        assert task['visual_evidence'] == saved['input']['payload']['visual_evidence']
        return completion(json.loads(saved['response']['choices'][0]['content']))
    reviewer = SemanticTakeReviewer(tmp_path, model='offline-replay', client=NS(chat=NS(completions=NS(create=create))))
    shot = story['execution_storyboard']['shots'][0]
    for _ in range(2):
        with pytest.raises(SemanticReviewUnavailableError, match='关键画面含混'):
            reviewer.review(str(paths[-1]), shot, identity_crop_entities=shot['characters'])
    assert len(calls) == 2
    assert hashes == [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]

