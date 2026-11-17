"""真实响应不改判；新版独立标注只验证合同与证据的可追溯性。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket, 'connect', lambda *_: pytest.fail('offline only'))
    monkeypatch.setattr(socket.socket, 'connect_ex', lambda *_: pytest.fail('offline only'))


def capture():
    fixture = json.loads((Path(__file__).parent/'fixtures/real_contribution_witness_failure.json').read_text())
    records = {}
    for name, source in fixture['sources'].items():
        assert hashlib.sha256(source['text'].encode()).hexdigest() == source['sha256']
        records[name] = json.loads(source['text'])
    story = records['result_admission.json']['execution_storyboard']
    support = next(v for k,v in records.items() if '/support/' in k)
    evidence = next(v for k,v in records.items() if k.endswith('.evidence.json'))
    return fixture, story, support, evidence


def test_actual_failure_has_same_scope_and_sampling_but_no_explanatory_witness():
    from pipeline.causality import compile_action_contract
    from pipeline.observation_support import support_input, apply_support, SupportJudgmentError
    _, story, saved, evidence = capture()
    original = deepcopy((story,saved,evidence))
    catalog = compile_action_contract(story['shots'][0]).review_projection['completion_contract']
    payload = support_input(catalog,evidence['response'])
    assert payload['cases'] == saved['input']['payload']['cases']
    manifest = saved['input']['payload']['visual_evidence']
    assert evidence['sample_timestamps'] == [f['timestamp'] for f in manifest['sources'][0]['frames']]
    with pytest.raises(SupportJudgmentError, match='语义不确定'):
        apply_support(catalog,evidence['response'],saved['proof'],visual_manifest=manifest)
    assert all(isinstance(v,str) for v in json.loads(saved['response']['choices'][0]['content'])['decisions'].values())
    assert (story,saved,evidence) == original


def test_new_generation_keeps_the_same_required_goal_as_its_audit():
    from pipeline.causality import compile_action_contract
    from pipeline.generation_instructions import local_goal_instructions
    from pipeline.observation_support import support_input
    _, story, _, evidence = capture()
    catalog = compile_action_contract(story['shots'][0]).review_projection['completion_contract']
    catalog['evidence_policy'] = 'witness-decisions-v1'  # 独立协议迁移测试，不改历史工作区。
    task = support_input(catalog,evidence['response'])
    instructions = local_goal_instructions(catalog)
    for case in task['cases'].values():
        criterion = case['requirement_context']
        text = criterion.get('goal_context',{}).get('action',criterion.get('action'))
        assert any(text in line for line in instructions)
    local = next(c['requirement_context'] for c in task['cases'].values() if c['requirement_context']['kind']=='contribution')
    assert not any('; '.join(u['action'] for u in local['execution_scope']['units']) == line for line in instructions)


def payload():
    return {'cases': {'g': {'sample_indices':[1,2], 'observation':'A hand redirects the moving object.'}},
        'visual_evidence': {'sources':[{'scope':'current_clip','frames':[
            {'index':1,'timestamp':0}, {'index':2,'timestamp':.25}, {'index':3,'timestamp':.5}]}]}}


@pytest.mark.parametrize('observation,requirement,issue,expected', [
    ('verified','fulfilled','none','supported'),
    ('verified','unfulfilled','requirement_gap','contradicted'),
    ('refuted','fulfilled','claim_conflict','unknown'),
    ('unresolved','unresolved','sampling_gap','unknown'),
])
def test_independent_facts_and_requirements_derive_one_final_verdict(observation,requirement,issue,expected):
    from pipeline.support_witness import decode_witnesses
    row = {'observation':observation,'requirement':requirement,'issue':issue,'evidence':['current_clip:1','current_clip:2']}
    result = decode_witnesses({'decisions':{'g':row}},payload())
    assert result['g']['verdict'] == expected and result['g']['witness'] == row


@pytest.mark.parametrize('change', [
    {'evidence':[]}, {'evidence':['current_clip:99']}, {'evidence':['previous_tail:1']},
    {'evidence':['current_clip:1','current_clip:1']}, {'issue':'sampling_gap'},
    {'requirement':'unfulfilled'}, {'observation':'refuted'}, {'unexpected':True},
])
def test_witness_cannot_claim_a_pass_without_valid_consistent_sources(change):
    from pipeline.support_witness import decode_witnesses
    row={'observation':'verified','requirement':'fulfilled','issue':'none','evidence':['current_clip:1']}
    with pytest.raises(ValueError):
        decode_witnesses({'decisions':{'g':{**row,**change}}},payload())


from test_temporal_evidence import opposite_motion


@pytest.fixture(scope='module')
def frames(opposite_motion):
    from pipeline.temporal_evidence import sample_clip
    return sample_clip(str(opposite_motion[0]), {}, scope='current_clip')


def inputs():
    from test_support_evidence_boundary import scene
    from pipeline.support_witness import POLICY
    catalog, raw = scene()
    catalog['evidence_policy'] = POLICY
    return catalog, raw


def response(observation='verified', requirement='fulfilled', issue='none'):
    return {'decisions':{'g':{'observation':observation,'requirement':requirement,
                            'issue':issue,'evidence':['current_clip:1','current_clip:2']}}}


def test_wire_contract_and_cached_proof_bind_actual_images(tmp_path, frames):
    from test_director_intent import Responses
    from pipeline.observation_support import verify_support, apply_support
    from pipeline.support_witness import decision_schema
    from pipeline.llm_options import ark_output_schema
    catalog, raw = inputs()
    client = Responses(response())
    args = dict(client=client, model='doubao-seed-2.1-lite', cache_dir=tmp_path,
                video_hash=frames['manifest']['sources'][0]['video_sha256'], visual_evidence=frames)
    checked, proof = verify_support(catalog, raw, **args)
    request = client.calls[0]
    sent = json.loads(request['messages'][-1]['content'])
    assert request['response_format']['json_schema']['schema'] == ark_output_schema(decision_schema(sent))
    assert request['messages'][1]['content'] == frames['content']
    assert '返回contradicted' not in request['messages'][0]['content']
    assert request['extra_body']['thinking']['type'] == 'disabled'
    assert request['max_completion_tokens'] <= 8192
    assert verify_support(catalog, raw, **args) == (checked, proof)
    assert checked == raw and len(client.calls) == 1
    for malformed in (None, [], 'supported', {**proof['decisions']['g'], 'verdict':'contradicted'},
                      {**proof['decisions']['g'], 'witness':{**response()['decisions']['g'], 'evidence':['current_clip:999']}}):
        bad = deepcopy(proof)
        bad['decisions']['g'] = malformed
        with pytest.raises(ValueError):
            apply_support(catalog, raw, bad, visual_manifest=frames['manifest'])


@pytest.mark.parametrize('fact,requirement,issue,code',[
    ('unresolved','unresolved','sampling_gap','semantic_uncertainty'),
    ('refuted','fulfilled','claim_conflict','evidence_conflict'),
])
def test_unknown_is_explained_and_recovery_never_requeries(tmp_path, frames, fact, requirement, issue, code):
    from test_director_intent import Responses
    from pipeline.observation_support import verify_support, SupportJudgmentError, assert_support_available
    from pipeline.run_state import RunWorkspace
    catalog, raw = inputs()
    client = Responses(response(fact,requirement,issue))
    video_hash = frames['manifest']['sources'][0]['video_sha256']
    for model in ('offline-v1','offline-v2'):
        with pytest.raises(SupportJudgmentError) as caught:
            verify_support(catalog, raw, client=client, model=model, cache_dir=tmp_path,
                           video_hash=video_hash, visual_evidence=frames)
        assert caught.value.code == code
        assert caught.value.details['g']['witness']['issue'] == issue
    assert len(client.calls) == 1
    path = next((tmp_path/'support').glob('*.json'))
    saved = json.loads(path.read_text())
    assert saved['state'] == 'returned' and 'error_type' not in saved
    # 模拟完整原响应已落盘、最终assessment尚未落盘的崩溃窗口。
    saved.pop('assessment')
    saved.pop('proof')
    saved['state'] = 'started'
    RunWorkspace._atomic_write_json(path, saved)
    with pytest.raises(SupportJudgmentError):
        assert_support_available(tmp_path, video_hash)
    assert len(client.calls) == 1


def test_visible_requirement_failure_cannot_be_relabelled_as_pass(tmp_path, frames):
    from test_director_intent import Responses
    from pipeline.observation_support import verify_support, apply_support, SupportJudgmentError
    catalog, raw = inputs()
    checked, _ = verify_support(catalog, raw, client=Responses(response('verified','unfulfilled','requirement_gap')),
        model='offline', cache_dir=tmp_path/'negative',
        video_hash=frames['manifest']['sources'][0]['video_sha256'], visual_evidence=frames)
    assert checked['observations']['g']['status'] == 'violated'
    raw['observations']['g']['status'] = 'violated'
    with pytest.raises(SupportJudgmentError, match='观察判定冲突'):
        verify_support(catalog, raw, client=Responses(response()), model='offline', cache_dir=tmp_path/'conflict',
            video_hash=frames['manifest']['sources'][0]['video_sha256'], visual_evidence=frames)


def test_latest_raw_review_replays_and_resume_preserves_unknown(tmp_path):
    from pipeline.semantic_review import SemanticTakeReviewer, SemanticReviewUnavailableError
    from test_full_production_flow import completion
    fixture, story, saved, evidence = capture()
    root = Path(__file__).parents[1]/fixture['source_workspace']
    video = root/'shots/shot_001.mp4'
    assert hashlib.sha256(video.read_bytes()).hexdigest() == fixture['video']['sha256']
    calls = []
    def create(**request):
        calls.append(request)
        user = request['messages'][-1]['content']
        if isinstance(user,list):
            return completion(deepcopy(evidence['response']))
        task = json.loads(user)
        assert task['cases'] == saved['input']['payload']['cases']
        assert task['visual_evidence'] == saved['input']['payload']['visual_evidence']
        return completion(json.loads(saved['response']['choices'][0]['content']))
    reviewer = SemanticTakeReviewer(tmp_path, model='offline-real-replay',
        client=NS(chat=NS(completions=NS(create=create))))
    for _ in range(2):
        with pytest.raises(SemanticReviewUnavailableError, match='语义不确定'):
            reviewer.review(str(video),story['shots'][0], identity_crop_entities=story['shots'][0]['characters'])
    assert len(calls) == 2
    record = json.loads(next(reviewer.cache_dir.glob('*.evidence.json')).read_text())
    assert record['failure_kind'] == 'semantic_uncertainty' and record['response'] == evidence['response']
    for name, source in fixture['sources'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == source['sha256']


def test_formal_compilation_preserves_joint_goal_without_expanding_local_work():
    from test_staging_direction import compile_story
    from pipeline.result_admission import apply_result_admission
    from pipeline.provider_prompt import compile_normal_provider_prompt
    from pipeline.causality import compile_action_contract
    from pipeline.observation_support import support_input
    fixture, old, _, evidence = capture()
    root = Path(__file__).parents[1]/fixture['source_workspace']
    authored = json.loads((root/'planning_authoring.json').read_text())
    admission = json.loads(fixture['sources']['result_admission.json']['text'])
    before = deepcopy(authored)
    story = apply_result_admission(compile_story(authored,old['production_plan']),admission['units'])
    assert authored == before
    for prior, new in zip(old['shots'],story['shots']):
        assert prior['action_beats'] == new['action_beats']
    catalog = compile_action_contract(story['shots'][0]).review_projection['completion_contract']
    task = support_input(catalog,evidence['response'])
    assert catalog['evidence_policy']=='grounded-requirements-v2'
    prompt = compile_normal_provider_prompt(story['shots'][0],story,has_observed_start=False)
    function_section = prompt.split('Required observable functions: ',1)[1].split('Play this ',1)[0]
    contribution = next(c['requirement_context'] for c in task['cases'].values()
                        if c['requirement_context']['kind']=='contribution')
    assert contribution['goal_context']['action'] in function_section
    assert all(u['shot_id']==1 for u in contribution['execution_scope']['units'])
    original_joint = next(c for c in catalog['criteria'].values() if c.get('execution_scope'))
    for unit in original_joint['execution_scope']['units']:
        if unit['shot_id'] != 1:
            assert unit['action'] not in function_section


@pytest.mark.parametrize('issue', ['sampling_gap','claim_conflict'])
def test_crash_recovery_reconstructs_specific_reason_from_raw_witness(tmp_path, frames, issue):
    from pipeline.observation_support import support_input, assert_support_available, SupportJudgmentError
    from pipeline.temporal_evidence import VISUAL_SUPPORT
    from pipeline.run_state import RunWorkspace
    catalog, raw = inputs()
    task = support_input(catalog,raw)
    task.update(protocol=VISUAL_SUPPORT,visual_evidence=frames['manifest'])
    body = response('refuted' if issue=='claim_conflict' else 'unresolved','unresolved',issue)
    (tmp_path/'support').mkdir()
    RunWorkspace._atomic_write_json(tmp_path/'support/captured.json', {
        'state':'started','input':{'video_hash':'raw-film','payload':task},
        'response':{'choices':[{'finish_reason':'stop','refused':False,'content':json.dumps(body)}]}})
    with pytest.raises(SupportJudgmentError) as caught:
        assert_support_available(tmp_path,'raw-film')
    assert caught.value.details['g']['witness']==body['decisions']['g']
    assert caught.value.code==('evidence_conflict' if issue=='claim_conflict' else 'semantic_uncertainty')


def test_capacity_and_missing_pixels_fail_before_model_call(tmp_path, frames):
    from pipeline.observation_support import verify_support
    catalog, raw = inputs()
    with pytest.raises(ValueError,match='实际原片'):
        verify_support(catalog,raw,client=None,model='offline',cache_dir=tmp_path,video_hash='raw-film')
    criterion = catalog['criteria'].pop('g')
    witness = raw['observations'].pop('g')
    catalog['criteria']={'x'*10000:criterion}
    raw['observations']={'x'*10000:witness}
    with pytest.raises(ValueError,match='容量'):
        verify_support(catalog,raw,client=None,model='offline',cache_dir=tmp_path,
            video_hash=frames['manifest']['sources'][0]['video_sha256'],visual_evidence=frames)
    assert not (tmp_path/'support').exists()
