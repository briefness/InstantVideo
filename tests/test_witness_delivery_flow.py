"""正式编排的离线服务替身；测试来源、停止和交付，不证明生成画质。"""
import json
import subprocess

import pytest

from pipeline.orchestrator import VideoPipeline
from pipeline.run_state import RunWorkspace
from pipeline.support_witness import CURRENT_POLICY as POLICY
from test_scheduled_production_flow import scheduled_services
from test_creation_schedule_flow import SCENARIOS


@pytest.mark.asyncio
@pytest.mark.parametrize('topic', SCENARIOS)
async def test_new_creation_carries_goal_and_witness_to_final_delivery(scheduled_services, topic):
    brief, services = scheduled_services(topic)
    calls = []
    original = services.create
    def create(**request):
        user = request['messages'][-1]['content']
        if isinstance(user,str):
            payload = json.loads(user)
            if payload.get('evidence_policy') == POLICY:
                calls.append(payload)
        return original(**request)
    services.create = create
    pipeline = VideoPipeline(resolution='480p', platforms=['youtube'], paid_take_budget=3, take_policy='single_pass')
    output = await pipeline.run(brief)
    assert len(calls) == len(services.submissions) == 3
    frozen = json.loads((pipeline.workspace/'storyboard.json').read_text())
    assert all(shot['evidence_policy'] == POLICY for shot in frozen['shots'])
    for task, submitted in zip(calls,services.submissions):
        for case in task['cases'].values():
            criterion = case['requirement_context']
            if criterion['kind'] in ('completion','contribution'):
                goal = criterion.get('goal_context',{}).get('action',criterion.get('action'))
                assert goal in submitted['prompt']
    supports = [json.loads(p.read_text()) for p in (pipeline.workspace/'semantic_reviews/support').glob('*.json')]
    assert len(supports) == 3
    assert all(row['witness']['finding'] and row['witness']['requirement']=='fulfilled'
               for saved in supports for row in saved['proof']['decisions'].values())
    subprocess.run(['ffmpeg','-v','error','-i',output,'-f','null','-'],check=True)
    assert RunWorkspace.resume(pipeline.workspace).manifest.status.value == 'succeeded'
    counts = (len(calls),len(services.submissions),len(services.author.calls))
    assert await VideoPipeline.from_workspace(pipeline.workspace).run() == output
    assert counts == (len(calls),len(services.submissions),len(services.author.calls))


@pytest.mark.asyncio
async def test_unresolved_pixels_stop_once_and_survive_formal_resume(scheduled_services):
    from pipeline.semantic_review import SemanticReviewUnavailableError
    from test_full_production_flow import completion
    from functional_observation_cases import annotated_support_response
    brief, services = scheduled_services('绳索救援')
    calls = []
    original = services.create
    def create(**request):
        calls.append(request)
        user = request['messages'][-1]['content']
        if isinstance(user,str) and json.loads(user).get('evidence_policy') == POLICY:
            response = annotated_support_response(request)
            first = next(iter(response['decisions'].values()))
            first.update(requirement='unresolved',issue='sampling_gap', finding='采样没有显示必要接触到释放的过程。')
            return completion(response)
        return original(**request)
    services.create = create
    pipeline = VideoPipeline(resolution='480p', platforms=['youtube'], paid_take_budget=3, take_policy='single_pass')
    with pytest.raises(SemanticReviewUnavailableError,match='采样不足'):
        await pipeline.run(brief)
    assert len(services.submissions) == 1
    counts = len(calls),len(services.submissions)
    with pytest.raises(SemanticReviewUnavailableError,match='采样不足'):
        await VideoPipeline.from_workspace(pipeline.workspace).run()
    assert counts == (len(calls),len(services.submissions))
    manifest = RunWorkspace.resume(pipeline.workspace).manifest
    assert len(manifest.paid_take_budget.reservations)==1
    assert manifest.shots['1'].canonical_take_id is None
    assert not (pipeline.workspace/'final.mp4').exists()
    evidence = json.loads(next((pipeline.workspace/'semantic_reviews').glob('*.evidence.json')).read_text())
    assert evidence['failure_kind']=='semantic_uncertainty'
    assert any(v['witness']['issue']=='sampling_gap' for v in evidence['unresolved_evidence'].values())
