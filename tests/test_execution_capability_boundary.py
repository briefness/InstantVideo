"""能力与创意维度独立；真实失败只读回放，所有外部模型禁止。"""
from copy import deepcopy
import json
from pathlib import Path
import pytest
import config

ROOT = Path(__file__).parents[1]/'output/20261113_132802_521726'

@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket,'connect',lambda *_:pytest.fail('offline only'))

def test_actual_compiled_plan_is_executable_without_rewriting_fast_cameras():
    from pipeline.storyboard import _validate_storyboard_richness
    story = json.loads((ROOT/'planning_compiled.json').read_text())
    before = deepcopy(story)
    saved = json.loads((ROOT/'planning_check.json').read_text())
    assert saved['critical'] and sum('稳定预算 5s' in x for x in saved['issues'])==3
    issues, critical = _validate_storyboard_richness(story)
    assert not critical, issues
    assert story == before

@pytest.mark.parametrize('seconds',[4,5,6,8,15])
@pytest.mark.parametrize('speed',['slow','fast','rapid','自然加速后跟随'])
def test_camera_words_do_not_change_capability_admission(seconds,speed):
    from pipeline.readiness import storyboard_readiness_issues
    story = json.loads((ROOT/'planning_compiled.json').read_text())
    story.pop('production_plan')
    story['shots'] = [story['shots'][0]]
    shot = story['shots'][0]
    shot['duration'] = seconds
    shot['camera']['speed'] = speed
    # 其他时间合同仍可能不满足；这里仅检查传输能力维度。
    issues = storyboard_readiness_issues(story)
    assert not any('provider' in x or 'generation profile' in x for x in issues)

def test_reduced_profile_fails_before_creative_call_and_never_shortens(monkeypatch):
    from pipeline.storyboard import generate_storyboard
    from test_director_intent import Responses, intent_reply
    chains = deepcopy(config.GENERATION_CHAINS)
    chains['480p'][0]['max_duration']=5
    monkeypatch.setattr(config,'GENERATION_CHAINS',chains)
    request='六秒展示产品'
    client=Responses(intent_reply(request,pacing='balanced'))
    monkeypatch.setattr('pipeline.storyboard.OpenAI',lambda **_:client)
    with pytest.raises(ValueError,match='能力无交集'):
        generate_storyboard(request,target_duration=6)
    assert len(client.calls)==1

@pytest.mark.parametrize('duration',[True,3,16,6.0,None])
def test_request_boundary_uses_one_strict_duration_contract(duration):
    from pipeline.provider_capabilities import execution_request_issues
    assert execution_request_issues(model=config.SEEDANCE_MODEL,resolution='480p',ratio='16:9',duration=duration)

def test_request_duration_cannot_disagree_with_frozen_shot():
    from pipeline.readiness import provider_submission_issues
    issues=provider_submission_issues('visible action',{'shot_id':1,'duration':6},{},
        model=config.SEEDANCE_MODEL,resolution='480p',duration=5,prompt_profile='policy_safe')
    assert any('计划时长' in x for x in issues)

@pytest.mark.asyncio
async def test_sdk_cannot_bypass_selected_profile(monkeypatch):
    from tools.seedance_api import SeedanceAPI
    chains=deepcopy(config.GENERATION_CHAINS)
    chains['480p'][0]['max_duration']=5
    monkeypatch.setattr(config,'GENERATION_CHAINS',chains)
    api=object.__new__(SeedanceAPI)
    async def forbidden(**kwargs): pytest.fail('must reject before provider')
    api._generate_ark=forbidden
    with pytest.raises(ValueError,match='generation profile'):
        await api.generate('test',duration=6,model=config.SEEDANCE_MODEL,resolution='480p')


def test_six_original_responses_reach_formal_compilation_unchanged(monkeypatch):
    from types import SimpleNamespace as NS
    from pipeline.storyboard import generate_storyboard
    names=['planning_intent_response.json','planning_identity_response.json',
           *[f'planning_shot_{i:03d}_response.json' for i in range(1,5)]]
    records=[json.loads((ROOT/n).read_text()) for n in names]
    calls=[]
    def create(**request):
        record=records[len(calls)]
        calls.append(request)
        return NS(choices=[NS(finish_reason=record['finish_reason'],message=NS(content=record['content'],refusal=None))])
    monkeypatch.setattr('pipeline.storyboard.OpenAI',lambda **_:NS(chat=NS(completions=NS(create=create))))
    options=json.loads((ROOT/'run_manifest.json').read_text())['options']
    result=generate_storyboard(options['request'],resolution=options['resolution'],aspect_ratio=options['aspect_ratio'],
        style=options['style'],allow_repair=False)
    original=json.loads((ROOT/'planning_compiled.json').read_text())
    # 新编译器选用独立核对协议；旧原稿与动作/摄影/时长保持逐字段相同。
    for shot in original['shots']:
        assert shot['evidence_policy'] == 'witness-decisions-v1'
        shot['evidence_policy'] = 'grounded-requirements-v2'
        assert 'execution_semantics' not in shot
        shot['execution_semantics'] = 'intent-bounded-v1'
    # 新规划记录能力快照；原始创作、动作、摄影和实际时长不变。
    assert result['production_plan']['duration_limits'] == {'min_duration': 4, 'max_duration': 15}
    original['production_plan']['duration_limits'] = result['production_plan']['duration_limits']
    assert result==original
    assert len(calls)==6
    contexts=[json.loads(r['messages'][-1]['content']) for r in calls[1:]]
    assert all(c['execution_capabilities']==contexts[0]['execution_capabilities'] for c in contexts)
    assert contexts[0]['execution_capabilities']['hard_limits']['max_duration']==15


@pytest.mark.asyncio
async def test_executor_does_not_truncate_to_fit_a_fallback(tmp_path, monkeypatch):
    from pipeline.generator import VideoGenerator
    from pipeline.readiness import GenerationReadinessError
    chains=deepcopy(config.GENERATION_CHAINS)
    chains['480p'][0]['max_duration']=5
    monkeypatch.setattr(config,'GENERATION_CHAINS',chains)
    g=VideoGenerator(str(tmp_path))
    async def forbidden(**kwargs): pytest.fail('must stop before paid submission')
    g.api.generate=forbidden
    shot={'shot_id':1,'duration':6,'prompt_en':'A single object rolls on a table.',
          'characters':[],'camera':{'primary_movement':'tracking','speed':'fast'}}
    with pytest.raises(GenerationReadinessError,match='generation profile'):
        await g._generate_single_shot(shot,None,None,{'shots':[shot],'characters':[],'resolution':'480p'})
    assert not list(tmp_path.glob('submission_requests/*.json'))


from test_scheduled_production_flow import scheduled_services

@pytest.mark.asyncio
@pytest.mark.parametrize('topic',['双人舞','货物交接','绳索救援'])
async def test_six_second_fast_camera_cross_scene_export_and_resume(scheduled_services,topic,tmp_path):
    import subprocess
    from test_creation_schedule_flow import scenario_story
    from test_director_intent import Responses, intent_reply
    from pipeline.orchestrator import VideoPipeline
    from tools.ffmpeg_ops import get_video_info
    brief,services=scheduled_services(topic)
    _,intent,story,_=scenario_story(topic)
    brief=f'18秒，分三段，连续展示{topic}。最后完成{topic}并清楚展示结果。'
    intent={**intent_reply(brief,pacing='balanced'),'choreography':'flexible',
            'ending_requirement':intent['ending_requirement'],'ending_evidence':intent['ending_evidence'],
            'requested_duration_seconds':18,'duration_evidence':'18秒',
            'requested_shot_count':3,'shot_count_evidence':'三段'}
    for shot in story['shots']:
        shot['camera']['speed']='fast'
    services.author=Responses(intent,story)
    sources=[]
    for index,source in enumerate(services.sources):
        path=tmp_path/f'extended-{index}.mp4'
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(source),'-vf','tpad=stop_mode=clone:stop_duration=2',
                        '-af','apad','-t','6.125',str(path)],check=True)
        sources.append(path)
    services.sources=sources
    pipeline=VideoPipeline(resolution='480p',platforms=['youtube'],paid_take_budget=3,take_policy='single_pass')
    output=await pipeline.run(brief)
    assert [r['duration'] for r in services.submissions]==[6,6,6]
    frozen=json.loads((pipeline.workspace/'storyboard.json').read_text())
    assert all(s['camera']['speed']=='fast' for s in frozen['shots'])
    v=next(s for s in get_video_info(output)['streams'] if s['codec_type']=='video')
    assert int(v['nb_frames'])==432
    counts=(len(services.author.calls),len(services.submissions),len(services.reviews))
    assert await VideoPipeline.from_workspace(pipeline.workspace).run()==output
    assert counts==(len(services.author.calls),len(services.submissions),len(services.reviews))


@pytest.mark.parametrize('field,value',[('resolution',[]),('resolution',None),('model',True),
    ('model',''),('model','foreign-model'),('ratio',{}),('ratio','100:1')])
def test_invalid_identifiers_fail_closed(field,value):
    from pipeline.provider_capabilities import execution_request_issues
    args=dict(model=config.SEEDANCE_MODEL,resolution='480p',ratio='16:9',duration=6)
    args[field]=value
    assert execution_request_issues(**args)


@pytest.mark.parametrize('tempo',['fast','natural','deliberate'])
@pytest.mark.parametrize('playback',['real_time','slow_motion'])
def test_performer_tempo_playback_and_camera_are_independent(tempo,playback):
    from pipeline.motion_timing import motion_timing_projection
    from pipeline.storyboard import _validate_storyboard_richness
    story=json.loads((ROOT/'planning_compiled.json').read_text())
    for shot in story['shots']:
        slot=shot['production_slot']
        slot['creative_brief']='A deliberate choice of motion and playback.'
        slot['motion_timing']={'performer_tempo':tempo,'playback':playback,'evidence':slot['creative_brief']}
        before=deepcopy(slot['motion_timing'])
        assert motion_timing_projection(shot)==before
    # 仅变速维度，镜头时长合法性不变化；不据标签证明真实速度。
    warnings,_=_validate_storyboard_richness(story)
    assert not any('稳定预算' in x or 'provider 时长' in x for x in warnings)
