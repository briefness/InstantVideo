"""非均分时长、独立观察走正式生成到导出；所有远程服务为显式离线替身。"""
import json
import subprocess

import pytest

from test_scheduled_production_flow import scheduled_services
from test_creation_schedule_flow import scenario_story, SCENARIOS
from test_director_intent import Responses
from pipeline.orchestrator import VideoPipeline
from pipeline.run_state import RunWorkspace
from pipeline.support_witness import CURRENT_POLICY
from tools.ffmpeg_ops import get_video_duration


@pytest.mark.asyncio
@pytest.mark.parametrize('topic', SCENARIOS)
async def test_content_duration_reaches_submission_review_export_and_resume(scheduled_services, tmp_path, topic):
    _, services = scheduled_services(topic)
    brief, value, authored, _ = scenario_story(topic)
    brief = brief.replace('15秒', '16秒')
    value.update(duration_mode='approximate', requested_duration_seconds=16,
                 duration_evidence='16秒', intent_evidence=brief, pacing='balanced')
    value['completion_goals'][0]['evidence'] = brief
    value['completion_goals'][0]['goal'] = brief
    authored['shot_durations'] = [5, 6, 6]
    services.author = Responses(value, authored)
    sources = []
    for index, seconds in enumerate([5, 6, 6]):
        path = tmp_path/f'flex-{index}.mp4'
        subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i',
            f'testsrc2=s=864x496:r=24:d={seconds+.125}', '-c:v','libx264',
            '-preset','ultrafast','-pix_fmt','yuv420p',str(path)], check=True)
        sources.append(path)
    services.sources = sources
    pipeline = VideoPipeline(resolution='480p', platforms=['youtube'], paid_take_budget=3, take_policy='single_pass')
    output = await pipeline.run(brief)
    assert [r['duration'] for r in services.submissions] == [5, 6, 6]
    assert get_video_duration(output) == pytest.approx(17, abs=1/24)
    frozen = json.loads((pipeline.workspace/'storyboard.json').read_text())
    assert frozen['production_plan']['target_duration'] == 16
    assert frozen['production_plan']['planned_duration'] == 17
    records = [json.loads(p.read_text()) for p in (pipeline.workspace/'semantic_reviews/support').glob('*.json')]
    assert len(records) == 3
    for record in records:
        payload = record['input']['payload']
        assert payload['evidence_policy'] == CURRENT_POLICY
        assert all('observation' not in c for c in payload['cases'].values())
        assert all(row['witness']['finding'] for row in record['proof']['decisions'].values())
    subprocess.run(['ffmpeg','-v','error','-i',output,'-f','null','-'],check=True)
    counts = (len(services.submissions), len(services.author.calls), len(services.reviews))
    assert await VideoPipeline.from_workspace(pipeline.workspace).run() == output
    assert counts == (len(services.submissions), len(services.author.calls), len(services.reviews))
    assert RunWorkspace.resume(pipeline.workspace).manifest.status.value == 'succeeded'
