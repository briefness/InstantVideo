"""审查适用性由实际输入决定，不能把缺少上镜与当前画面不可见混为一谈。"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from pipeline.semantic_review import SemanticReviewUnavailableError
from test_sequence_review import reviewer_for, offline_frames, process_shot, event_response


@pytest.mark.parametrize('mode', ['direct_contact', 'directed_path', 'area_effect', 'indirect_effect'])
@pytest.mark.parametrize('boundary', [None, False])
def test_no_reference_does_not_request_or_consume_a_boundary_judgment(tmp_path, mode, boundary):
    response = event_response()
    response['boundary_continuity_valid'] = boundary
    original = deepcopy(response)
    video, reviewer, sdk = reviewer_for(tmp_path, response)
    assert reviewer.review(str(video), process_shot(mode)).accepted
    schema = sdk.calls[0]['response_format']['json_schema']['schema']
    assert 'boundary_continuity_valid' not in schema['properties']
    assert response == original and len(sdk.calls) == 1


def test_original_record_reaches_timing_diagnostic_without_becoming_accepted(tmp_path, monkeypatch):
    capture = json.loads((Path(__file__).parent / 'fixtures/real_first_shot_observation_failure.json').read_text())
    response = capture['evidence']['response']
    before = deepcopy(response)
    video, reviewer, sdk = reviewer_for(tmp_path, response)
    monkeypatch.setattr('tools.ffmpeg_ops.get_video_info', lambda _: {
        'streams': [{'codec_type': 'video', 'duration': '5', 'avg_frame_rate': '24/1'}]})
    with pytest.raises(SemanticReviewUnavailableError, match='时间证据'):
        reviewer.review(str(video), capture['shot'])
    assert response == before and len(sdk.calls) == 1
    files = list((tmp_path / 'semantic_reviews').glob('*.json'))
    assert files and all(f.name.endswith('.evidence.json') for f in files)


@pytest.mark.parametrize('boundary', [None, False])
def test_real_boundary_unknown_stops_and_known_failure_remains_failed(tmp_path, boundary):
    response = event_response()
    response.update(boundary_continuity_valid=boundary, environment_continuity_valid=True,
        action_handoff_valid=True, screen_direction_valid=True, prop_continuity_valid=True)
    video, reviewer, sdk = reviewer_for(tmp_path, response)
    previous = tmp_path / 'tail.jpg'
    previous.write_bytes(b'previous observation')
    if boundary is None:
        with pytest.raises(SemanticReviewUnavailableError, match='观察不足|看不清'):
            reviewer.review(str(video), process_shot('direct_contact'), previous_frame_path=str(previous))
    else:
        assert not reviewer.review(str(video), process_shot('direct_contact'), previous_frame_path=str(previous)).accepted
    assert len(sdk.calls) == 1
