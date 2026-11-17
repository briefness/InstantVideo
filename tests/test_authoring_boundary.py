import json
from pathlib import Path

import pytest

from pipeline.action_authoring import AuthoredShot, compile_action_units
from pipeline.storyboard import generate_storyboard
from authoring_cases import authoring_from_legacy
from test_authoring_transport import http_client
from test_director_intent import draft, intent_reply


def test_real_cross_shot_overlap_remains_rejected_without_rewriting():
    record = json.loads((Path(__file__).parent / 'fixtures/real_cross_shot_overlap_failure.json').read_text())
    shot = record['shot']
    assert shot['action_plan'][0]['overlap'] is True
    with pytest.raises(ValueError, match='搭接必须来自相邻'):
        compile_action_units(AuthoredShot.model_validate(shot), {
            'duration': 5, 'min_interaction_events': 3, 'min_interaction_seconds': 3.5,
        })
    assert shot['action_plan'][0]['overlap'] is True


def test_formal_request_distinguishes_cross_shot_carry_from_in_shot_overlap(monkeypatch):
    from pipeline.storyboard_authoring import generate_event_storyboard
    from test_action_authoring import plan_for
    raw = authoring_from_legacy(draft())
    # carry 是历史合同字段，使用冻结计划验证旧作者接口的实际 HTTP 请求。
    values = [{k: v for k, v in raw.items() if k != 'shots'}]
    values += [{'shot': {k: v for k, v in s.items() if k != 'shot_id'}} for s in raw['shots']]
    sdk, calls = http_client(monkeypatch, values)
    with sdk:
        generate_event_storyboard(sdk, 'continuous action', plan_for(),
            aspect_ratio='16:9', resolution='720p', style='cinematic')
    for call in calls[1:]:
        schema = call['response_format']['json_schema']['schema']
        timeline = schema['properties']['shot']['properties']['action_plan']
        first = timeline['prefixItems'][0]['oneOf']
        for variant in first[:-1]:
            assert variant['properties']['overlap'] == {'type': 'boolean', 'const': False}
            assert 'carry' in variant['properties']
        assert first[-1] == {'$ref': '#/$defs/PassiveUnit'}
        assert all('const' not in v['properties']['overlap']
                   for v in schema['$defs']['InteractionUnit']['oneOf'])
