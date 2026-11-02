"""Inspect actual OpenAI SDK HTTP bytes; never contact an external service."""
import json
from copy import deepcopy
from pathlib import Path

import httpx
from openai import OpenAI, BadRequestError
import pytest

from pipeline.llm_options import ark_output_schema
from pipeline.storyboard import generate_storyboard
from authoring_cases import authoring_from_legacy
from test_director_intent import draft, intent_reply
from structured_transport_cases import completion_response


def http_client(monkeypatch, responses):
    calls = []
    values = iter(responses)

    def receive(request):
        body = json.loads(request.content)
        calls.append(body)
        value = next(values)
        if isinstance(value, httpx.Response):
            return value
        return completion_response(value, body)

    sdk = OpenAI(api_key='offline-key', base_url='https://offline.invalid/v1',
                 http_client=httpx.Client(transport=httpx.MockTransport(receive)))
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **kw: sdk)
    return sdk, calls


def test_formal_authoring_sends_strict_schema_in_actual_http_body(monkeypatch):
    raw = authoring_from_legacy(draft())
    values = [intent_reply('continuous action'), {k: v for k, v in raw.items() if k != 'shots'}]
    values += [{'shot': {k: v for k, v in s.items() if k != 'shot_id'}} for s in raw['shots']]
    sdk, calls = http_client(monkeypatch, values)
    with sdk:
        assert len(generate_storyboard('continuous action', target_duration=15)['shots']) == 3
    assert len(calls) == 5
    assert calls[0]['response_format']['json_schema']['name'] == 'director_intent'
    for call in calls:
        fmt = call['response_format']
        assert fmt['type'] == 'json_schema' and fmt['json_schema']['strict'] is True
        contract = fmt['json_schema']['schema']
        assert all(v['additionalProperties'] is False for v in contract.get('oneOf', [contract]))
    schema = calls[2]['response_format']['json_schema']['schema']
    assert schema['properties']['shot']['additionalProperties'] is False
    assert 'is_final' not in schema['properties']['shot']['properties']
    variants = schema['$defs']['InteractionUnit']['oneOf']
    assert all(v['properties']['actor']['enum'] == ['attacker', 'defender'] for v in variants)
    assert all('response' in v['required'] for v in variants)
    assert {v['properties']['response']['const']: v['properties']['effect']['properties']['mode']['enum']
            for v in variants} == {
        'impact': ['direct_contact', 'directed_path', 'area_effect', 'indirect_effect'],
        'evasion': ['directed_path'], 'deflection': ['direct_contact', 'directed_path'],
    }


def test_unsupported_schema_stops_without_json_mode_fallback(monkeypatch):
    refusal = httpx.Response(400, json={'error': {'message': 'json_schema unsupported', 'type': 'invalid_request_error'}})
    sdk, calls = http_client(monkeypatch, [intent_reply('continuous action'), refusal])
    with sdk, pytest.raises(BadRequestError):
        generate_storyboard('continuous action', target_duration=15)
    assert len(calls) == 2
    assert calls[-1]['response_format']['type'] == 'json_schema'


def test_real_protocol_violation_remains_rejected_even_if_provider_ignores_schema(monkeypatch):
    record = json.loads((Path(__file__).parent / 'fixtures/real_authoring_protocol_failure.json').read_text())
    original = deepcopy(record)
    sdk, calls = http_client(monkeypatch, [intent_reply('continuous action'), record['identity'], record['response']])
    with sdk, pytest.raises(ValueError, match='is_final'):
        generate_storyboard('continuous action', target_duration=15)
    assert len(calls) == 3 and record == original


def test_schema_projection_preserves_property_names_and_local_constraints():
    from pipeline.action_authoring import AuthoredShot
    schema = AuthoredShot.model_json_schema()
    original = deepcopy(schema)
    projected = ark_output_schema(schema)
    assert schema == original
    assert 'minItems' not in projected['properties']['action_plan']
    assert 'discriminator' not in projected['properties']['action_plan']['items']
    assert 'oneOf' in projected['properties']['action_plan']['items']
    assert 'required' in projected and '$defs' in projected
    unusual = {'type': 'object', 'properties': {'minimum': {'type': 'integer', 'minimum': 1}}}
    assert ark_output_schema(unusual)['properties'] == {'minimum': {'type': 'integer'}}
    assert original['properties']['action_plan']['minItems'] == 1  # Still enforced locally.
