"""A captured cross-shot reference must not masquerade as an in-shot edge."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from pipeline.action_authoring import authored_shot_response_schema
from pipeline.authoring_context import SceneLocation, contextual_shot_schema
from pipeline.authoring_schedule import scheduled_shot_schema, bind_authored_schedule
from pipeline.llm_options import ark_output_schema
from pipeline.production_plan import build_production_plan


def test_original_cross_shot_link_is_invalid_in_both_wire_and_local_contract():
    record = json.loads((Path(__file__).parent / 'fixtures/real_entry_scope_failure.json').read_text())
    for source in record['sources'].values():
        assert hashlib.sha256(source['text'].encode()).hexdigest() == source['sha256']
    identity = json.loads(record['sources']['planning_identity.json']['text'])
    envelope = json.loads(record['sources']['planning_shot_003_response.json']['text'])
    assert envelope['finish_reason'] == 'stop'
    response = json.loads(envelope['content'])
    before = deepcopy(response)
    assert 'opening' not in response['shot']['action_plan']
    assert response['shot']['action_plan']['interaction_1']['link_previous'] is True
    plan = build_production_plan('action', 15, intent=record['intent'])
    slot = plan['slots'][-1]
    base = authored_shot_response_schema([c['name'] for c in identity['characters']],
        require_outcomes=True, scene_entities=identity['scene_entities'], bound_handoffs=True)
    schema = scheduled_shot_schema(contextual_shot_schema(base,
        [SceneLocation(**s) for s in identity['locations']], identity['scene_entities']), slot)
    for shape in (schema, ark_output_schema(schema)):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(response, shape)
    with pytest.raises(ValueError, match='前项'):
        bind_authored_schedule(response['shot'], slot)
    assert response == before


@pytest.mark.parametrize('has_opening', [False, True])
@pytest.mark.parametrize('linked', [False, True])
@pytest.mark.parametrize('overlap', [False, True])
def test_entry_scope_is_bound_before_provider_decoding(has_opening, linked, overlap):
    from test_authoring_schedule import raw_shot, passive, schema_for, slot
    from pipeline.action_authoring import AuthoredShot, compile_action_units
    raw = raw_shot()
    if has_opening:
        raw['action_plan']['opening'] = {'first': passive(), 'following': [passive('hold')]}
    raw['action_plan']['interaction_1'].update(link_previous=linked, overlap=overlap)
    valid = not overlap and (has_opening or not linked)
    for schema in (schema_for(slot()), ark_output_schema(schema_for(slot()))):
        wire = {'shot': {k: v for k, v in raw.items() if k != 'shot_id'}}
        if valid:
            jsonschema.validate(wire, schema)
        else:
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate(wire, schema)
    if valid:
        compiled = bind_authored_schedule(raw, slot())
        assert len(compiled['action_plan']) == (4 if has_opening else 2)
        compile_action_units(AuthoredShot.model_validate(compiled), slot())
    else:
        with pytest.raises(ValueError):
            bind_authored_schedule(raw, slot())


@pytest.mark.parametrize('topic', ['partner dance', 'parcel transfer', 'rope rescue'])
def test_cross_shot_entry_motion_stays_in_state_without_an_in_shot_edge(topic):
    from test_authoring_schedule import raw_shot, schema_for, slot
    from pipeline.action_authoring import AuthoredShot, compile_action_units
    raw = raw_shot()
    raw['start_state']['open_motion'] = f'{topic}: forward motion continues across the cut'
    raw['action_plan']['interaction_1']['action'] = f'{topic}: carries the incoming motion into contact'
    original = deepcopy(raw)
    jsonschema.validate({'shot': {k: v for k, v in raw.items() if k != 'shot_id'}}, ark_output_schema(schema_for(slot())))
    bound = bind_authored_schedule(raw, slot())
    assert bound['start_state'] == original['start_state']
    beats = compile_action_units(AuthoredShot.model_validate(bound), slot())
    assert beats[0]['action'] == original['action_plan']['interaction_1']['action']
    assert not beats[0].get('flow_from_previous')
    assert raw == original
