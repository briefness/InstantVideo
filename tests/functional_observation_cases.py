"""Explicit synthetic witness annotations for offline service fixtures, never real footage."""
from copy import deepcopy
import json


def annotated_support_response(request):
    """明确的离线支持标注，只测试协议与编排，不评价任何原片。"""
    payload = json.loads(request['messages'][-1]['content'])
    assert payload['protocol'] in {'observation-support-v4-evidence-context', 'observation-support-v5-visual-evidence'}
    if payload['protocol'] == 'observation-support-v5-visual-evidence':
        assert any(s['scope'] == 'current_clip' for s in payload['visual_evidence']['sources'])
        assert any(p.get('type') == 'image_url' for m in request['messages']
                   if isinstance(m['content'], list) for p in m['content'])
    if payload.get('evidence_policy') == 'grounded-requirements-v2':
        assert 'observed_context' not in payload
        current = next(s for s in payload['visual_evidence']['sources'] if s['scope']=='current_clip')
        refs = [f'current_clip:{current["frames"][0]["index"]}', f'current_clip:{current["frames"][-1]["index"]}']
        return {'decisions': {key:{'requirement':'fulfilled', 'issue':'none',
            'finding':'Offline annotation: completed.',
            'evidence':refs} for key in payload['cases']}}
    assert 'observed_context' in payload
    if payload.get('evidence_policy') == 'witness-decisions-v1':
        return {'decisions': {key:{'observation':'verified','requirement':'fulfilled','issue':'none',
            'evidence':[f'current_clip:{i}' for i in case['sample_indices'][:8]]}
            for key,case in payload['cases'].items()}}
    return {'decisions': {key: 'supported' for key in payload['cases']}}


def annotated_observation_response(value, request):
    schema = request['response_format']['json_schema']['schema']
    if 'decisions' in schema['properties']:
        return annotated_support_response(request)
    if 'observations' not in schema['properties']:
        return value
    contract = json.JSONDecoder().raw_decode(request['messages'][-1]['content'][0]['text'].split('Shot contract:\n')[1])[0]
    criteria = contract['completion_contract']['criteria']
    requested = schema['properties']['observations']['properties']
    count = (next(iter(requested.values()))['properties']['sample_indices']['items']['maximum']
             if requested else schema['properties']['terminal_observation']['properties']['sample_index']['const'])
    result = {k: deepcopy(v) for k, v in value.items() if k in schema['properties']}
    for event in result.get('action_event_evidence', []):
        event.pop('result_visible', None)
    result['observations'] = {k: {'status': 'satisfied',
        'observation': 'Offline synthetic annotation: action completed with its functional response.',
        'sample_indices': [1, count]} for k in requested}
    for event in value.get('action_event_evidence', []):
        if event.get('result_visible') is False:
            key = f"beat:{event['event_id']}/completion"
            affected = criteria[key].get('supports', key)
            for target in affected if isinstance(affected, list) else [affected]:
                result['observations'][target].update(status='violated',
                    observation='Offline synthetic annotation: the intended function did not complete.')
    last = max((k for k in requested if criteria[k]['kind'] == 'completion'),
               key=lambda k: criteria[k]['beat_id'], default=next(iter(requested), None))
    state = value['observed_end_state']
    if last and result['observations'][last]['status'] == 'satisfied':
        result['observations'][last]['observation'] = '; '.join(state[k] for k in (
            'action_phase', 'pose_and_gaze', 'prop_state', 'open_motion') if state.get(k))
    result['end_state_context'] = {k: state.get(k, '') for k in schema['properties']['end_state_context']['properties']}
    if 'terminal_observation' in schema['properties']:
        result['terminal_observation'] = {'sample_index': count, 'status': 'observed',
            **{k: state.get(k, '') for k in ('action_phase', 'pose_and_gaze', 'prop_state', 'open_motion')}}
    else:
        result['end_state_sources'] = {k: [last] if state.get(k) else []
                                      for k in schema['properties']['end_state_sources']['properties']}
    return result
