"""Mechanical projections for replay tests, never used by production authoring."""
from copy import deepcopy


class SyntheticAuthoredResponse(dict):
    """Marker for authored test annotations, never captured model responses."""


def entity_resolution_fixture(unit):
    """独立测试标注的机械投影；不读取正文猜测归属，不用于真实响应。"""
    source = unit['resolution']
    if 'states' in source:
        return deepcopy(source)
    claims = [{'entity': unit['target'], 'state': source['target_state'], 'importance': 'required'}]
    if source['actor_state'] is not None:
        claims.append({'entity': unit['actor'], 'state': source['actor_state'], 'importance': 'required'})
    claims.extend(deepcopy(source['scene_changes']))
    states = {}
    for claim in claims:
        states.setdefault(claim['entity'], []).append({k: claim[k] for k in ('state', 'importance')})
    return {'states': states}


def scheduled_shot_fixture(shot, slot):
    """Represent the same synthetic performance in the slot-bound wire format."""
    result = deepcopy(shot)
    from pipeline.production_plan import CANONICAL_FRAMING
    if slot.get('framing_family') in CANONICAL_FRAMING:
        result['camera']['start_framing'] = CANONICAL_FRAMING[slot['framing_family']]
    units = result.pop('action_plan')
    schedule = {}
    opening = []
    if slot.get('result_hold_seconds') and units and units[-1]['kind'] == 'hold':
        schedule['result_hold'] = {k: v for k, v in units.pop().items() if k not in ('id', 'kind')}
    count = 0
    following = opening
    for unit in units:
        unit = {k: v for k, v in unit.items() if k != 'id'}
        if unit['kind'] == 'interaction':
            unit['resolution'] = entity_resolution_fixture(unit)
        if unit['kind'] == 'interaction' and count < slot['min_interaction_events']:
            count += 1
            unit.pop('kind')
            unit['following'] = []
            schedule[f'interaction_{count}'] = unit
            following = unit['following']
        else:
            following.append(unit)
    if opening:
        schedule['opening'] = {'first': opening[0], 'following': opening[1:]}
    result['action_plan'] = schedule
    return result


def contextual_shot_fixture(shot):
    """Synthetic annotations only: preserve the independent legacy prop sets."""
    result = deepcopy(shot)
    visible = result.pop('key_props', [])
    persistent = result.pop('continuity_props', [])
    result['prop_roles'] = [
        {'entity': name, 'role': ('visible_and_persistent' if name in visible and name in persistent
                                  else 'visible' if name in visible else 'persistent')}
        for name in dict.fromkeys([*visible, *persistent])]
    return result


def authoring_from_legacy(story):
    result = {k: deepcopy(story[k]) for k in ('title', 'mood', 'music_style', 'theme_elements', 'story_arc', 'characters') if k in story}
    result.setdefault('theme_elements', [c['name'] for c in result['characters']])
    result['shots'] = []
    for shot in story['shots']:
        units = []
        for i, beat in enumerate(shot['action_beats']):
            unit = {'id': f'unit_{i}', 'kind': beat['activity'], 'actor': beat['actor'],
                    'target': beat.get('target', ''), 'action': beat['action'], 'result': beat['visible_result']}
            if beat['activity'] == 'interaction':
                g = beat['interaction_geometry']
                if g['actor'] != beat['actor'] or g['target'] != beat['target']:
                    raise ValueError(f'Shot {shot["shot_id"]}: conflicting event ownership')
                unit.update(effect={'mode': g['interaction_mode'], 'motion': g['effect_motion'],
                                    'scope': g['outcome_scope'], 'source': g['source'],
                                    'region': g['effect_region'], 'unaffected': g['unaffected_behavior']},
                            response=beat.get('response_kind') or 'impact', carry=beat.get('flow_from_previous') or '')
                if i and beat.get('start_seconds', 0) < shot['action_beats'][i - 1].get('end_seconds', 0):
                    unit['overlap'] = True
            units.append(unit)
        result['shots'].append({
            **{k: deepcopy(shot[k]) for k in ('shot_id', 'scene_id', 'scene_description', 'camera',
                                             'blocking', 'start_state', 'end_state', 'key_props',
                                             'continuity_props', 'lighting', 'mood') if k in shot},
            'visual_description': shot['scene_description'],
            'story_change': shot['narrative_beat']['state_change'],
            'story_state_after': shot['narrative_beat']['state_after'], 'action_plan': units})
    return result
