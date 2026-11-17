"""Render explicitly annotated test judgments in the requested wire protocol.

This is test data preparation, never production recovery or a model substitute
used to evaluate the recorded invalid response. Missing/fabricated keys are kept.
"""
from copy import deepcopy
import json


def annotated_outcome(value, story):
    """Explicit synthetic evidence for protocol tests, never recorded verdicts."""
    result = deepcopy(value)
    if ((story.get('review_scope') or {}).get('is_final_batch') is False
            or not (story.get('production_plan', {}).get('director_intent') or {}).get('ending_requirement')):
        return result
    accepted = result.pop('requested_outcome_valid')
    i = len(story['shots']) - 1
    j = len(story['shots'][i]['action_beats']) - 1
    result['requested_outcome'] = {'relation': 'entailed' if accepted else 'undetermined',
        'evidence': [f'/shots/{i}/action_beats/{j}/visible_result'],
        'reason': result['reason']}
    return result


def wire_verdict(value, kwargs):
    shape = kwargs.get('response_format', {}).get('json_schema', {}).get('schema', {})
    units = shape.get('properties', {}).get('units', {})
    if units.get('type') != 'object' or not isinstance(value, dict) or not isinstance(value.get('units'), list):
        return value
    result = deepcopy(value)
    bound = {}
    for entry in result['units']:
        key = f"{entry.pop('shot_id')}:{entry.pop('beat_index')}"
        assert key not in bound, 'fixture repeats an entire review unit'
        properties = units['properties'].get(key, {}).get('properties', {})
        if properties.get('result_admission', {}).get('type') == 'null':
            audit = entry.pop('result_admission', None)
            assert audit is None or (audit['requirements_preserved'] and audit['continuous_performance']
                and (audit['decisions'] == [] or all(d['minimal'] for d in audit['decisions']))), 'do not erase a negative fixture judgment'
        elif isinstance(entry.get('result_admission'), dict):
            decisions = {}
            for decision in entry['result_admission']['decisions']:
                source = decision.pop('source')
                assert source not in decisions, 'fixture repeats a result claim'
                decisions[source] = decision
            entry['result_admission']['decisions'] = decisions
        if properties.get('outcome_contract_valid', {}).get('type') == 'null':
            assert entry.pop('outcome_contract_valid', None) is None
        bound[key] = entry
    result['units'] = bound
    payload = json.loads(kwargs['messages'][-1]['content'])
    from pipeline.goal_allocation import VERSION, LEGACY_VERSION
    if payload.get('version') in (VERSION, LEGACY_VERSION):
        goals = payload['evidence_sources']['goals']
        if 'goal_allocation' not in result:
            # Compatibility for synthetic single-goal fixtures only. Real captured
            # object responses return above unchanged; they are never repaired.
            assert len(goals) == 1, 'multi-goal fixtures require explicit goal evidence'
            ref, goal = next(iter(goals.items()))
            shot_ids = list(dict.fromkeys(k.split(':')[0] for k in bound))
            result['goal_allocation'] = {ref: {'parts': [{'quote': goal['goal'],
                'units': [k for k in bound if k.split(':')[0] == shot_id], 'relation': 'entailed',
                'reason': 'Explicit synthetic fixture requires this ongoing function in every shot; not semantic evaluation.'}
                for shot_id in shot_ids]}}
        for unit in bound.values():
            assert 'completion_intent' not in unit, 'new fixtures must annotate goals, not repair old intent judgments'
    elif payload.get('version') == 'planning-semantics-v10-source-goals':
        goals = payload['evidence_sources']['goals']
        for unit in result['units'].values():
            if 'completion_intent' not in unit:
                path, goal = next(iter(goals.items()))
                unit['completion_intent'] = {'kind': 'required', 'goal': goal['goal'], 'basis': goal['basis'],
                    'reference': path, 'reason': 'Explicit synthetic single-goal fixture; not semantic extraction.'}
            if unit['completion_intent']['kind'] == 'required':
                unit['completion_intent'].pop('goal', None)
                unit['completion_intent'].pop('basis', None)
    elif payload.get('version') == 'planning-semantics-v9-completion-intent':
        result = strict_intent_wire(result, payload['storyboard'])
    if 'boundary_review' in payload and 'continuity_valid' in result:
        accepted = result.pop('continuity_valid')
        assert type(accepted) is bool, 'synthetic continuity annotation must be explicit'
        result['boundaries'] = {key: {
            'state_relation': 'supported' if accepted else 'contradicted',
            'action_progression': 'supported' if accepted else 'contradicted',
            'previous_evidence': [next(iter(pair['previous']))],
            'next_evidence': [next(iter(pair['next']))],
            'reason': result['reason'],
        } for key, pair in payload['boundary_review']['boundaries'].items()}
    if 'evidence_sources' in payload and payload.get('version') not in ('planning-semantics-v10-source-goals', VERSION, LEGACY_VERSION):
        result = sourced_wire(result, payload['storyboard'])
    return result


def strict_intent_wire(value, story):
    """Old synthetic cases deliberately require every action; never annotate captured replies.

    This retains their negative action-omission assertions. Semantic distinctions
    between functions and realizations have separate, explicitly authored cases.
    """
    from pipeline.action_results import result_claims
    result = deepcopy(value)
    for i, shot in enumerate(story['shots']):
        for j, beat in enumerate(shot['action_beats']):
            unit = result['units'].get(f"{shot['shot_id']}:{j}")
            if unit is not None and 'completion_intent' not in unit:
                claims = result_claims(beat)
                responses = [claims[path]['state'] for path, decision in
                    (unit.get('result_admission') or {}).get('decisions', {}).items() if decision.get('basis') == 'response']
                unit['completion_intent'] = {'kind': 'required', 'goal': '; '.join([beat['action'], *responses]), 'basis': 'story',
                    'reference': f'/shots/{i}/narrative_beat/state_change',
                    'reason': 'Strict synthetic test interpretation; every original action remains required.'}
    return result


def sourced_wire(value, story):
    """Explicit synthetic annotations only; never use on recorded service responses."""
    from pipeline.planning_review_sources import review_evidence_sources
    sources = review_evidence_sources(story)
    result = deepcopy(value)
    for key, unit in result['units'].items():
        for path, decision in (unit.get('result_admission') or {}).get('decisions', {}).items():
            basis = decision['basis']
            evidence = decision.get('evidence')
            if basis == 'direction':
                reference = None
            elif basis == 'continuity':
                reference = decision['evidence']
            elif basis == 'user':
                reference = next((ref for ref in sources['allowed'][key]['user']
                                  if isinstance(evidence, str) and evidence in sources['fields'][ref]), evidence)
            else:
                reference = sources['allowed'][key]['response'][0]
            minimal = decision.get('minimal')
            composition = ('single_category' if minimal else 'mixed_categories') if type(minimal) is bool else minimal
            unit['result_admission']['decisions'][path] = {
                **{name: value for name, value in decision.items()
                   if name not in ('basis', 'evidence', 'minimal')},
                'basis': basis, 'reference': reference,
                'composition': composition, 'reason': evidence}
    return result
