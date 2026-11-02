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
            assert audit is None or (audit['decisions'] == [] and audit['requirements_preserved']
                                     and audit['continuous_performance']), 'do not erase a negative fixture judgment'
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
    if 'evidence_sources' in payload:
        result = sourced_wire(result, payload['storyboard'])
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
