"""Bind review questions to source data before asking for semantic judgments.

The model returns judgments under fixed keys. It never invents foreign keys,
counts source claims, or repeats their prose to identify them.
"""
from copy import deepcopy

from pipeline.result_admission import review_claims
from pipeline.goal_allocation import VERSION as ALLOCATION_VERSION
from pipeline.goal_allocation import LEGACY_VERSION as LEGACY_ALLOCATION_VERSION
from pipeline.goal_allocation import SCOPED_VERSION as SCOPED_ALLOCATION_VERSION
from pipeline.planning_evidence import MAX_REVIEW_UNITS


LEGACY_VERSION = 'planning-semantics-v6-bound-targets'
CITATION_VERSION = 'planning-semantics-v7-bound-citations'
SOURCE_VERSION = 'planning-semantics-v8-evidence-sources'
VERSION = 'planning-semantics-v9-completion-intent'
GOAL_VERSION = 'planning-semantics-v10-source-goals'
SUPPORTED_VERSIONS = (LEGACY_VERSION, CITATION_VERSION, SOURCE_VERSION, VERSION, GOAL_VERSION,
                      LEGACY_ALLOCATION_VERSION, SCOPED_ALLOCATION_VERSION, ALLOCATION_VERSION)
TARGET_INSTRUCTION = '''输出units是对象，不是列表。逐项填写review_targets中的固定键，不返回shot_id或beat_index。
有candidate_results的项目必须给result_admission；其中decisions也是对象，键与candidate_results完全一致。
程序已绑定项目与原稿；不要重抄陈述作为键，不返回source，不新增或合并项目。只给出分类、证据及判断。
没有candidate_results的项目不需要result_admission（可省略或null），但仍须完成原因、类型、证据及全片连续性判断。
'''
INSTRUCTION = TARGET_INSTRUCTION + '''
basis=continuity时，evidence只能选择continuity_sources.allowed中当前项目允许的一个完整路径。
continuity_sources.fields提供路径对应的原文。路径由程序绑定，不按镜头编号猜数组下标，不拼接解释；依赖理由写在reason中。
存在可选路径不代表依赖成立；仍须独立判断后续动作是否确实离不开该结果。
'''

SOURCED_INSTRUCTION = '''每项结果归类只返回basis、reference、composition、reason四个字段。
reference只选择evidence_sources.allowed中本项目、本basis允许的地址；fields是程序提供的原文。
user只引用用户原文，continuity只引用后续依赖，response只引用本次动作或其反应。
direction的reference必须为null，reason说明替换该细节为何仍满足用户目标和后续因果。
每项reason单独解释引用如何支持该项分类；引用存在不代表分类正确，不能把分镜当成用户要求。
composition=single_category表示整条陈述属于所选类别；mixed_categories表示混入不同必要性类别，uncertain表示无法确定。
这里判断的是类别是否混合，不是字数、细节数量或“越短越好”。同属必要结果的多个后果可以是single_category。
含必要结局又含可替换姿势的陈述须为mixed_categories；无法判断时用uncertain，不自动拆句、改稿或降级必要要求。
'''


def _sourced_decision_schema(sources: dict, key: str) -> dict:
    variants = []
    for basis in ('user', 'continuity', 'response', 'direction'):
        choices = sources['allowed'][key].get(basis, [])
        if basis != 'direction' and not choices:
            continue
        variants.append({'type': 'object', 'properties': {
            'basis': {'type': 'string', 'const': basis},
            'reference': {'type': 'null'} if basis == 'direction' else {'type': 'string', 'enum': choices},
            'composition': {'type': 'string', 'enum': ['single_category', 'mixed_categories', 'uncertain']},
            'reason': {'type': 'string', 'minLength': 1, 'maxLength': 1600}},
            'required': ['basis', 'reference', 'composition', 'reason'], 'additionalProperties': False})
    return {'oneOf': variants}


def _decode_sourced_decision(judgment: dict, sources: dict, key: str) -> dict:
    _exact_keys(judgment, ('basis', 'reference', 'composition', 'reason'), '结果证据')
    basis, reference = judgment['basis'], judgment['reference']
    reason, composition = judgment['reason'], judgment['composition']
    if (not isinstance(basis, str) or basis not in ('user', 'continuity', 'response', 'direction')
            or not isinstance(composition, str) or composition not in ('single_category', 'mixed_categories', 'uncertain')
            or not isinstance(reason, str) or not reason.strip() or len(reason) > 1600):
        raise ValueError('结果证据的分类、混合判断或解释无效')
    if basis == 'direction':
        if reference is not None:
            raise ValueError('表演建议不能冒用要求来源')
        evidence = reason
    else:
        if not isinstance(reference, str) or reference not in sources['allowed'][key].get(basis, []):
            raise ValueError(f'分镜审查项目 {key} 的 {basis} 证据必须选择对应来源的原文')
        evidence = sources['fields'][reference] if basis == 'user' else reference
    return {'basis': basis, 'evidence': evidence,
            'minimal': composition == 'single_category', 'reason': reason}


def review_targets(storyboard: dict, *, admit_intents: bool = False) -> dict:
    targets = {}
    for shot in storyboard['shots']:
        for index, beat in enumerate(shot['action_beats']):
            key = f"{shot['shot_id']}:{index}"
            if key in targets:
                raise ValueError('分镜审查项目重复，不能覆盖原稿来源')
            targets[key] = {'shot_id': shot['shot_id'], 'beat_index': index,
                            'outcome_contract_required': bool(beat.get('outcomes')),
                            'candidate_results': review_claims(shot, beat)}
            if admit_intents:
                targets[key]['completion_owners'] = [i + 1 for i in range(len(shot['action_beats'])) if i != index]
    if not 1 <= len(targets) <= MAX_REVIEW_UNITS:
        raise ValueError('分镜审查项目数量超出单批范围')
    return targets


def compile_review_response_schema(base: dict, targets: dict, *, sources: dict | None = None,
                                   evidence_sources: dict | None = None) -> dict:
    """One source-bound shape for decoding and local validation; no new review."""
    schema = deepcopy(base)
    decision = deepcopy(schema['$defs']['ResultDecision'])
    decision['properties'].pop('source')
    decision['required'].remove('source')
    if evidence_sources is None:
        schema['$defs']['BoundResultDecision'] = decision
    units = {}
    for key, target in targets.items():
        unit = deepcopy(schema['$defs']['UnitVerdict'])
        for field in ('shot_id', 'beat_index'):
            unit['properties'].pop(field)
            unit['required'].remove(field)
        if 'completion_owners' in target:
            from pipeline.completion_intent import intent_schema
            unit['properties']['completion_intent'] = intent_schema(evidence_sources, key, target['completion_owners'])
            unit['required'].append('completion_intent')
        else:
            unit['properties'].pop('completion_intent', None)
        if target['outcome_contract_required']:
            unit['properties']['outcome_contract_valid'] = {'type': 'boolean'}
            unit['required'].append('outcome_contract_valid')
        else:
            unit['properties']['outcome_contract_valid'] = {'type': 'null'}
        claims = target['candidate_results']
        if claims:
            decision_name = 'BoundResultDecision'
            if evidence_sources is not None:
                decision_name = f'BoundResultDecision_{len(units)}'
                schema['$defs'][decision_name] = _sourced_decision_schema(evidence_sources, key)
            elif sources is not None:
                decision_name = f'BoundResultDecision_{len(units)}'
                prose = deepcopy(decision)
                prose['properties']['basis'] = {'type': 'string', 'enum': ['user', 'response', 'direction']}
                variants = [prose]
                if sources['allowed'][key]:
                    citation = deepcopy(decision)
                    citation['properties']['basis'] = {'type': 'string', 'const': 'continuity'}
                    citation['properties']['evidence'] = {'type': 'string', 'enum': sources['allowed'][key]}
                    variants.append(citation)
                schema['$defs'][decision_name] = {'oneOf': variants}
            admission = deepcopy(schema['$defs']['ResultAdmission'])
            admission['properties']['decisions'] = {
                'type': 'object', 'properties': {path: {'$ref': f'#/$defs/{decision_name}'} for path in claims},
                'required': list(claims), 'additionalProperties': False}
            unit['properties']['result_admission'] = admission
            unit['required'].append('result_admission')
        else:
            unit['properties']['result_admission'] = {'type': 'null'}
        units[key] = unit
    schema['properties']['units'] = {'type': 'object', 'properties': units,
                                      'required': list(units), 'additionalProperties': False}
    for name in ('UnitVerdict', 'ResultAdmission', 'ResultDecision'):
        schema['$defs'].pop(name)
    return schema


def _exact_keys(value: object, keys, label: str) -> None:
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f'分镜审查{label}必须逐项对应原稿固定键，不能缺少、添加或使用自由文本引用')


def decode_review_response(raw: dict, targets: dict, *, sources: dict | None = None,
                           evidence_sources: dict | None = None) -> dict:
    """Lower fixed-key answers to the existing verdict, without inferring facts."""
    if not isinstance(raw, dict):
        raise ValueError('分镜审查响应必须是对象')
    _exact_keys(raw.get('units'), targets, '项目')
    units = []
    for key, target in targets.items():
        source = raw['units'][key]
        if not isinstance(source, dict) or {'shot_id', 'beat_index'} & set(source):
            raise ValueError('分镜审查项目必须仅含判断，编号由程序绑定')
        unit = deepcopy(source)
        if 'completion_owners' in target:
            from pipeline.completion_intent import validate_intent
            unit['completion_intent'] = validate_intent(unit.get('completion_intent'), evidence_sources, key)
        elif unit.get('completion_intent') is not None:
            raise ValueError('历史审查不能自行增加功能审定')
        if target['outcome_contract_required']:
            if type(unit.get('outcome_contract_valid')) is not bool:
                raise ValueError(f'分镜审查项目 {key} 缺少结果合同判断')
        elif unit.get('outcome_contract_valid') is not None:
            raise ValueError(f'分镜审查项目 {key} 没有结果合同，不能虚构判断')
        claims = target['candidate_results']
        admission = unit.get('result_admission')
        if claims:
            if not isinstance(admission, dict):
                raise ValueError(f'分镜审查项目 {key} 缺少待审结果判断')
            _exact_keys(admission.get('decisions'), claims, f'项目 {key} 的结果归类')
            decisions = []
            for path in claims:
                judgment = admission['decisions'][path]
                if not isinstance(judgment, dict) or 'source' in judgment:
                    raise ValueError('结果来源已由程序绑定，模型只能填写判断')
                if evidence_sources is not None:
                    judgment = _decode_sourced_decision(judgment, evidence_sources, key)
                if (sources is not None and judgment.get('basis') == 'continuity'
                        and judgment.get('evidence') not in sources['allowed'][key]):
                    raise ValueError(f'分镜审查项目 {key} 的衔接引用必须选择已绑定的后续原文')
                decisions.append({**judgment, 'source': path})
            admission['decisions'] = decisions
        elif admission is not None:
            raise ValueError(f'分镜审查项目 {key} 没有候选结果，不能虚构归类')
        units.append({**unit, 'shot_id': target['shot_id'], 'beat_index': target['beat_index']})
    return {**deepcopy(raw), 'units': units}
