"""把有限创作控制域编译为提供方支持的枚举；不修改任何创作响应。"""
from copy import deepcopy
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
import hashlib
import json

from jsonschema import Draft202012Validator


MAX_DOMAIN_VALUES = 2048
LOCAL_KEYWORDS = frozenset({
    'minLength', 'maxLength', 'pattern', 'format', 'minimum', 'maximum',
    'exclusiveMinimum', 'exclusiveMaximum', 'multipleOf', 'minItems',
    'maxItems', 'uniqueItems', 'minProperties', 'maxProperties'})


def schema_nodes(node):
    """只遍历schema位置，不把const、enum或字段名中的数据当成schema。"""
    if not isinstance(node, dict):
        return
    yield node
    for key in ('properties', '$defs', 'patternProperties'):
        for child in node.get(key, {}).values():
            yield from schema_nodes(child)
    for key in ('items', 'additionalProperties', 'unevaluatedProperties', 'unevaluatedItems'):
        yield from schema_nodes(node.get(key))
    for key in ('anyOf', 'oneOf', 'allOf', 'prefixItems'):
        for child in node.get(key, []):
            yield from schema_nodes(child)


def bounded_authoring_schema(schema):
    """有界整数精确降低；小数仅降低类型定义明确声明的离散创作域。

    非有限域仍是本地约束，不伪称语法保证。不同字段共用同值域定义，
    既不按动作类型特判，也不在解码后夹取、取整或补默认值。
    """
    result = deepcopy(schema)
    json.dumps(result, allow_nan=False)
    Draft202012Validator.check_schema(result)
    definitions = {}
    for node in list(schema_nodes(result)):
        if node.get('type') not in ('integer', 'number') or 'enum' in node or 'const' in node:
            continue
        if node['type'] == 'number' and 'x-authoring-step' not in node:
            continue
        lowers = [(Decimal(str(node[k])), k == 'exclusiveMinimum')
                  for k in ('minimum', 'exclusiveMinimum') if k in node]
        uppers = [(Decimal(str(node[k])), k == 'exclusiveMaximum')
                  for k in ('maximum', 'exclusiveMaximum') if k in node]
        if not lowers or not uppers:
            if 'x-authoring-step' in node:
                raise ValueError('离散创作控制必须声明上下界；未调用模型')
            continue
        lower, lower_open = max(lowers)
        upper = min(v for v, _ in uppers)
        upper_open = any(v == upper and opened for v, opened in uppers)
        try:
            step = Decimal(1) if node['type'] == 'integer' else Decimal(str(node['x-authoring-step']))
        except InvalidOperation:
            raise ValueError('离散创作控制的步长必须为有限正数；未调用模型') from None
        if not step.is_finite() or step <= 0:
            raise ValueError('离散创作控制的步长必须为有限正数；未调用模型')
        first = int((lower / step).to_integral_value(rounding=ROUND_CEILING))
        last = int((upper / step).to_integral_value(rounding=ROUND_FLOOR))
        first += int(lower_open and first * step == lower)
        last -= int(upper_open and last * step == upper)
        if last < first or last - first + 1 > MAX_DOMAIN_VALUES:
            raise ValueError('创作数值域为空或超出结构化请求容量；未调用模型')
        values = []
        multiple = Decimal(str(node['multipleOf'])) if 'multipleOf' in node else None
        numeric_validator = Draft202012Validator({k: node[k] for k in (
            'type', 'minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum', 'multipleOf') if k in node})
        for i in range(first, last + 1):
            value = i * step
            if multiple is None or value % multiple == 0:
                number = int(value) if value == value.to_integral_value() else float(value)
                # JSON的小数由本地验证器按浮点解释；请求枚举不可比同一验证器更宽。
                if numeric_validator.is_valid(number):
                    values.append(number)
        if not values:
            raise ValueError('创作数值域没有可执行取值；未调用模型')
        domain = {'type': node['type'], 'enum': values}
        name = 'AuthoringDomain_' + hashlib.sha256(json.dumps(domain).encode()).hexdigest()[:16]
        if name in result.get('$defs', {}) and result['$defs'][name] != domain:
            raise ValueError('创作控制域定义冲突；未调用模型')
        definitions[name] = domain
        # 保留原上下界便于本地审计；$ref同样约束请求和解码。
        reference = {'$ref': '#/$defs/' + name}
        if '$ref' in node and node['$ref'] != reference['$ref']:
            if reference not in node.get('allOf', []):
                node.setdefault('allOf', []).append(reference)
        else:
            node.update(reference)
    if definitions:
        result.setdefault('$defs', {}).update(definitions)
    return result


def describe_local_constraints(schema):
    """向作者公开服务端无法强制的约束；仍必须通过本地验证。"""
    for node in schema_nodes(schema):
        constraints = {k: node[k] for k in sorted(LOCAL_KEYWORDS & node.keys())}
        if constraints:
            notice = '本地执行约束（仍须校验）：' + json.dumps(constraints, ensure_ascii=False)
            if not node.get('description', '').endswith(notice):
                node['description'] = (node.get('description', '') + '\n' + notice).strip()


def diagnostic_errors(error):
    """先比较外层形状/判别值，再定位字段；并列分支仅报告共同错误。"""
    if not error.context or error.validator not in ('oneOf', 'anyOf'):
        return [error]
    branches = {}
    for child in error.context:
        branches.setdefault(child.schema_path[0], []).extend(diagnostic_errors(child))
    structural = {'type', 'const', 'enum', 'required', 'additionalProperties'}
    depth = max((len(e.absolute_path) - len(error.absolute_path)
                 for errors in branches.values() for e in errors), default=0)
    def rank(errors):
        return tuple(sum(e.validator in structural and
                         len(e.absolute_path) - len(error.absolute_path) == level
                         for e in errors) for level in range(depth + 1))
    best = min(map(rank, branches.values()))
    compatible = [errors for errors in branches.values() if rank(errors) == best]
    if len(compatible) == 1:
        return compatible[0]
    def identity(e):
        return (tuple(e.absolute_path), e.validator, json.dumps(e.validator_value, sort_keys=True))
    common = set(map(identity, compatible[0]))
    for errors in compatible[1:]:
        common.intersection_update(map(identity, errors))
    return [e for e in compatible[0] if identity(e) in common] or [error]
