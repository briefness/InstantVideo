"""创作源契约的请求投影与解码共用边界，不把服务端子集当作完整校验。"""
from copy import deepcopy
import hashlib
import json

from jsonschema import Draft202012Validator

from pipeline.llm_options import ark_output_schema
from pipeline.authoring_controls import (
    LOCAL_KEYWORDS, bounded_authoring_schema, describe_local_constraints, diagnostic_errors)


class AuthoringContractError(ValueError):
    def __init__(self, issues):
        self.issues = issues
        fields = '；'.join(f"{v['path']} ({v['constraint']})" for v in issues[:8])
        super().__init__('创作响应契约不满足：' + fields + '；原稿已保留，未进入动作编译或重新生成')


def _pointer(parts):
    return '/' + '/'.join(str(p).replace('~', '~0').replace('/', '~1') for p in parts)


class AuthoringContract:
    """只做原文验证，不提供默认内容、类型纠正、删除或修稿。"""
    def __init__(self, schema, *, generation_constraints=False):
        self.generation_constraints = generation_constraints
        self.schema = bounded_authoring_schema(schema) if generation_constraints else deepcopy(schema)
        if generation_constraints:
            describe_local_constraints(self.schema)
        self.local_constraints = []
        unsupported = LOCAL_KEYWORDS

        def visit(node, path=()):
            if not isinstance(node, dict):
                return
            reference = node.get('$ref')
            if reference is not None and (not isinstance(reference, str) or not reference.startswith('#')):
                raise ValueError('创作契约只允许本地引用；未请求模型或外部schema')
            for key in unsupported & node.keys():
                self.local_constraints.append({'path': _pointer(path), 'constraint': key, 'value': node[key]})
            for key in ('properties', '$defs', 'patternProperties'):
                for name, child in node.get(key, {}).items():
                    visit(child, (*path, key, name))
            for key in ('items', 'additionalProperties', 'unevaluatedProperties', 'unevaluatedItems'):
                if key in node:
                    visit(node[key], (*path, key))
            for key in ('anyOf', 'oneOf', 'allOf', 'prefixItems'):
                for i, child in enumerate(node.get(key, [])):
                    visit(child, (*path, key, i))
        visit(self.schema)
        Draft202012Validator.check_schema(self.schema)
        self.validator = Draft202012Validator(self.schema)
        self.wire_schema = ark_output_schema(self.schema)
        self.local_constraints.sort(key=lambda v: (v['path'], v['constraint']))

    def record(self):
        def digest(value):
            return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        return {'version': 'authoring-contract-v2' if self.generation_constraints else 'authoring-contract-v1',
                'schema': deepcopy(self.schema),
                'schema_sha256': digest(self.schema), 'wire_sha256': digest(self.wire_schema),
                'local_constraints': deepcopy(self.local_constraints)}

    def validate(self, value):
        # Python JSON解码器可接纳NaN/Infinity；它们不属于JSON数值，且能绕过schema比较。
        try:
            json.dumps(value, allow_nan=False)
        except (ValueError, TypeError):
            raise AuthoringContractError([{'path': '/', 'constraint': 'json'}]) from None
        issues = []
        def collect(error):
            if len(issues) >= 32:
                return
            for leaf in diagnostic_errors(error):
                if len(issues) >= 32:
                    break
                item = {'path': _pointer(leaf.absolute_path), 'constraint': leaf.validator}
                if item not in issues:
                    issues.append(item)
        for error in self.validator.iter_errors(value):
            collect(error)
            if len(issues) >= 32:
                break
        if issues:
            raise AuthoringContractError(issues)
