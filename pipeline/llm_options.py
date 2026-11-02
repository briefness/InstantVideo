"""Provider-specific options for bounded JSON tasks, not general reasoning tasks."""
import re


def ark_output_schema(schema: dict) -> dict:
    """Project Pydantic JSON Schema onto Ark's documented decoding subset.

    Numeric/string/cardinality bounds and discriminators remain local validators.
    Property names are data, so they must not be filtered as schema keywords.
    """
    scalar = {'type', '$ref', 'const', 'enum', 'required', 'description'}
    mappings = {'properties', '$defs'}
    children = {'items', 'additionalProperties', 'unevaluatedProperties', 'unevaluatedItems'}
    sequences = {'anyOf', 'oneOf', 'allOf', 'prefixItems'}
    result = {}
    for key, value in schema.items():
        if key in scalar:
            result[key] = value
        elif key in mappings:
            result[key] = {name: ark_output_schema(child) for name, child in value.items()}
        elif key in children:
            result[key] = ark_output_schema(value) if isinstance(value, dict) else value
        elif key in sequences:
            result[key] = [ark_output_schema(child) for child in value]
    return result


def structured_output_options(model: str, *, reasoning: bool = False) -> dict:
    # 已核对 Ark 的 Seed 2.0/2.1 Chat API；不向未知模型/自定义 endpoint 猜传参数。
    # 提取任务关闭思考以保护输出预算；逐镜创作显式开启，不能把格式约束当成推理能力。
    if re.fullmatch(r'doubao-seed-2[.-][01]-(?:lite|mini|pro|turbo|code-preview)(?:-\d{6})?', model):
        return {'extra_body': {'thinking': {'type': 'enabled' if reasoning else 'disabled'}}}
    return {}
