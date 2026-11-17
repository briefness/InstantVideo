"""将有出处的创作选择编译为各阶段共用的表达方法，而非题材规则。"""
from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pipeline.intent_evidence import RequiredEvidence, has_evidence


# v1 是可重放的语义版本；修改既有方法的含义时须保留旧版本并新增版本。
MethodId = Literal[
    'spatial_readability', 'movement_phrasing', 'physical_problem_solving',
    'restrained_reaction', 'ensemble_space', 'suspense_information',
    'subjective_association', 'observational_duration', 'expressive_space',
    'material_detail', 'musical_phrasing', 'sound_counterpoint',
]
ReferenceId = Literal[
    'wuxia_kinetic', 'physical_visual_comedy', 'domestic_restraint', 'urban_memory',
    'film_noir', 'observational_viewpoint', 'surreal_association', 'musical_spatial',
    'king_hu', 'jackie_chan', 'buster_keaton', 'ang_lee', 'edward_yang',
    'alfred_hitchcock', 'satoshi_kon', 'wong_kar_wai',
]

METHODS = {
    'spatial_readability': 'Stage the authored action with readable screen direction, eyelines and relationships between subjects and their surroundings; reveal a changed axis before relying on it.',
    'movement_phrasing': 'Connect authored movement through preparation, acceleration, redirection and recovery; carry momentum across gestures without mandatory alternating turns or resets. This does not prescribe playback speed.',
    'physical_problem_solving': 'Let established objects, obstacles and body mechanics make the authored problem and response legible; earn physical humor through preparation and consequence, not unplanned incidents.',
    'restrained_reaction': 'Express emotional changes through small shifts in gaze, distance, posture and handling of established objects; leave room for a reaction without forcing exaggerated gestures.',
    'ensemble_space': 'Use depth, thresholds and relative placement to reveal the established relationships within a group; keep the current dramatic focus readable without giving every person an isolated close-up.',
    'suspense_information': 'Control when the viewer notices established information relative to the characters; use framing, eyelines and offscreen space to sustain the authored uncertainty without inventing a threat or twist.',
    'subjective_association': 'Connect established images through shape, gesture or sound when the authored story calls for subjective association; keep factual state transitions distinct from subjective impressions.',
    'observational_duration': 'Attend to ordinary physical processes and environmental activity within the allocated shot; let behavior unfold without forcing a dramatic pose. Observation does not require slow playback or longer footage.',
    'expressive_space': 'Use lighting contrast, spatial depth and negative space to express the authored emotional pressure, while preserving the scene layout and recognizable subjects.',
    'material_detail': 'Make the established surfaces, weight, wear and contact with light tangible; details should support the current task rather than replace it with unrelated inserts.',
    'musical_phrasing': 'Organize authored gestures and camera emphasis into preparation, accent and release with readable shared space; do not assume that event boundaries are beats or that all movement has equal duration.',
    'sound_counterpoint': 'Use the authored relationship between visible action and sound to shape expectation; retain useful environmental and contact sounds instead of masking them with continuous musical emphasis.',
}

# 只接入已核对的代表性组合；完整研究见两份 atlas。候选关系不自动施加全部方法。
REFERENCES = {
    'wuxia_kinetic': ('凌厉流动武侠', ['movement_phrasing', 'spatial_readability']),
    'physical_visual_comedy': ('默片式视觉喜剧', ['physical_problem_solving', 'spatial_readability']),
    'domestic_restraint': ('家庭日常与克制观看', ['restrained_reaction', 'ensemble_space']),
    'urban_memory': ('都市情绪与记忆抒情', ['subjective_association', 'restrained_reaction']),
    'film_noir': ('黑色电影与宿命犯罪', ['expressive_space', 'suspense_information']),
    'observational_viewpoint': ('观察式纪实', ['observational_duration', 'material_detail']),
    'surreal_association': ('超现实主义与梦逻辑', ['subjective_association', 'expressive_space']),
    'musical_spatial': ('歌舞电影的身体与空间', ['musical_phrasing', 'spatial_readability']),
    'king_hu': ('胡金铨：空间、视线与动作节奏', ['spatial_readability', 'movement_phrasing']),
    'jackie_chan': ('成龙：环境任务与身体解题', ['physical_problem_solving', 'movement_phrasing']),
    'buster_keaton': ('巴斯特·基顿：身体、空间与机械因果', ['physical_problem_solving', 'spatial_readability']),
    'ang_lee': ('李安：关系压力与克制反应', ['restrained_reaction', 'ensemble_space']),
    'edward_yang': ('杨德昌：群体关系与社会空间', ['ensemble_space', 'observational_duration']),
    'alfred_hitchcock': ('希区柯克：观众信息与角色信息的差', ['suspense_information', 'spatial_readability']),
    'satoshi_kon': ('今敏：主观联想与转场关系', ['subjective_association', 'suspense_information']),
    'wong_kar_wai': ('王家卫：记忆、重复与情绪间接表达', ['subjective_association', 'restrained_reaction']),
}


class CreativeDirectionSpec(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

    version: Literal['creative-direction-v1'] = 'creative-direction-v1'
    references: list[ReferenceId] = Field(default_factory=list, max_length=2)
    methods: list[MethodId] = Field(default_factory=list, max_length=4)
    evidence: RequiredEvidence
    rationale: str = Field(min_length=1, max_length=600)
    editing: Literal['authored', 'cut'] = 'authored'
    sound: Literal['authored', 'native_only'] = 'authored'

    @model_validator(mode='after')
    def coherent_selection(self):
        if not has_evidence(self.evidence) or not self.rationale.strip():
            raise ValueError('创作方向需要原文依据和取舍理由')
        if len(set(self.methods)) != len(self.methods) or len(set(self.references)) != len(self.references):
            raise ValueError('创作方法和参考不能重复')
        if not self.methods and self.editing == self.sound == 'authored':
            raise ValueError('只有参考名称而无表达选择时应省略创作方向')
        return self


def creative_direction_projection(shot: dict) -> dict:
    raw = (shot.get('production_slot') or {}).get('creative_direction')
    return CreativeDirectionSpec.model_validate(raw).model_dump() if raw is not None else {}


def creative_direction_instruction(direction: dict) -> str:
    """安全请求仅投影受控方法；不重新引入需求、引用或取舍理由等自由文本。"""
    if not direction:
        return ''
    selected = CreativeDirectionSpec.model_validate(direction)
    parts = [METHODS[method] for method in selected.methods]
    if selected.editing == 'cut':
        parts.append('Compose shot boundaries for direct cuts; do not bake fades or dissolves into the footage.')
    if selected.sound == 'native_only':
        parts.append('Use only scene-native sound, with no musical score or added background music.')
    return ('Creative direction v1: ' + ' '.join(parts)
            + ' Apply relevant methods to this authored shot, not every method to every beat. '
            'Explicit subject tempo, playback treatment, camera blocking, exact choreography, '
            'reference motion, ordered outcomes and allocated duration take precedence. '
            'Do not add story events, change the ending or infer slow motion from a cinematic reference.')


def creative_authoring_context(direction: dict) -> dict:
    selected = CreativeDirectionSpec.model_validate(direction).model_dump()
    return {**selected, 'guidance': creative_direction_instruction(selected)}


def creative_catalog_prompt() -> str:
    catalog = {key: {'name': value[0], 'candidate_methods': value[1]} for key, value in REFERENCES.items()}
    return ('\ncreative_direction是可选的表达选择，没有明确风格/方法/剪辑/声音要求时返回null；'
            '普通cinematic或电影感不要求套用导演。理解完整语义、否定和用户优先级，'
            '不要根据题材自动套用风格，不要把所有候选方法相加。可用目录：\n'
            + json.dumps({'references': catalog, 'methods': METHODS}, ensure_ascii=False)
            + '\n仅选与任务相符的0至4种方法和0至2个参考ID；未收录的参考不得伪称已收录，'
            '可按用户明确描述的特点选择通用方法。导演只是部分创作方法的参考，不是整部作品的复制模板。'
            'evidence按当前来源协议选择request或style_input中的依据；rationale简述选择与冲突取舍，'
            '显式要求优先于风格惯例。editing为authored，只有明确要求硬切/不叠化时选cut；'
            'sound为authored，只有明确要求不配乐/仅现场声时选native_only。'
            '风格不推导动作数量、速度、总时长或慢镜头，这些由既有独立字段负责。')


def creative_catalog_fingerprint() -> str:
    payload = {'methods': METHODS, 'references': REFERENCES,
               'schema': CreativeDirectionSpec.model_json_schema(), 'interpretation': creative_catalog_prompt()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


AUTHORING_DIRECTION = '''
creative_direction是全片的表达方向，guidance是其受控解释；rationale用于理解取舍。
按当前戏剧任务选用相关方法，不要求每个镜头或节拍逐项展示所有方法。先尊重用户明确要求、叙事因果和动作执行条件，再安排表演、空间、光影、声音与剪辑。
参考名称不替代具体创作；用当前人物和环境表达这些方法，不移植原作人物、情节或标志性道具。不得为了风格增加剧情结果、拉慢动作填时长、改写动作配额或精确参考路径。
editing=cut时为硬切设计出入镜；sound=native_only时只设计现场声音，不规划配乐。其余选择由当前场景创作决定。
'''
