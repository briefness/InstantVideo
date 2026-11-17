"""动作语义的公共词汇；类型约束与各阶段说明共用，不根据正文猜分类。"""
from typing import Literal, get_args


Activity = Literal['interaction', 'performance', 'transition', 'hold']
IndependentActivity = Literal['performance', 'transition', 'hold']
ResponseKind = Literal['impact', 'evasion', 'deflection', 'state_change']
ACTIVITIES = get_args(Activity)
INDEPENDENT_ACTIVITIES = get_args(IndependentActivity)
RESPONSE_MODES = {
    'impact': ('direct_contact', 'directed_path', 'area_effect', 'indirect_effect'),
    'evasion': ('directed_path',),
    'deflection': ('direct_contact', 'directed_path'),
    'state_change': ('direct_contact', 'directed_path', 'area_effect', 'indirect_effect'),
}

AUTHORING_SEMANTICS = (
    '动作单元表达执行者的连续行为及其可见结果。actor是已登记的人物或适用的场景主体。'
    'performance表示主体自身的主动表演或变化，没有单独对外部对象施加作用；'
    'target在performance中仅表示关注对象，可留空，不表示对象受到作用，不填写effect/response/resolution。'
    'transition表示位置或场景进入状态的过渡；hold表示已有状态的持续展示；这两类也不产生独立外部作用。'
    'interaction表示确实作用于另一个已登记对象，target必须是实际受影响者，不是旁观者或空间参照物；'
    'effect记录作用方式，response记录实际结果机制。state_change表示受控操作或非冲击性的状态改变，'
    '结果是目标的可见变化，不要求受击或防御；impact表示冲击命中，evasion表示避开作用路径，'
    'deflection表示改变入射路径。不要把受击者的反应反写成对发起者的新作用。'
    '先选择符合实际行为的语义再表达因果，不得为了使用某种类型而发明动作或目标反应。'
)

REVIEW_SEMANTICS = (
    'classification_matches_description比较既定activity类型与原描述：performance是主体自身的主动表演或变化，'
    'target仅为关注对象，不承担外部作用或反应；transition是位置/进入状态的过渡，hold是已有状态的持续。'
    '这些类型均无需外部冲击，但不能掩盖描述中真正作用于其他对象的行为。'
    'interaction需要对实际target的作用及相应结果；state_change要求目标发生受控、非冲击的可见状态变化，'
    '不能凭空要求受击、损伤或闪避；impact、evasion、deflection仍分别要求命中、避开或拨挡的原因结果。'
    '不能把关注对象当作被作用者，也不能把空间参照物当作实际受影响对象。'
    '所有类型仍需实现用户明确事件、正确参与者、因果和跨镜依赖；标签合法不等于内容正确。'
)

STATE_CHANGE_EXECUTION = (
    'response_kind=state_change: show the controlled operation or non-impulsive change and the '
    'resulting visible state of the actual target. The response is that state change, not a hit, '
    'injury or defensive reaction. effect_at marks the operation onset; reaction_at marks the '
    'target state becoming established. Preserve the authored method and required result.'
)
