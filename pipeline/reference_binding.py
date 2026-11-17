"""参考图传递场景事实，剪辑边界决定是否还必须传递上一镜的像素构图。"""
from pipeline.production_plan import reference_policy
from pipeline.storyboard import _should_use_previous_tail_reference


def observed_start_state(shot: dict, observed: dict) -> dict:
    """世界状态继承真实观察；有意切镜的摄影状态由当前镜头拥有。"""
    planned = shot.get('start_state') or {}
    return {key: str(planned.get(key, '') if key == 'camera'
                    and shot.get('continuity_from_previous') == 'intentional_cut'
                    else observed.get(key) or planned.get(key, ''))
            for key in ('location', 'subject', 'action_phase', 'camera', 'screen_direction',
                        'pose_and_gaze', 'prop_state', 'open_motion', 'lighting')}


def resolve_image_references(shot: dict, tail: str | None, previous: dict | None,
                             identities: list[str]) -> tuple[list[str], str | None]:
    continuity = shot.get('continuity_from_previous')
    identities = list(dict.fromkeys(identities))
    if continuity == 'seamless' and tail:
        return [tail], 'first_frame'
    if continuity == 'intentional_cut':
        use_tail = bool(tail) and (
            (reference_policy(shot) == 'identity_only' and not identities)
            or _should_use_previous_tail_reference(shot, previous, has_identity_reference=bool(identities)))
        # 切镜保留状态参考，但不能将旧构图设为新镜的精确首帧。
        refs = list(dict.fromkeys(([tail] if use_tail else []) + identities))
        if len(refs) > 9:
            raise ValueError('切镜状态与身份参考超过 9 张；未丢弃参考或提交生成')
        return refs, 'reference_image' if refs else None
    if tail:
        return [tail], 'first_frame'
    if len(identities) > 9:
        raise ValueError('身份参考超过 9 张；未丢弃参考或提交生成')
    return identities, 'reference_image' if identities else None
