"""Provider-facing prompt profiles compiled from the Production Contract."""

from __future__ import annotations

from pipeline.causality import compile_action_contract, interaction_geometry, event_execution_instruction
from pipeline.action_timing import event_shot
from pipeline.performance import performance_instruction, outcome_text, allocated_interaction_target, local_event_execution, scene_frame_execution
from pipeline.production_plan import classify_framing
from pipeline.staging_direction import staging_projection
from pipeline.generation_instructions import VERSIONS as SHOOTING_VERSIONS, shooting_projection, shooting_instruction_parts
from pipeline.visual_invariants import (
    compile_visual_invariants,
    initial_screen_positions,
)


_FRAMING_LABELS = {
    "wide": "wide shot",
    "medium": "medium shot",
    "close_detail": "close detail shot",
}


def compile_generation_preview(storyboard: dict, *, model: str) -> dict:
    """正式规划出口的只读投影，不伪造未来的参考图或观察状态。"""
    from pipeline.film_rhythm import planned_interaction_seconds
    from pipeline.readiness import provider_submission_issues
    rows = []
    for shot in storyboard['shots']:
        prompts = {
            'normal': compile_normal_provider_prompt(shot, storyboard, has_observed_start=False),
            'policy_safe': compile_policy_safe_prompt(shot, storyboard=storyboard,
                has_state_reference=False, image_role=None, reference_count=0),
        }
        rows.append({'shot_id': shot['shot_id'], 'duration': shot['duration'],
            **({'execution_projection': shooting_projection(shot, storyboard)}
               if shot.get('generation_execution') in SHOOTING_VERSIONS else {}),
            'allocated_interaction_seconds': round(planned_interaction_seconds([shot]), 3),
            'reference_policy': (shot.get('production_slot') or {}).get('reference_policy'),
            'profiles': {profile: {'prompt': prompt,
                'staging_source_fingerprint': staging_projection(shot, profile=profile)['fingerprint'],
                'omitted_staging_fields': staging_projection(shot, profile=profile)['omitted_fields'],
                'preflight_issues': provider_submission_issues(prompt, shot, storyboard,
                    prompt_profile=profile, model=model, resolution=storyboard['resolution'],
                    duration=shot['duration'])} for profile, prompt in prompts.items()}})
    return {'version': 'generation-preview-v1', 'scope': 'planned-text-projection',
            'model': model, 'resolution': storyboard['resolution'], 'aspect_ratio': storyboard['aspect_ratio'],
            'runtime_bindings': 'deferred: accepted prior state and identity/reference assets are not available in planning',
            'shots': rows}


def has_explicit_action_contract(shot: dict) -> bool:
    """Whether this shot opts into phase-owned action compilation."""
    geometry = interaction_geometry(shot)
    return str(geometry.get("effect_phase", "")).strip() in {
        "none", "setup", "active", "aftermath",
    }


def compile_normal_provider_prompt(
    shot: dict,
    storyboard: dict,
    *,
    has_observed_start: bool,
) -> str:
    """Compile a normal provider prompt without re-consuming action prose.

    Explicit Action Contract shots intentionally do not inherit ``prompt_en``,
    ``scene_description``, or ``negative_prompt``.  Those fields predate the
    contract and can otherwise smuggle an outcome into a setup or aftermath.
    Legacy shots keep their old caller path for backward compatibility.
    """
    if not has_explicit_action_contract(shot):
        return ""

    contract = compile_action_contract(shot)
    if shot.get('generation_execution') in SHOOTING_VERSIONS:
        return _compile_shooting_prompt(shot, storyboard, contract, has_observed_start=has_observed_start)
    if scene_frame_execution(shot):
        return _compile_scene_frame_prompt(shot, storyboard, contract, has_observed_start=has_observed_start)
    parts = ["Create one coherent cinematic shot"]
    parts.extend(_normal_visual_parts(shot, storyboard))
    visual = _structured_text(shot.get('visual_description'))
    if visual:
        parts.append(f"Authored visual direction for setting, materials and light: {visual}")
    direction = staging_projection(shot)
    if direction['instruction']:
        parts.append(direction['instruction'])
    parts.extend(compile_visual_invariants(shot).prompt_parts)
    if has_observed_start:
        parts.append(
            "begin from the observed physical and action state in Image 1 while applying the current camera composition"
        )
    elif contract.prompt_start_state:
        parts.append(f"start exactly with {contract.prompt_start_state}")
    parts.extend(contract.prompt_parts)
    parts.extend(_normal_composition_parts(shot, has_observed_start=has_observed_start))
    policy = contract.review_projection.get('performance_policy') or {}
    if not policy.get('motion_timing'):
        parts.append("do not introduce events beyond the ordered contract" if contract.review_projection.get('action_events')
                     or (shot.get('production_slot') or {}).get('authoring_protocol') == 'scene-units-v1'
                     else "do not introduce another major action")
    return ". ".join(part for part in parts if part).strip() + "."


def _compile_shooting_prompt(shot: dict, storyboard: dict, contract, *, has_observed_start: bool) -> str:
    from pipeline.camera_contract import compile_camera
    # 保留原有边界验证及续拍路径验证，不把其协议说明再次交给视频模型。
    staging_projection(shot)
    projection = shooting_projection(shot, storyboard)
    parts = ['Create one coherent cinematic shot', *_normal_visual_parts(shot, storyboard)]
    camera_parts, _ = compile_camera(shot)
    parts.extend(p for p in compile_visual_invariants(shot).prompt_parts if p not in camera_parts)
    if has_observed_start:
        parts.append('Begin from the observed physical and action state in Image 1; apply the current camera composition')
    elif contract.prompt_start_state:
        parts.append('start exactly with ' + contract.prompt_start_state)
    parts.extend(_normal_composition_parts(shot, has_observed_start=has_observed_start))
    parts.extend(shooting_instruction_parts(projection))
    return '. '.join(p for p in parts if p).strip() + '.'


def _compile_scene_frame_prompt(shot: dict, storyboard: dict, contract, *, has_observed_start: bool) -> str:
    """必要功能/摄影、局部动作、艺术呈现分别编译；不重写作者动作或抽取新义务。"""
    from pipeline.completion_intent import intent_instruction
    from pipeline.camera_contract import compile_camera
    required = intent_instruction(shot)
    camera_parts, _ = compile_camera(shot)
    parts = ['Create one coherent cinematic shot', required, *camera_parts]
    if has_observed_start:
        parts.append('Begin from the observed physical and action state in Image 1; '
                     'the current camera selects a new view of that world state')
    elif contract.prompt_start_state:
        parts.append(f'start exactly with {contract.prompt_start_state}')
    parts.append('Execute the following local actions to realize the required functions above. '
                 'Keep necessary participants, destinations, support surfaces and causal results; '
                 'incidental gaze and resting poses may adapt. A named nearby object is not interchangeable '
                 'with the surface or recipient specified by the action')
    parts.extend(p for p in contract.prompt_parts if p != required)
    parts.extend(_normal_visual_parts(shot, storyboard))
    visual = _structured_text(shot.get('visual_description'))
    if visual:
        parts.append(f'Authored visual direction for setting, materials and light: {visual}')
    parts.append(staging_projection(shot)['instruction'])
    parts.extend(p for p in compile_visual_invariants(shot).prompt_parts if p not in camera_parts)
    parts.extend(_normal_composition_parts(shot, has_observed_start=has_observed_start))
    if not (contract.review_projection.get('performance_policy') or {}).get('motion_timing'):
        parts.append('do not introduce events beyond the ordered contract')
    return '. '.join(p for p in parts if p).strip() + '.'


def compile_policy_safe_prompt(
    shot: dict,
    *,
    storyboard: dict | None = None,
    has_state_reference: bool,
    image_role: str | None,
    reference_count: int,
    retake_instruction: str | None = None,
) -> str:
    """Compile a non-graphic provider retry without reusing rejected prose.

    The normal profile is intentionally not an input.  A moderation retry must
    be independently compiled from bounded Production Contract fields, or the
    rejected prompt will leak back through action, narrative, character, and
    negative-prompt injectors.
    """
    action_contract = compile_action_contract(shot)
    geometry = action_contract.canonical_geometry
    actor = _entity_label(geometry.get("actor"))
    target = _entity_label(geometry.get("target"))
    parts = [
        "Create one coherent cinematic shot with a restrained, non-graphic fictional treatment",
    ]
    parts.extend(_normal_visual_parts(shot, storyboard or {}))
    parts.extend(compile_visual_invariants(shot).prompt_parts)
    if retake_instruction:
        parts.append(retake_instruction)
    camera = shot.get("camera", {})
    camera = camera if isinstance(camera, dict) else {}
    framing = _FRAMING_LABELS.get(classify_framing(camera.get("start_framing")))
    if framing:
        parts.append(f"camera: stable {framing}")
    positions = [
        f"{entity}={position}"
        for entity, position in initial_screen_positions(shot)
    ]
    if positions:
        parts.append("screen positions: " + ", ".join(positions))

    events = action_contract.review_projection.get('action_events') or []
    performance = action_contract.review_projection.get('performance_policy')
    if performance and performance['version'] == 'performance-v2-outcomes':
        if shot.get('completion_policy'):
            from pipeline.action_timing import sequence_events
            from pipeline.performance import performance_review_projection
            # 生成保留全部作者事件，包括不具备独立否决权的辅助交互。
            events = performance_review_projection(shot, sequence_events(shot))['action_events']
        by_id = {event['event_id']: event for event in events}
        for index, beat in enumerate(shot['action_beats'], 1):
            from pipeline.action_timing import beat_timing_instruction
            relation = beat_timing_instruction(beat, index)
            if relation:
                parts.append(relation)
            event = by_id.get(index)
            if event:
                from pipeline.action_results import functional_outcomes
                responses = (functional_outcomes(beat) if shot.get('completion_policy') else event.get('functional_responses'))
                parts.append(f"event {index}: {_entity_label(event['actor'])} causes a readable "
                             f"{event['response_kind']} response by {_entity_label(event['target'])}; "
                             f"mode={event['interaction_geometry']['interaction_mode']}; "
                             f"scope={event['interaction_geometry']['outcome_scope']}; "
                             f"required visible states: {outcome_text(event['outcomes'])}; "
                             + (("suggested response realization: " if shot.get('completion_policy') else
                                 "complete functional response with flexible staging: ") + outcome_text(responses) + '; '
                                if responses else '')
                             +
                             f"unaffected: {event['interaction_geometry']['unaffected_behavior']}; "
                             f"{event['flow_from_previous']}"
                             + (event_execution_instruction(event) if event['response_kind'] == 'state_change' else ''))
            else:
                parts.append(f"Beat {index} {beat['activity']}: {_entity_label(beat['actor'])}; "
                             f"visible state: {beat['visible_result']}; no new physical effect")
    elif events:
        by_id = {event['event_id']: event for event in events}
        for index, beat in enumerate(action_contract.review_projection['action_beats'], 1):
            event = by_id.get(index)
            if event is None:
                # 省去自由动作描写，但保留入场/收势的身份、时间和可见状态义务。
                parts.append(
                    (f"Beat {index} {beat['activity']}: " if performance else
                     f"{beat['start_seconds']:g}-{beat['end_seconds']:g}s {beat['activity']}: ")
                    +
                    f"{_entity_label(beat['actor'])} toward {_entity_label(beat.get('target'))}; "
                    f"contracted visible state: {beat.get('visible_result', '')}; no new physical effect"
                )
                continue
            child = compile_action_contract({**event_shot(event),
                'action_semantics': shot.get('action_semantics')})
            parts.append(
                (f"event {event['event_id']}: " if performance else
                 f"event {event['event_id']} ({event['start_seconds']:g}-{event['end_seconds']:g}s), "
                 f"effect at {event['effect_at_seconds']:g}s, response at {event['reaction_at_seconds']:g}s: ")
                + _policy_action_clause(child, _entity_label(event['actor']), _entity_label(event['target']))
                + ('; ' + event_execution_instruction(event) if event_execution_instruction(event) else '')
            )
    elif (shot.get('production_slot') or {}).get('authoring_protocol') == 'scene-units-v1':
        # 生成读取编译后的动作源；观察投影省去重复答案，不代表动作可以省略。
        for index, beat in enumerate(shot['action_beats'], 1):
            parts.append(f"Beat {index} {beat['activity']}: {_entity_label(beat['actor'])}; "
                         f"{beat['start_seconds']:g}-{beat['end_seconds']:g}s; "
                         f"visible state: {beat['visible_result']}; no new physical effect")
    else:
        parts.append(_policy_action_clause(action_contract, actor, target))
    # 保留合同时间窗和用户明确结局；不复制被拒绝的自由动作描述。
    for part in action_contract.prompt_parts:
        if part.startswith('Choreography is performance-guided:') and performance:
            parts.append(performance_instruction(performance, include_brief=False,
                local_execution=(local_event_execution(shot)
                                 and performance.get('result_authority') == 'reviewed-results-v1'),
                interaction_target_seconds=allocated_interaction_target(shot)))
        elif part.startswith(('Timed action schedule:', 'Timed scene schedule:', 'Required final visible outcome:', 'Motion timing v1:',
                              'Creative direction v1:')):
            parts.append(part)
    # 功能义务来自共享编译器，不以人类可读标题作为跨模块数据协议。
    from pipeline.completion_intent import intent_instruction
    from pipeline.observed_boundary import EXECUTION_VERSION, instruction as origin_instruction
    from pipeline.generation_instructions import local_goal_instructions
    from pipeline.result_observations import completion_contract
    obligations = ('; '.join(local_goal_instructions(completion_contract(shot)))
                   if shot.get('generation_execution') == EXECUTION_VERSION else intent_instruction(shot))
    if shot.get('generation_execution') == EXECUTION_VERSION:
        parts.append(origin_instruction(shot))
    if obligations:
        if scene_frame_execution(shot):
            parts.insert(1, obligations)
        else:
            parts.append(obligations)
    from pipeline.spatial_guidance import reference_kind
    parts.extend(_reference_clauses(
        has_state_reference=has_state_reference,
        image_role=image_role,
        reference_count=reference_count,
        spatial_reference=reference_kind(shot),
    ))
    if action_contract.phase == "setup":
        parts.append(
            "show only a readable preparation endpoint and independent target motion"
        )
    elif action_contract.phase == "aftermath":
        parts.append("show only the already established state with no new action")
    else:
        parts.append(
            "show readable cause and response with abstract, broad-audience visual staging"
        )
    parts.append("stable subjects, coherent motion, no text artifacts, no added major action")
    return ". ".join(part for part in parts if part).strip() + "."


def _policy_action_clause(contract: object, actor: str, target: str) -> str:
    """Use canonical Contract values instead of reinterpreting raw geometry."""
    phase = str(getattr(contract, "phase", "none")).strip()
    mode = str(getattr(contract, "mode", "none")).strip()
    scope = str(getattr(contract, "outcome_scope", "none")).strip()
    motion = str(getattr(contract, "effect_motion", "none")).strip()
    geometry = getattr(contract, "canonical_geometry", {})
    geometry = geometry if isinstance(geometry, dict) else {}
    visible_result = str(
        getattr(contract, "contracted_visible_result", "") or ""
    ).strip()
    unaffected_behavior = str(geometry.get("unaffected_behavior", "") or "").strip()
    source = actor or "the primary subject"
    receiver = target or "the intended subject"

    if phase == "setup":
        return (
            f"setup phase: {source} visibly prepares toward {receiver}; "
            "show intent only, with no contact or target-state change"
        )
    if phase == "aftermath":
        return (
            f"aftermath phase: show the already established non-graphic state of "
            f"{receiver}; do not introduce a new physical action"
        )
    if phase != "active":
        return f"show one readable non-graphic story beat centered on {source}"

    scope_label = {
        "single": "one clearly isolated intended target within the visible effect region",
        "subset": "only the intended subset",
        "all": "the full intended group",
    }.get(scope, "only the intended scope")
    if mode == "direct_contact":
        action = "one controlled contact action with actor and target sharing the frame"
    elif mode == "directed_path":
        action = "one directed action with a clearly readable path from source to target"
    elif mode == "area_effect":
        action = "one contained area action with a clearly readable affected region"
    elif mode == "indirect_effect":
        action = "one indirect action with a clearly readable intermediary and result"
    else:
        action = "one clearly readable action"
    result_clause = (
        f"contracted visible result: {visible_result}"
        if visible_result else "show only the contract-defined visible endpoint"
    )
    boundary_clause = (
        f"subjects outside that scope keep this declared unaffected behavior: {unaffected_behavior}"
        if unaffected_behavior else "subjects outside that scope preserve their prior independent behavior"
    )
    return (
        f"active {mode or 'action'} phase, effect motion {motion}: {source} performs {action} toward {receiver}; "
        f"{scope_label}; {result_clause}; {boundary_clause}"
    )


def _normal_visual_parts(shot: dict, storyboard: dict) -> list[str]:
    """Render bounded visual metadata; never free-form action/narrative prose."""
    parts: list[str] = []
    for key, label in (("style", "visual style"), ("mood", "mood")):
        value = _structured_text(storyboard.get(key))
        if value:
            parts.append(f"{label}: {value}")

    scene_id = _entity_label(shot.get("scene_id"))
    if scene_id:
        parts.append(f"setting: {scene_id}")
    lighting = _structured_text(shot.get("lighting"))
    if lighting:
        parts.append(f"lighting: {lighting}")

    visible = [
        _entity_label(entity)
        for entity in shot.get("required_visible_entities", [])
        if _entity_label(entity)
    ]
    if visible:
        parts.append("keep these subjects clearly visible: " + ", ".join(visible))
    props = [_entity_label(prop) for prop in shot.get("key_props", []) if _entity_label(prop)]
    if props:
        label = ('scene inventory (not a demand to include every prop in frame): '
                 if scene_frame_execution(shot) else 'visible props: ')
        parts.append(label + ", ".join(props))
    parts.extend(_normal_identity_parts(shot, storyboard))

    return parts


def _normal_composition_parts(shot: dict, *, has_observed_start: bool) -> list[str]:
    camera = shot.get("camera")
    camera = camera if isinstance(camera, dict) else {}
    parts: list[str] = []
    framing = _FRAMING_LABELS.get(classify_framing(camera.get("start_framing")))
    if framing:
        parts.append(f"camera: {framing}")
    positions = [
        f"{entity}={position}"
        for entity, position in initial_screen_positions(shot)
    ]
    if positions:
        parts.append("screen positions: " + ", ".join(positions))
    composition_change = str(shot.get("composition_change", "")).strip()
    if composition_change in {"medium", "large"} and has_observed_start:
        change = (
            "a clearly different shot size or angle"
            if composition_change == "medium"
            else "unmistakably different coverage"
        )
        parts.append(f"use {change} from the supplied previous tail")
    return parts


def _normal_identity_parts(shot: dict, storyboard: dict) -> list[str]:
    """Project declared character identity only, never the shot's free-form prose."""
    requested = {
        str(name).strip()
        for name in [*shot.get("characters", []), *shot.get("required_visible_entities", [])]
        if str(name).strip()
    }
    characters = storyboard.get("characters")
    if not isinstance(characters, list):
        return []
    identities = []
    for character in characters:
        if not isinstance(character, dict):
            continue
        name = str(character.get("name", "")).strip()
        description = _structured_text(character.get("description"))
        mobility = _structured_text(character.get("mobility"))
        if not name or name not in requested:
            continue
        details = [description] if description else []
        if mobility and mobility != "unspecified":
            details.append(f"mobility {mobility}")
        if details:
            identities.append(f"declared appearance for {_entity_label(name)}: {', '.join(details)}")
    return identities


def _structured_text(value: object) -> str:
    """Normalize scalar visual metadata without claiming semantic sanitization."""
    return " ".join(str(value or "").strip().split())


def _reference_clauses(
    *,
    has_state_reference: bool,
    image_role: str | None,
    reference_count: int,
    spatial_reference: bool | str = False,
) -> list[str]:
    if spatial_reference:
        from pipeline.spatial_guidance import reference_clauses
        return reference_clauses(reference_count, image_role, has_state_reference,
                                 observed_layout=spatial_reference == 'observed-layout-v1')
    clauses: list[str] = []
    if has_state_reference:
        clauses.append(
            "Image 1 controls only the accepted prior scene state, subject placement, and pose"
        )
        if reference_count > 1:
            clauses.append(
                "remaining images control identity and appearance only, not pose or framing"
            )
    elif image_role == "reference_image" and reference_count:
        clauses.append(
            "reference images control identity and appearance only, not pose, background, or framing"
        )
    elif image_role == "first_frame" and reference_count:
        clauses.append("begin from the supplied first frame without replaying a completed action")
    return clauses


def _entity_label(value: object) -> str:
    return " ".join(
        str(value or "").strip().replace("_", " ").replace("-", " ").split()
    )
