"""从分镜到生成、参考和评审的站位约束回归；禁止用生成结果猜测计划。"""

import json
from pathlib import Path

import pytest

import config
from pipeline.motion_assessment import assess_document, blank_rating, contract_digest
from pipeline.motion_plan import BONES, MotionPlan, compile_motion_previs
from pipeline.provider_prompt import compile_normal_provider_prompt, compile_policy_safe_prompt
from pipeline.readiness import provider_submission_issues
from pipeline.semantic_review import _review_contract
from pipeline.visual_invariants import compile_visual_invariants, visual_invariants_fingerprint
from tools.motion_renderer import draw_motion_frame


@pytest.fixture
def source():
    directory = Path(__file__).resolve().parents[1] / "examples/contact_previs"
    storyboard = json.loads((directory / "storyboard.json").read_text())
    return storyboard, MotionPlan.model_validate_json((directory / "motion.json").read_text())


def test_facing_and_fixed_camera_survive_both_prompt_profiles_and_review(source):
    storyboard, _ = source
    shot = storyboard["shots"][0]
    projection = compile_visual_invariants(shot).review_projection
    assert projection["body_orientations"] == {
        "attacker": "profile toward screen-right", "defender": "profile toward screen-left",
    }
    assert projection["camera_movement"] == "fixed"
    normal = compile_normal_provider_prompt(shot, storyboard, has_observed_start=False)
    safe = compile_policy_safe_prompt(shot, storyboard=storyboard, has_state_reference=False,
                                      image_role=None, reference_count=0)
    for prompt in (normal, safe):
        assert "attacker=profile toward screen-right" in prompt
        assert "defender=profile toward screen-left" in prompt
        assert "camera remains fixed" in prompt
    assert _review_contract(shot)["visual_invariants"] == projection


def test_preflight_detects_lost_facing_constraint(source):
    storyboard, _ = source
    shot = storyboard["shots"][0]
    prompt = compile_normal_provider_prompt(shot, storyboard, has_observed_start=False)
    prompt = prompt.replace("attacker=profile toward screen-right", "attacker=left")
    issues = provider_submission_issues(prompt, shot, storyboard, prompt_profile="normal",
                                        model=config.SEEDANCE_MODEL, resolution="480p", duration=4)
    assert any("Positive Locks" in issue for issue in issues)


def test_orientation_changes_invalidate_acceptance_fingerprint(source):
    storyboard, _ = source
    shot = storyboard["shots"][0]
    before = visual_invariants_fingerprint(shot)
    shot["blocking"]["defender"]["body_orientation"] = "facing screen-right"
    assert visual_invariants_fingerprint(shot) != before


def test_opposite_facing_with_identical_coordinates_is_rejected(source):
    storyboard, plan = source
    rig = plan.rigs[1]
    rig.facing = "right"
    for pose in rig.keyframes:
        pose.angles = {name: ((180 - pose.angles.get(name, default) + 180) % 360) - 180
                       for name, (_, _, default) in BONES.items()}
    report = compile_motion_previs(storyboard["shots"][0], "16:9", plan)["diagnostics"]
    assert report["status"] == "invalid"
    assert any("朝向" in issue for issue in report["issues"])


def test_rig_order_does_not_change_identity_colors(source):
    storyboard, plan = source
    compiled = compile_motion_previs(storyboard["shots"][0], "16:9", plan)
    original = draw_motion_frame(plan, compiled["frames"][0], annotated=False)
    reordered = plan.model_copy(deep=True)
    reordered.rigs.reverse()
    other = compile_motion_previs(storyboard["shots"][0], "16:9", reordered)
    assert draw_motion_frame(reordered, other["frames"][0], annotated=False).tobytes() == original.tobytes()


def test_reference_visibly_encodes_side_facing(source):
    storyboard, plan = source
    frame = compile_motion_previs(storyboard["shots"][0], "16:9", plan)["frames"][0]
    image = draw_motion_frame(plan, frame, annotated=False)
    for rig in plan.rigs:
        x, y = frame["entities"][rig.entity]["head"]
        direction = 1 if rig.facing == "right" else -1
        # 在头部圆形轮廓外检查朝向鼻尖，不靠标注文字推断朝向。
        assert image.getpixel((round((x + direction * 0.032) * 360), round(y * 360))) != (16, 24, 32)
        assert image.getpixel((round((x - direction * 0.032) * 360), round(y * 360))) == (16, 24, 32)


def test_review_cannot_pass_facing_error_with_good_contact_timing():
    from pipeline.motion_assessment import REVIEW_VERSION
    entry = {"sha256": "b" * 64, "frames": 96}
    contract = {"visual_invariants": {"camera_movement": "fixed"}}
    rating = blank_rating("a" * 32, entry["sha256"], 96)
    rating.update(assessment="reviewed", identity_ok=True, scope_ok=True, endpoint_ok=True,
                  causal_link_ok=True, contact_frame=48, reaction_frame=55,
                  visual_invariants_ok=False)
    doc = {"version": REVIEW_VERSION, "reviewer_id": "test", "contract": contract,
           "contract_sha256": contract_digest(contract), "ratings": [rating]}
    assert assess_document(doc, {"a" * 32: entry}, contract)["verdicts"]["a" * 32] == "fail"
    del rating["visual_invariants_ok"]
    with pytest.raises(ValueError):
        assess_document(doc, {"a" * 32: entry}, contract)


def test_v2_review_remains_historical_and_cannot_gain_missing_checks():
    entry = {"sha256": "b" * 64, "frames": 96}
    contract = {"action": "contact", "visual_invariants": {"camera_movement": "fixed"}}
    previous = {"action": "contact"}
    rating = {"review_id": "a" * 32, **entry, "assessment": "reviewed", "identity_ok": True,
              "scope_ok": True, "endpoint_ok": True, "causal_link_ok": True,
              "contact_frame": 48, "reaction_frame": 55, "notes": ""}
    doc = {"version": "motion-review-v2", "reviewer_id": "historic", "contract": previous,
           "contract_sha256": contract_digest(previous), "ratings": [rating]}
    result = assess_document(doc, {"a" * 32: entry}, contract)
    assert result["version"] == "motion-review-v2"
    assert result["verdicts"]["a" * 32] == "pass"


@pytest.mark.parametrize("value", [None, [], {"x": None}, {"x": {"body_orientation": []}},
                                   {"x": {"body_orientation": "facing screen-right; EXTRA_ACTION"}}])
def test_orientation_boundary_does_not_promote_unrecognized_prose(value):
    from pipeline.visual_invariants import safe_body_orientations
    assert safe_body_orientations(value) == {}


@pytest.mark.parametrize("movement", ["fixed", "static", "locked_off"])
def test_fixed_camera_aliases_have_one_acceptance_identity(source, movement):
    storyboard, _ = source
    shot = storyboard["shots"][0]
    original = visual_invariants_fingerprint(shot)
    shot["camera"]["primary_movement"] = movement
    assert visual_invariants_fingerprint(shot) == original


def test_old_and_current_review_cannot_form_new_consensus():
    from pipeline.motion_assessment import REVIEW_VERSION, compare_assessments
    old = {"version": "motion-review-v2", "reviewer_id": "old"}
    new = {"version": REVIEW_VERSION, "reviewer_id": "new"}
    with pytest.raises(ValueError, match="新版"):
        compare_assessments(old, new, [])
