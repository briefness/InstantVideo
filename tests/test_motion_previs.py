"""Offline contracts and failure boundaries for skeletal previsualization."""

import copy
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from pipeline.motion_plan import (
    BONES, MotionPlan, compile_motion_previs, motion_source_binding, sample_motion_rig,
)
from tools.motion_previs import build_motion_previs, main, read_previs_json, select_motion_shot


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "contact_previs"


@pytest.fixture
def inputs():
    return (json.loads((EXAMPLE / "storyboard.json").read_text()),
            json.loads((EXAMPLE / "motion.json").read_text()))


def compile_example(storyboard, raw):
    shot, ratio = select_motion_shot(storyboard, raw["shot_id"])
    return compile_motion_previs(shot, ratio, MotionPlan.model_validate(raw))


def test_example_is_deterministic_binds_contract_and_preserves_inputs(inputs):
    storyboard, raw = inputs
    before = copy.deepcopy(inputs)
    compiled = compile_example(storyboard, raw)
    assert compiled == compile_example(storyboard, raw)
    assert inputs == before
    assert compiled["frame_count"] == 96
    assert compiled["frames"][-1]["time_seconds"] == 95 / 24
    report = compiled["diagnostics"]
    assert report["status"] == "valid_geometry"
    assert report["first_contact_frame"] == 48
    assert report["first_target_motion_frame"] == 55
    assert report["contact_distance"] == pytest.approx(0, abs=1e-14)
    assert report["generation_quality"] == report["semantic_acceptance"] == "not_evaluated"


def test_interpolated_frames_preserve_lengths_and_pixel_contact(inputs):
    storyboard, raw = inputs
    plan = MotionPlan.model_validate(raw)
    compiled = compile_example(storyboard, raw)
    for sample in compiled["frames"]:
        for rig in plan.rigs:
            joints = sample["entities"][rig.entity]
            for name, (parent, length, _) in BONES.items():
                assert math.dist(joints[parent], joints[name]) == pytest.approx(length, abs=1e-12)
    contact = compiled["frames"][48]["entities"]
    assert contact["attacker"]["lead_hand"][0] * 360 == pytest.approx(314.4)
    assert contact["defender"]["lead_hand"][1] * 360 == pytest.approx(144)


@pytest.mark.parametrize("field,value", [
    ("width", 999999), ("width", 641), ("height", True), ("fps", 120),
    ("shot_id", True), ("version", "future"), ("source_sha256", "short"),
])
def test_malformed_or_excessive_plan_rejected(inputs, field, value):
    _, raw = inputs
    raw[field] = value
    with pytest.raises(ValidationError):
        MotionPlan.model_validate(raw)


@pytest.mark.parametrize("bad_angle", [float("nan"), float("inf"), "45", True, 181])
def test_untrusted_angles_rejected(inputs, bad_angle):
    _, raw = inputs
    raw["rigs"][0]["keyframes"][0]["angles"]["head"] = bad_angle
    with pytest.raises(ValidationError):
        MotionPlan.model_validate(raw)


def test_duplicate_and_missing_keyframes_rejected(inputs):
    storyboard, raw = inputs
    raw["rigs"][0]["keyframes"][1]["frame"] = 0
    with pytest.raises(ValidationError):
        MotionPlan.model_validate(raw)
    raw["rigs"][0]["keyframes"][1]["frame"] = 18
    raw["rigs"][0]["keyframes"][-1]["frame"] = 94
    with pytest.raises(ValueError, match="最后关键帧"):
        compile_example(storyboard, raw)


def test_unknown_bones_and_participants_rejected(inputs):
    storyboard, raw = inputs
    raw["rigs"][0]["lengths"] = {"unknown": 0.2}
    with pytest.raises(ValidationError):
        MotionPlan.model_validate(raw)
    raw["rigs"][0].pop("lengths")
    raw["rigs"][0]["entity"] = "stranger"
    with pytest.raises(ValueError, match="人物"):
        compile_example(storyboard, raw)


@pytest.mark.parametrize("change", ["duration", "camera", "contract"])
def test_stale_binding_rejected(inputs, change):
    storyboard, raw = inputs
    shot = storyboard["shots"][0]
    if change == "duration":
        shot["duration"] = 5
    elif change == "camera":
        shot["camera"]["start_framing"] = "medium shot"
    else:
        shot["action_beats"][0]["visible_result"] = "different endpoint"
    with pytest.raises(ValueError, match="绑定不匹配"):
        compile_example(storyboard, raw)


@pytest.mark.parametrize("kind", ["setup", "path", "group", "camera", "continuity", "third_party"])
def test_unsupported_shots_fail_closed(inputs, kind):
    storyboard, _ = inputs
    shot = storyboard["shots"][0]
    if kind == "setup":
        shot["interaction_geometry"]["effect_phase"] = "setup"
    elif kind == "path":
        shot["interaction_geometry"]["interaction_mode"] = "directed_path"
    elif kind == "group":
        shot["interaction_geometry"]["outcome_scope"] = "all"
    elif kind == "camera":
        shot["camera"]["primary_movement"] = "pan"
    elif kind == "continuity":
        shot["continuity_from_previous"] = "seamless"
    else:
        shot["required_visible_entities"].append("bystander")
    with pytest.raises(ValueError):
        select_motion_shot(storyboard, 1)


@pytest.mark.parametrize("kind", ["event_contract", "multiple_peaks", "multiple_interactions", "key_props", "continuity_props"])
def test_v1_compiler_cannot_collapse_events_or_ignore_props(inputs, kind):
    storyboard, raw = inputs
    shot = storyboard["shots"][0]
    if kind == "event_contract":
        shot["action_beats"][0].update(interaction_geometry=shot["interaction_geometry"],
                                     effect_at_seconds=2, reaction_at_seconds=2.5)
    elif kind == "multiple_peaks":
        shot["action_beats"].append(copy.deepcopy(shot["action_beats"][0]))
    elif kind == "multiple_interactions":
        shot["action_beats"][0]["activity"] = "interaction"
        shot["action_beats"].append(copy.deepcopy(shot["action_beats"][0]))
        shot["action_beats"][0]["phase"] = "trigger"
    else:
        shot[kind] = ["training staff"]
    raw["source_sha256"] = motion_source_binding(shot, "16:9")
    with pytest.raises(ValueError, match="v1"):
        compile_motion_previs(shot, "16:9", MotionPlan.model_validate(raw))


@pytest.mark.parametrize("end", [3, 2.1])
def test_compiler_and_legacy_cli_use_contract_interaction_window(inputs, tmp_path, end):
    board, raw = inputs
    shot = board["shots"][0]
    base = shot["action_beats"][0]
    shot["action_beats"] = [
        {**base, "phase": "trigger", "activity": "transition", "start_seconds": 0, "end_seconds": 1},
        {**base, "activity": "interaction", "start_seconds": 1, "end_seconds": end},
        {**base, "phase": "aftermath", "activity": "hold", "start_seconds": end, "end_seconds": 4},
    ]
    raw["source_sha256"] = motion_source_binding(shot, "16:9")
    diagnostics = compile_motion_previs(shot, "16:9", MotionPlan.model_validate(raw))["diagnostics"]
    assert diagnostics["status"] == ("valid_geometry" if end == 3 else "invalid")
    board_path, plan_path = tmp_path / "board.json", tmp_path / "plan.json"
    board_path.write_text(json.dumps(board))
    plan_path.write_text(json.dumps(raw))
    assert main(["validate", "--storyboard", str(board_path), "--plan", str(plan_path)]) == (0 if end == 3 else 2)
    if end != 3:
        output = tmp_path / "output"
        assert main(["render", "--storyboard", str(board_path), "--plan", str(plan_path), "--output", str(output)]) == 2
        assert not output.exists()


def test_one_frame_geometry_tolerance_cannot_cross_contract_window(inputs):
    board, raw = inputs
    shot = board["shots"][0]
    start = 49 / 24
    base = shot["action_beats"][0]
    shot["action_beats"] = [
        {**base, "phase": "trigger", "activity": "transition", "start_seconds": 0, "end_seconds": start},
        {**base, "activity": "interaction", "start_seconds": start, "end_seconds": 3},
        {**base, "phase": "aftermath", "activity": "hold", "start_seconds": 3, "end_seconds": 4},
    ]
    raw["contact"]["frame"] = 49
    raw["source_sha256"] = motion_source_binding(shot, "16:9")
    diagnostics = compile_motion_previs(shot, "16:9", MotionPlan.model_validate(raw))["diagnostics"]
    assert diagnostics["first_contact_frame"] == 48
    assert diagnostics["issues"] == ["骨架实际接触和反应没有同时落在合同的交互时窗内"]


def test_no_contact_is_not_accepted(inputs):
    storyboard, raw = inputs
    for pose in raw["rigs"][1]["keyframes"]:
        pose["root"][0] += 0.1
    report = compile_example(storyboard, raw)["diagnostics"]
    assert report["status"] == "invalid"
    assert report["first_contact_frame"] is None
    assert any("没有相遇" in issue for issue in report["issues"])


def test_target_reacting_before_contact_is_not_accepted(inputs):
    storyboard, raw = inputs
    raw["rigs"][1]["keyframes"][1]["root"][0] -= 0.01
    report = compile_example(storyboard, raw)["diagnostics"]
    assert report["status"] == "invalid"
    assert report["first_target_motion_frame"] < 48
    assert any("之前" in issue for issue in report["issues"])


def test_no_reaction_or_clipped_skeleton_is_not_accepted(inputs):
    storyboard, raw = inputs
    for pose in raw["rigs"][1]["keyframes"]:
        pose["root"] = [0.6825, 0.63]
        pose["angles"] = {"lead_elbow": 0.0, "lead_hand": 0.0}
    report = compile_example(storyboard, raw)["diagnostics"]
    assert any("缺少" in issue for issue in report["issues"])
    for pose in raw["rigs"][0]["keyframes"]:
        pose["root"][1] = 0.1
    report = compile_example(storyboard, raw)["diagnostics"]
    assert report["out_of_frame_count"] == 96


def test_contact_after_end_and_non_integer_duration_rejected(inputs):
    storyboard, raw = inputs
    raw["contact"]["reaction_frame"] = 95
    with pytest.raises(ValueError, match="收势"):
        compile_example(storyboard, raw)
    storyboard["shots"][0]["duration"] = "4"
    with pytest.raises(ValueError, match="整数"):
        select_motion_shot(storyboard, 1)


def test_sampling_outside_timeline_is_rejected(inputs):
    _, raw = inputs
    rig = MotionPlan.model_validate(raw).rigs[0]
    with pytest.raises(ValueError):
        sample_motion_rig(rig, 96, 640, 360)


def test_invalid_geometry_does_not_render_or_publish(inputs, tmp_path, monkeypatch):
    storyboard, raw = inputs
    for pose in raw["rigs"][0]["keyframes"]:
        pose["root"][1] = 0.1
    def forbidden(*args, **kwargs):
        pytest.fail("invalid input reached renderer")
    monkeypatch.setattr("tools.motion_previs.render_motion_media", forbidden)
    with pytest.raises(ValueError, match="预演检查失败"):
        build_motion_previs(storyboard, MotionPlan.model_validate(raw), tmp_path / "result")
    assert not list(tmp_path.iterdir())


def test_existing_output_is_never_overwritten(inputs, tmp_path):
    storyboard, raw = inputs
    with pytest.raises(FileExistsError):
        build_motion_previs(storyboard, MotionPlan.model_validate(raw), tmp_path)
    assert not list(tmp_path.iterdir())


def test_render_failure_cleans_staging_without_publishing(inputs, tmp_path, monkeypatch):
    storyboard, raw = inputs
    def broken(*args, **kwargs):
        raise RuntimeError("encoder failed")
    monkeypatch.setattr("tools.motion_previs.render_motion_media", broken)
    with pytest.raises(RuntimeError, match="encoder"):
        build_motion_previs(storyboard, MotionPlan.model_validate(raw), tmp_path / "result")
    assert not list(tmp_path.iterdir())


def test_json_limits_and_invalid_root(tmp_path):
    large = tmp_path / "large.json"
    large.write_bytes(b" " * (2 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="2 MiB"):
        read_previs_json(large)
    large.write_text("[]")
    with pytest.raises(ValueError, match="根节点"):
        read_previs_json(large)


def test_duplicate_shot_ids_rejected(inputs):
    storyboard, _ = inputs
    storyboard["shots"].append(copy.deepcopy(storyboard["shots"][0]))
    with pytest.raises(ValueError, match="唯一"):
        select_motion_shot(storyboard, 1)


def test_offline_cli_without_provider_imports(capsys):
    assert main(["validate", "--storyboard", str(EXAMPLE / "storyboard.json"),
                 "--plan", str(EXAMPLE / "motion.json")]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["generation_quality"] == "not_evaluated"
    result = subprocess.run([
        "python", "-c",
        "import sys; import tools.motion_previs; "
        "assert 'tools.seedance_api' not in sys.modules; "
        "assert 'pipeline.generator' not in sys.modules",
    ], capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required")
def test_real_offline_render_is_complete_and_hashed(inputs, tmp_path):
    storyboard, raw = inputs
    output = build_motion_previs(storyboard, MotionPlan.model_validate(raw), tmp_path / "preview")
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["state_kind"] == "planned_only"
    assert manifest["source_sha256"] == motion_source_binding(storyboard["shots"][0], "16:9")
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    report = json.loads((output / "diagnostics.json").read_text())
    assert int(report["encoded_video"]["nb_frames"]) == 96
    assert float(report["encoded_video"]["duration"]) == 4
    protocol = json.loads((output / "experiment.json").read_text())
    assert protocol["paid_take_budget"] == 0
    assert protocol["arms"][1]["submission_ready"] is False
    assert not (output / "run_manifest.json").exists()
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("payload", ['{"shots": [], "shots": []}', '[' * 2000 + '0' + ']' * 2000])
def test_ambiguous_or_deep_json_rejected(tmp_path, payload):
    path = tmp_path / "input.json"
    path.write_text(payload)
    with pytest.raises(ValueError):
        read_previs_json(path)


def test_directory_is_not_treated_as_json(tmp_path):
    with pytest.raises(ValueError, match="普通文件"):
        read_previs_json(tmp_path)


@pytest.mark.parametrize("metadata", [
    {"name": "defender", "reference_mode": "group"},
    {"name": "attacker", "mobility": "tracked"},
])
def test_character_metadata_prevents_group_or_non_biped_mapping(inputs, metadata):
    storyboard, _ = inputs
    storyboard["characters"] = [metadata]
    with pytest.raises(ValueError, match="双足"):
        select_motion_shot(storyboard, 1)


def test_vertical_stage_preserves_contact_geometry(inputs):
    storyboard, raw = inputs
    storyboard["ratio"] = "9:16"
    raw.update(width=360, height=640)
    raw["source_sha256"] = motion_source_binding(storyboard["shots"][0], "9:16")
    for rig in raw["rigs"]:
        rig["lengths"] = {name: length * 0.55 for name, (_, length, _) in BONES.items()}
        for pose in rig["keyframes"]:
            if rig["entity"] == "defender":
                pose["root"][0] = 0.3 + 0.68 * 0.55
    report = compile_example(storyboard, raw)["diagnostics"]
    assert report["status"] == "valid_geometry"
    assert report["contact_distance"] == pytest.approx(0, abs=1e-14)


def test_rig_angles_interpolate_across_wrap_without_full_rotation(inputs):
    _, raw = inputs
    rig = MotionPlan.model_validate(raw).rigs[0]
    rig.keyframes = rig.keyframes[:2]
    rig.keyframes[0].angles["lead_hand"] = 170
    rig.keyframes[1].angles["lead_hand"] = -170
    points = sample_motion_rig(rig, 9, 640, 360)
    assert points["lead_hand"][0] - points["lead_elbow"][0] == pytest.approx(-0.14)
    assert points["lead_hand"][1] == pytest.approx(points["lead_elbow"][1])


def test_symlink_output_is_not_replaced(inputs, tmp_path):
    storyboard, raw = inputs
    link = tmp_path / "linked"
    link.symlink_to(tmp_path / "missing", target_is_directory=True)
    with pytest.raises(FileExistsError):
        build_motion_previs(storyboard, MotionPlan.model_validate(raw), link)
    assert link.is_symlink()


def test_concurrent_output_reservation_preserves_winner(inputs, tmp_path, monkeypatch):
    storyboard, raw = inputs
    destination = tmp_path / "result"
    def concurrent_writer(*args):
        destination.mkdir()
        (destination / "winner.txt").write_text("keep me")
        return {}
    monkeypatch.setattr("tools.motion_previs.render_motion_media", concurrent_writer)
    with pytest.raises(FileExistsError):
        build_motion_previs(storyboard, MotionPlan.model_validate(raw), destination)
    assert (destination / "winner.txt").read_text() == "keep me"
    assert list(tmp_path.iterdir()) == [destination]


def test_publication_failure_rolls_back_only_its_files(inputs, tmp_path, monkeypatch):
    import os
    storyboard, raw = inputs
    real_link = os.link
    calls = []
    def fail_second(source, destination):
        calls.append(destination)
        if len(calls) == 2:
            raise OSError("disk full")
        real_link(source, destination)
    monkeypatch.setattr("tools.motion_previs.os.link", fail_second)
    monkeypatch.setattr("tools.motion_previs.render_motion_media", lambda *args: {})
    with pytest.raises(OSError, match="disk full"):
        build_motion_previs(storyboard, MotionPlan.model_validate(raw), tmp_path / "result")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("failure", ["timeout", "malformed_probe", "mismatched_probe"])
def test_encoder_failures_do_not_publish(inputs, tmp_path, monkeypatch, failure):
    storyboard, raw = inputs
    def fake_run(command, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 90)
        payload = {} if failure == "malformed_probe" else {"streams": [{
            "width": 640, "height": 360, "avg_frame_rate": "24/1",
            "nb_frames": "95", "duration": "4",
        }]}
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload))
    monkeypatch.setattr("tools.motion_renderer.shutil.which", lambda name: name)
    monkeypatch.setattr("tools.motion_renderer.subprocess.run", fake_run)
    with pytest.raises(RuntimeError):
        build_motion_previs(storyboard, MotionPlan.model_validate(raw), tmp_path / "result")
    assert not list(tmp_path.iterdir())
