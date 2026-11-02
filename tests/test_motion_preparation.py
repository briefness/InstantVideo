"""Production preparation uses SDK doubles, real geometry and real FFmpeg only."""
import copy
import fcntl
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import config
import pipeline.motion_preparation as preparation
from pipeline.motion_plan import motion_source_binding
from pipeline.motion_reference import probe_motion_reference


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "contact_previs"
MODEL = "doubao-seedance-2-0-mini-260615"


class NumericPlanner:
    def __init__(self, *values):
        self.values = values
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        index = len(self.calls)
        self.calls.append(kwargs)
        if index >= len(self.values):
            raise AssertionError("Unexpected planner call, including recovery")
        value = self.values[index]
        raw = value if isinstance(value, str) else json.dumps(value)
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content=raw, refusal=None))])


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError("A motion preparation test attempted network access")
    monkeypatch.setattr(httpx.Client, "send", reject)
    monkeypatch.setattr(httpx.AsyncClient, "send", reject)
    monkeypatch.setattr(config, "SEEDANCE_MODEL", MODEL)


@pytest.fixture
def inputs():
    board = json.loads((EXAMPLE / "storyboard.json").read_text())
    plan = json.loads((EXAMPLE / "motion.json").read_text())
    return board, {key: plan[key] for key in ("rigs", "contact")}


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.fail("Real FFmpeg and ffprobe are required for this integration test")
    workspace = tmp_path_factory.mktemp("automatic-preparation")
    board = json.loads((EXAMPLE / "storyboard.json").read_text())
    raw = json.loads((EXAMPLE / "motion.json").read_text())
    planner = NumericPlanner({key: raw[key] for key in ("rigs", "contact")})
    original = config.SEEDANCE_MODEL
    config.SEEDANCE_MODEL = MODEL
    try:
        report = preparation.prepare_motion_previsualization(board, workspace, client=planner)
    finally:
        config.SEEDANCE_MODEL = original
    assert report["status"] == "prepared_unbound", report
    return workspace, board, report, planner


@pytest.fixture
def prepared_copy(rendered, tmp_path):
    original, board, report, _ = rendered
    workspace = tmp_path / "production"
    shutil.copytree(original, workspace)
    return workspace, copy.deepcopy(board), copy.deepcopy(report)


def test_formal_preparation_renders_actual_video_in_same_workspace(rendered):
    workspace, board, report, planner = rendered
    assert len(planner.calls) == 1
    assert board == json.loads((EXAMPLE / "storyboard.json").read_text())
    entry = report["shots"][0]
    video = workspace / entry["reference_path"]
    media = probe_motion_reference(video.read_bytes(), 4, "16:9")
    assert int(media["nb_frames"]) == 96
    assert media["codec_name"] == "h264"
    assert entry["source_sha256"] == motion_source_binding(board["shots"][0], "16:9")
    assert entry["reference_sha256"] == hashlib.sha256(video.read_bytes()).hexdigest()
    package = workspace / entry["package"]
    plan = json.loads((package / "motion_plan.json").read_text())
    assert (plan["width"], plan["height"], plan["fps"]) == (640, 360, 24)
    assert plan["source_sha256"] == entry["source_sha256"]
    assert (package / "contact_sheet.png").is_file()
    assert json.loads((workspace / "motion_preparation.json").read_text()) == report
    request = json.loads(planner.calls[0]["messages"][1]["content"])
    assert request["shot"] == board["shots"][0]
    assert request["last_frame"] == 95 and request["bones"] and not request["feedback"]
    assert not (workspace / "runs").exists()


def test_resume_revalidates_bytes_without_model_or_render(prepared_copy, monkeypatch):
    workspace, board, report = prepared_copy
    monkeypatch.setattr(preparation, "build_motion_previs", lambda *a: pytest.fail("unexpected rerender"))
    planner = NumericPlanner()
    resumed = preparation.prepare_motion_previsualization(board, workspace, client=planner)
    assert resumed == report
    assert planner.calls == []


@pytest.mark.parametrize("change", ["source", "description", "text_model", "video_model"])
def test_changed_inputs_fail_closed_and_preserve_record(prepared_copy, monkeypatch, change):
    workspace, board, _ = prepared_copy
    before = (workspace / "motion_preparation.json").read_bytes()
    if change == "source":
        board["shots"][0]["camera"]["start_framing"] = "medium shot"
    elif change == "description":
        board["shots"][0]["scene_description"] += " Different floor."
    elif change == "text_model":
        monkeypatch.setattr(config, "LLM_MODEL", "different-planner")
    else:
        monkeypatch.setattr(config, "SEEDANCE_MODEL", "doubao-seedance-2-0-fast-260128")
    planner = NumericPlanner()
    report = preparation.prepare_motion_previsualization(board, workspace, client=planner)
    assert report["status"] == "failed" and not planner.calls
    assert (workspace / "motion_preparation.json").read_bytes() == before


@pytest.mark.parametrize("damage", ["reference", "plan", "missing", "manifest", "symlink"])
def test_corruption_is_not_regenerated(prepared_copy, tmp_path, damage):
    workspace, board, previous = prepared_copy
    package = workspace / previous["shots"][0]["package"]
    if damage in {"reference", "plan", "manifest"}:
        filename = {"reference": "reference.mp4", "plan": "motion_plan.json", "manifest": "manifest.json"}[damage]
        (package / filename).write_bytes(b"corrupt")
    elif damage == "missing":
        (package / "reference.mp4").unlink()
    else:
        path = package / "reference.mp4"
        outside = tmp_path / "outside.mp4"
        path.rename(outside)
        path.symlink_to(outside)
    planner = NumericPlanner()
    report = preparation.prepare_motion_previsualization(board, workspace, client=planner)
    assert report["status"] == "failed" and not planner.calls
    assert json.loads((workspace / "motion_preparation.json").read_text())["status"] == "failed"


def test_one_geometry_correction_then_actual_export(inputs, tmp_path):
    board, value = inputs
    invalid = copy.deepcopy(value)
    for pose in invalid["rigs"][1]["keyframes"]:
        pose["root"][0] += 0.1
    planner = NumericPlanner(invalid, value)
    report = preparation.prepare_motion_previsualization(board, tmp_path, client=planner)
    assert report["status"] == "prepared_unbound", report
    assert len(planner.calls) == 2
    assert json.loads(planner.calls[1]["messages"][1]["content"])["feedback"]
    assert (tmp_path / report["shots"][0]["reference_path"]).is_file()


def test_failed_geometry_is_persisted_and_never_retried_on_resume(inputs, tmp_path):
    board, value = inputs
    for pose in value["rigs"][1]["keyframes"]:
        pose["root"][0] += 0.1
    planner = NumericPlanner(value, value)
    report = preparation.prepare_motion_previsualization(board, tmp_path, client=planner)
    assert report["status"] == "failed" and report["shots"][0]["diagnostics"]
    assert len(planner.calls) == 2
    assert preparation.prepare_motion_previsualization(board, tmp_path, client=planner) == report
    assert len(planner.calls) == 2 and not list(tmp_path.rglob("*.mp4"))


@pytest.mark.parametrize("bad", ["malformed", "duplicate", "overlong", "script", "too_many_frames"])
def test_untrusted_model_output_cannot_select_paths_or_unbounded_resources(inputs, tmp_path, bad):
    board, value = inputs
    if bad == "malformed":
        value = "not json"
    elif bad == "duplicate":
        value = '{"rigs": [], "contact": {}, "rigs": []}'
    elif bad == "overlong":
        value = " " * 100001
    elif bad == "script":
        value["script"] = "touch /tmp/should-not-run"
    else:
        value["rigs"][0]["keyframes"] *= 65
    planner = NumericPlanner(value, value)
    report = preparation.prepare_motion_previsualization(board, tmp_path, client=planner)
    assert report["status"] == "failed"
    assert not list(tmp_path.rglob("*.mp4"))
    assert len(planner.calls) == (2 if bad == "too_many_frames" else 1)


@pytest.mark.parametrize("unsupported", ["camera", "continuity", "multiple_shots", "prop", "events", "multiple_beats", "front"])
def test_unsupported_never_calls_planner(inputs, tmp_path, unsupported):
    board, _ = inputs
    shot = board["shots"][0]
    if unsupported == "camera":
        shot["camera"]["primary_movement"] = "pan"
    elif unsupported == "continuity":
        shot["continuity_from_previous"] = "seamless"
    elif unsupported == "multiple_shots":
        other = copy.deepcopy(shot)
        other["shot_id"] = 2
        board["shots"].append(other)
    elif unsupported == "prop":
        shot["key_props"] = ["walking stick"]
    elif unsupported == "events":
        shot["action_beats"][0].update(interaction_geometry=shot["interaction_geometry"],
                                     effect_at_seconds=2, reaction_at_seconds=2.5)
    elif unsupported == "multiple_beats":
        shot["action_beats"].append(copy.deepcopy(shot["action_beats"][0]))
    else:
        shot["blocking"]["attacker"]["body_orientation"] = "facing camera"
    planner = NumericPlanner()
    report = preparation.prepare_motion_previsualization(board, tmp_path, client=planner)
    assert report["status"] == "unsupported", report
    assert not planner.calls and not list(tmp_path.rglob("*.mp4"))


def _bind(board, entry):
    board["shots"][0]["motion_reference"] = {
        "path": entry["reference_path"], "sha256": entry["reference_sha256"],
        "source_sha256": entry["source_sha256"], "model": MODEL, "asset_uri": "asset://offline_test",
    }


def test_binding_prepared_video_transitions_to_ready_without_calls(prepared_copy):
    workspace, board, previous = prepared_copy
    _bind(board, previous["shots"][0])
    planner = NumericPlanner()
    report = preparation.prepare_motion_previsualization(board, workspace, client=planner)
    assert report["status"] == "ready" and not planner.calls
    assert report["shots"][0]["reference_sha256"] == previous["shots"][0]["reference_sha256"]
    assert preparation.prepare_motion_previsualization(board, workspace, client=planner) == report


def test_binding_does_not_bypass_package_integrity(prepared_copy):
    workspace, board, previous = prepared_copy
    _bind(board, previous["shots"][0])
    (workspace / previous["shots"][0]["package"] / "motion_plan.json").write_text("{}")
    planner = NumericPlanner()
    assert preparation.prepare_motion_previsualization(board, workspace, client=planner)["status"] == "failed"
    assert not planner.calls


def test_existing_valid_binding_needs_no_new_plan(prepared_copy):
    workspace, board, previous = prepared_copy
    _bind(board, previous["shots"][0])
    (workspace / "motion_preparation.json").unlink()
    planner = NumericPlanner()
    report = preparation.prepare_motion_previsualization(board, workspace, client=planner)
    assert report["status"] == "ready" and not planner.calls


def test_symlink_lock_is_refused_without_touching_target(inputs, tmp_path):
    board, _ = inputs
    target = tmp_path / "protected"
    target.write_text("preserve")
    (tmp_path / ".motion_preparation.lock").symlink_to(target)
    planner = NumericPlanner()
    report = preparation.prepare_motion_previsualization(board, tmp_path, client=planner)
    assert report["status"] == "failed" and not planner.calls
    assert target.read_text() == "preserve"


def test_concurrent_prepare_has_no_side_effect(inputs, tmp_path):
    board, _ = inputs
    planner = NumericPlanner()
    with (tmp_path / ".motion_preparation.lock").open("w") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = preparation.prepare_motion_previsualization(board, tmp_path, client=planner)
    assert report["status"] == "failed" and not planner.calls
    assert not (tmp_path / "motion_preparation.json").exists()


@pytest.mark.parametrize("bad", [{}, {"shots": []}, {"shots": [None]}, {"shots": [{"shot_id": "../outside"}]}])
def test_invalid_input_persists_failure_without_calls(tmp_path, bad):
    planner = NumericPlanner()
    report = preparation.prepare_motion_previsualization(bad, tmp_path, client=planner)
    assert report["status"] == "failed" and not planner.calls
    assert json.loads((tmp_path / "motion_preparation.json").read_text()) == report


def test_corrupt_record_is_preserved_without_calls(prepared_copy):
    workspace, board, _ = prepared_copy
    record = workspace / "motion_preparation.json"
    record.write_text("{}")
    before = record.read_bytes()
    planner = NumericPlanner()
    assert preparation.prepare_motion_previsualization(board, workspace, client=planner)["status"] == "failed"
    assert not planner.calls and record.read_bytes() == before


def test_package_without_planning_record_is_not_adopted(prepared_copy):
    workspace, board, _ = prepared_copy
    (workspace / "motion_preparation.json").unlink()
    planner = NumericPlanner()
    report = preparation.prepare_motion_previsualization(board, workspace, client=planner)
    assert report["status"] == "failed" and not planner.calls


def test_noncombat_contact_uses_same_numeric_pipeline(inputs, tmp_path):
    board, value = inputs
    board = json.loads(json.dumps(board).replace("attacker", "instructor").replace("defender", "student"))
    value = json.loads(json.dumps(value).replace("attacker", "instructor").replace("defender", "student"))
    shot = board["shots"][0]
    shot["scene_description"] = "康复课程：指导者伸手轻触学员手掌，学员接触后向后舒展。"
    shot["prompt_en"] = "An instructor makes gentle palm contact with a student in a rehabilitation exercise."
    shot["primary_action"] = "Instructor reaches the student's palm; the student stretches after contact."
    shot["action_beats"][0]["action"] = "gently reaches the student's palm"
    shot["action_beats"][0]["visible_result"] = "student stretches after their palms meet"
    planner = NumericPlanner(value)
    report = preparation.prepare_motion_previsualization(board, tmp_path, client=planner)
    assert report["status"] == "prepared_unbound", report
    generated = json.loads((tmp_path / report["shots"][0]["package"] / "motion_plan.json").read_text())
    assert {rig["entity"] for rig in generated["rigs"]} == {"instructor", "student"}
    assert (tmp_path / report["shots"][0]["reference_path"]).is_file()


@pytest.mark.parametrize("end", [3, 2.1])
def test_preparation_uses_common_temporal_diagnostics(inputs, tmp_path, end):
    board, raw = inputs
    shot = board["shots"][0]
    base = shot["action_beats"][0]
    shot["action_beats"] = [
        {**base, "phase": "trigger", "activity": "transition", "start_seconds": 0, "end_seconds": 1},
        {**base, "activity": "interaction", "start_seconds": 1, "end_seconds": end},
        {**base, "phase": "aftermath", "activity": "hold", "start_seconds": end, "end_seconds": 4},
    ]
    planner = NumericPlanner(raw, raw)
    report = preparation.prepare_motion_previsualization(board, tmp_path, client=planner)
    assert report["status"] == ("prepared_unbound" if end == 3 else "failed"), report
    if end != 3:
        assert any("交互时窗" in issue for issue in report["shots"][0]["diagnostics"])
