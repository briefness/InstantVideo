"""Real local media, fake provider, and fail-closed motion lineage checks."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import config
from pipeline.causality import ACTION_CONTRACT_VERSION
from pipeline.generator import VideoGenerator
from pipeline.models import RunStatus, ShotSpec
from pipeline.motion_plan import MotionPlan, compile_motion_previs, motion_source_binding
from pipeline.motion_reference import (
    ensure_motion_lineage, motion_reference_context, prepare_motion_reference,
    probe_motion_reference, validate_video_reference, validate_workspace_motion,
)
from pipeline.orchestrator import VideoPipeline
from pipeline.run_state import PaidTakeBudgetExhaustedError, RunWorkspace
from tools.motion_previs import build_motion_previs
from tools.motion_renderer import draw_motion_frame
from tools.prepare_motion_run import prepare_motion_run
from tools.seedance_api import SeedanceAPI


MODEL = "doubao-seedance-2-0-mini-260615"
EXAMPLE = Path(__file__).resolve().parents[1] / "examples/contact_previs"


def test_historic_reversed_color_mapping_is_not_submitted(package, tmp_path):
    import shutil
    storyboard, _, path = package
    old = tmp_path / "old-package"
    shutil.copytree(path, old)
    manifest_path = old / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["rig_labels"] = {"A": "defender", "B": "attacker"}
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="角色颜色"):
        prepare_motion_run(storyboard, old, tmp_path / "runs", model=MODEL, asset_uri="asset://test")
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("model", ["doubao-seedance-2.0", "doubao-seedance-2.0-fast", "doubao-seedance-2.0-mini"])
def test_agent_plan_reference_preserves_model_identity(package, tmp_path, model):
    storyboard, _, path = package
    workspace = prepare_motion_run(storyboard, path, tmp_path, model=model, asset_uri="asset://test")
    shot = RunWorkspace.resume(workspace.path).load_storyboard()["shots"][0]
    validate_video_reference("asset://test", model, None, 0)
    assert prepare_motion_reference(shot, "16:9", workspace.path, model)["model"] == model
    with pytest.raises(ValueError, match="模型"):
        prepare_motion_reference(shot, "16:9", workspace.path, MODEL)


@pytest.mark.asyncio
async def test_agent_plan_model_reaches_sdk_unchanged():
    create = Mock(return_value=SimpleNamespace(id="plan-task"))
    api = object.__new__(SeedanceAPI)
    api.ark_client = SimpleNamespace(content_generation=SimpleNamespace(tasks=SimpleNamespace(create=create)))
    api._poll_ark = AsyncMock(return_value={"status": "pending", "provider_task_id": "plan-task"})
    await api.generate("motion", duration=4, model="doubao-seedance-2.0-fast", reference_video="asset://test")
    assert create.call_args.kwargs["model"] == "doubao-seedance-2.0-fast"
    assert create.call_args.kwargs["content"][-1]["role"] == "reference_video"


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    storyboard = json.loads((EXAMPLE / "storyboard.json").read_text())
    plan = MotionPlan.model_validate_json((EXAMPLE / "motion.json").read_text())
    path = build_motion_previs(storyboard, plan, tmp_path_factory.mktemp("motion") / "package")
    return storyboard, plan, path


@pytest.fixture
def workspace(package, tmp_path, monkeypatch, use_test_model):
    use_test_model(MODEL)
    monkeypatch.setattr(config, "SEMANTIC_REVIEW_ENABLED", False)
    # Any unexpected provider access fails this offline test.
    monkeypatch.setattr(SeedanceAPI, "_init_clients", lambda self: None)
    storyboard, _, path = package
    return prepare_motion_run(storyboard, path, tmp_path, model=MODEL, asset_uri="asset://test-motion")


def test_clean_frame_contains_only_background_and_rig_colors(package):
    storyboard, plan, path = package
    compiled = compile_motion_previs(storyboard["shots"][0], "16:9", plan)
    clean = draw_motion_frame(plan, compiled["frames"][48], annotated=False)
    assert set(clean.getdata()) == {(16, 24, 32), (84, 217, 193), (255, 179, 99)}
    assert clean.tobytes() != draw_motion_frame(plan, compiled["frames"][48]).tobytes()
    payload = (path / "reference.mp4").read_bytes()
    stream = probe_motion_reference(payload, 4, "16:9")
    assert stream["width"] * stream["height"] >= 407696
    assert int(stream["nb_frames"]) == 96
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["files"]["reference.mp4"] == hashlib.sha256(payload).hexdigest()


def test_workspace_is_zero_budget_planned_only_and_round_trips(workspace):
    restored = RunWorkspace.resume(workspace.path)
    shot = restored.load_storyboard()["shots"][0]
    context = prepare_motion_reference(shot, "16:9", restored.path, MODEL)
    assert context["input_mode"] == "reference_video"
    assert context["sha256"] == shot["motion_reference"]["sha256"]
    assert context["source_sha256"] == motion_source_binding(ShotSpec.model_validate(shot).model_dump(), "16:9")
    assert restored.options.paid_take_budget == 0
    assert restored.manifest.shots["1"].canonical_take_id is None
    assert not any(shot.get("observed_end_state", {}).values())
    with pytest.raises(PaidTakeBudgetExhaustedError):
        restored.reserve_paid_take(1)


@pytest.mark.parametrize("change", ["bytes", "missing", "escape", "absolute", "symlink", "source", "model", "continuity"])
def test_reference_invalidations_stop_before_submission(workspace, change, tmp_path):
    shot = workspace.load_storyboard()["shots"][0]
    spec = shot["motion_reference"]
    path = workspace.path / spec["path"]
    model = MODEL
    if change == "bytes":
        path.write_bytes(b"different")
    elif change == "missing":
        path.unlink()
    elif change == "escape":
        spec["path"] = "../outside.mp4"
    elif change == "absolute":
        spec["path"] = str(path)
    elif change == "symlink":
        target = tmp_path / "copied.mp4"
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
    elif change == "source":
        shot["duration"] = 5
    elif change == "model":
        model = "doubao-seedance-2-0-fast-260128"
    else:
        shot["continuity_from_previous"] = "seamless"
    with pytest.raises((ValueError, OSError)):
        prepare_motion_reference(shot, "16:9", workspace.path, model)


@pytest.mark.parametrize("kind", ["event_contract", "multiple_peaks", "multiple_interactions", "key_props", "continuity_props"])
def test_manual_binding_cannot_bypass_v1_capability_gate(workspace, kind, monkeypatch):
    shot = workspace.load_storyboard()["shots"][0]
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
    shot["motion_reference"]["source_sha256"] = motion_source_binding(shot, "16:9")
    monkeypatch.setattr("pipeline.motion_reference.probe_motion_reference",
                        lambda *args: pytest.fail("Unsupported contracts must stop before media validation"))
    with pytest.raises(ValueError, match="v1"):
        prepare_motion_reference(shot, "16:9", workspace.path, MODEL)


def test_real_reference_package_rejects_rebound_out_of_window_motion(package, tmp_path):
    import shutil
    from tools.publish_motion_reference import prepare_upload

    board, _, original = package
    board = copy.deepcopy(board)
    path = tmp_path / "out-of-window"
    shutil.copytree(original, path)
    shot = board["shots"][0]
    base = shot["action_beats"][0]
    shot["action_beats"] = [
        {**base, "phase": "trigger", "activity": "transition", "start_seconds": 0, "end_seconds": 1},
        {**base, "activity": "interaction", "start_seconds": 1, "end_seconds": 2.1},
        {**base, "phase": "aftermath", "activity": "hold", "start_seconds": 2.1, "end_seconds": 4},
    ]
    source_sha = motion_source_binding(shot, "16:9")
    raw = json.loads((path / "motion_plan.json").read_text())
    raw["source_sha256"] = source_sha
    (path / "motion_plan.json").write_text(json.dumps(raw))
    (path / "source_shot.json").write_text(json.dumps({"ratio": "16:9", "shots": [shot]}))
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["source_sha256"] = source_sha
    for name in ("motion_plan.json", "source_shot.json"):
        manifest["files"][name] = hashlib.sha256((path / name).read_bytes()).hexdigest()
    (path / "manifest.json").write_text(json.dumps(manifest))
    # 字节和来源指纹全部一致，也不能让接触后的反应超出交互时窗。
    with pytest.raises(ValueError, match="未通过"):
        prepare_upload(path)
    with pytest.raises(ValueError):
        prepare_motion_run(board, path, tmp_path / "runs", model=MODEL, asset_uri="asset://test")
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("asset,model,role,count", [
    ("data:video/mp4;base64,AAAA", MODEL, None, 0),
    ("file:///tmp/ref.mp4", MODEL, None, 0),
    ("http://example.com/mutable.mp4", MODEL, None, 0),
    ("asset://../bad", MODEL, None, 0),
    ("asset://ok", "unknown", None, 0),
    ("asset://ok", MODEL, "first_frame", 1),
    ("asset://ok", MODEL, "last_frame", 1),
    ("asset://ok", MODEL, None, 1),
    ("asset://ok", MODEL, "reference_image", 10),
])
def test_provider_boundary_rejects_unsupported_inputs(asset, model, role, count):
    with pytest.raises(ValueError):
        validate_video_reference(asset, model, role, count)


@pytest.mark.asyncio
async def test_api_sends_reference_video_and_preserves_checkpoint_before_poll():
    events = []
    create = Mock(return_value=SimpleNamespace(id="task-1"))
    api = object.__new__(SeedanceAPI)
    api.ark_client = SimpleNamespace(content_generation=SimpleNamespace(tasks=SimpleNamespace(create=create)))
    async def poll(task_id, timeout):
        assert events == ["task-1"]
        return {"status": "pending", "provider_task_id": task_id}
    api._poll_ark = poll
    await api.generate("motion", model=MODEL, image_urls=["https://example.com/person.png"],
                       image_role="reference_image", reference_video="asset://motion-1",
                       on_submitted=events.append)
    content = create.call_args.kwargs["content"]
    assert content[-1] == {"type": "video_url", "video_url": {"url": "asset://motion-1"}, "role": "reference_video"}
    assert content[1]["role"] == "reference_image"
    await api.generate("ignored", model=MODEL, task_id="task-1", reference_video="invalid")
    assert create.call_count == 1


@pytest.mark.asyncio
async def test_api_conflicting_first_frame_never_calls_provider():
    api = object.__new__(SeedanceAPI)
    api.ark_client = Mock()
    with pytest.raises(ValueError, match="混用"):
        await api.generate("test", model=MODEL, image_urls=["x.png"],
                           image_role="first_frame", reference_video="asset://x")
    api.ark_client.content_generation.tasks.create.assert_not_called()


@pytest.mark.parametrize("change", ["removed", "hash", "asset", "model"])
def test_lineage_change_cannot_be_reaccepted_or_resumed(workspace, change):
    generator = VideoGenerator(str(workspace.path))
    shot = workspace.load_storyboard()["shots"][0]
    stored = generator._generation_provenance(shot, [], None)
    assert stored["motion_reference"] == generator._acceptance_context(shot, [], None)["motion_reference"]
    generator._write_generation_provenance(1, stored, "task-1")
    if change == "removed":
        shot["motion_reference"] = None
    elif change == "hash":
        shot["motion_reference"]["sha256"] = "0" * 64
    elif change == "asset":
        shot["motion_reference"]["asset_uri"] = "asset://replacement"
    else:
        shot["motion_reference"]["model"] = "doubao-seedance-2-0-fast-260128"
    with pytest.raises(ValueError, match="禁止复用"):
        ensure_motion_lineage(shot, workspace.path / "shots/shot_001.mp4")
    assert not generator._provenance_matches(str(workspace.path / "shots/shot_001.mp4"), shot, [], None)


@pytest.mark.asyncio
async def test_generator_submits_motion_and_records_identity_with_fake_api(workspace):
    generator = VideoGenerator(str(workspace.path))
    generator.reserve_paid_take = Mock(return_value="fake-reservation")
    async def generate(**kwargs):
        assert kwargs["reference_video"] == "asset://test-motion"
        assert "teal skeleton is attacker" in kwargs["prompt"]
        kwargs["on_submitted"]("fake-task")
        return {"status": "pending", "provider_task_id": "fake-task"}
    generator.api.generate = AsyncMock(side_effect=generate)
    storyboard = workspace.load_storyboard()
    result = await generator._generate_single_shot(storyboard["shots"][0], None, None, storyboard)
    assert result.provider_task_id == "fake-task"
    assert generator.api.generate.await_count == 1
    assert generator.reserve_paid_take.call_count == 1
    lineage = generator._read_generation_provenance(str(workspace.path / "shots/shot_001.mp4"))
    assert lineage["motion_reference"]["model"] == MODEL
    assert lineage["motion_reference"]["input_mode"] == "reference_video"


@pytest.mark.asyncio
async def test_generator_zero_budget_never_submits(workspace):
    generator = VideoGenerator(str(workspace.path), reserve_paid_take=workspace.reserve_paid_take)
    generator.api.generate = AsyncMock(side_effect=AssertionError("must not submit"))
    storyboard = workspace.load_storyboard()
    result = await generator._generate_single_shot(storyboard["shots"][0], None, None, storyboard)
    assert result.status == "failed"
    generator.api.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_pending_motion_polls_exact_task_without_budget_or_submit(workspace):
    generator = VideoGenerator(str(workspace.path))
    storyboard = workspace.load_storyboard()
    shot = storyboard["shots"][0]
    descriptor = {
        "task_id": "pending-1", "prompt_profile": "normal", "prompt_fingerprint": "a" * 64,
        "compiled_contract_version": ACTION_CONTRACT_VERSION,
        "compiled_contract_fingerprint": generator._compiled_contract_fingerprint(shot),
    }
    generator._write_generation_provenance(1, generator._generation_provenance(
        shot, [], None, prompt_profile="normal", prompt_fingerprint="a" * 64,
    ), "pending-1")
    generator.resume_tasks = {1: descriptor}
    generator.api.poll_task = AsyncMock(return_value={"status": "pending", "provider_task_id": "pending-1"})
    generator.api.generate = AsyncMock(side_effect=AssertionError("must not resubmit"))
    generator.reserve_paid_take = Mock(side_effect=AssertionError("must not reserve"))
    result = await generator._generate_single_shot(shot, None, None, storyboard)
    assert result.provider_task_id == "pending-1"
    generator.api.poll_task.assert_awaited_once()
    assert generator.api.poll_task.call_args.args[0] == "pending-1"
    generator.api.generate.assert_not_awaited()
    generator.reserve_paid_take.assert_not_called()


@pytest.mark.asyncio
async def test_completed_output_cannot_bypass_changed_motion(workspace):
    generator = VideoGenerator(str(workspace.path))
    storyboard = workspace.load_storyboard()
    shot = storyboard["shots"][0]
    generator._write_generation_provenance(1, generator._generation_provenance(shot, [], None), "done")
    final = workspace.path / "final.mp4"
    final.write_bytes(b"existing final")
    workspace.manifest.final_path = "final.mp4"
    workspace.manifest.status = RunStatus.succeeded
    workspace._save_manifest()
    (workspace.path / shot["motion_reference"]["path"]).write_bytes(b"changed")
    pipeline = VideoPipeline.from_workspace(workspace.path)
    pipeline._run = AsyncMock(side_effect=AssertionError("must not continue"))
    with pytest.raises(ValueError, match="指纹变化"):
        await pipeline.run()
    pipeline._run.assert_not_awaited()


def test_multiple_shots_rejected_before_work(workspace):
    storyboard = workspace.load_storyboard()
    second = copy.deepcopy(storyboard["shots"][0])
    second["shot_id"] = 2
    storyboard["shots"].append(second)
    with pytest.raises(ValueError, match="单镜头"):
        validate_workspace_motion(storyboard, workspace.path, MODEL)


def test_unbound_package_stays_offline(package, tmp_path):
    storyboard, _, path = package
    workspace = prepare_motion_run(storyboard, path, tmp_path, model=MODEL)
    shot = workspace.load_storyboard()["shots"][0]
    assert motion_reference_context(shot)["asset_uri"] is None
    assert workspace.options.paid_take_budget == 0


@pytest.mark.asyncio
async def test_completed_custom_canonical_path_cannot_hide_removed_motion(workspace):
    generator = VideoGenerator(str(workspace.path))
    storyboard = workspace.load_storyboard()
    shot = storyboard["shots"][0]
    accepted = workspace.path / "shots/accepted_custom.mp4"
    accepted.write_bytes(b"canonical")
    accepted.with_name("accepted_custom_generation.json").write_text(json.dumps(
        generator._generation_provenance(shot, [], None)
    ))
    workspace.record_shot(shot_id=1, status="success", local_path=str(accepted))
    shot["motion_reference"] = None
    workspace.save_storyboard(storyboard)
    final = workspace.path / "final.mp4"
    final.write_bytes(b"existing final")
    workspace.manifest.final_path = "final.mp4"
    workspace.manifest.status = RunStatus.succeeded
    workspace._save_manifest()
    with pytest.raises(ValueError, match="禁止复用"):
        await VideoPipeline.from_workspace(workspace.path).run()


@pytest.mark.parametrize("change", ["missing_clean", "tampered_clean", "malformed_manifest"])
def test_bad_packages_never_create_workspace(package, tmp_path, change):
    import shutil
    storyboard, _, original = package
    copy_path = tmp_path / "package"
    shutil.copytree(original, copy_path)
    if change == "missing_clean":
        (copy_path / "reference.mp4").unlink()
    elif change == "tampered_clean":
        (copy_path / "reference.mp4").write_bytes(b"untrusted replacement")
    else:
        manifest = json.loads((copy_path / "manifest.json").read_text())
        manifest["files"] = []
        (copy_path / "manifest.json").write_text(json.dumps(manifest))
    output_root = tmp_path / "runs"
    with pytest.raises((OSError, ValueError)):
        prepare_motion_run(storyboard, copy_path, output_root, model=MODEL)
    assert not output_root.exists()


def test_twelve_fps_previs_reference_keeps_duration_when_upsampled(package, tmp_path):
    from tools.motion_renderer import render_motion_media
    storyboard, plan, _ = package
    compiled = compile_motion_previs(storyboard["shots"][0], "16:9", plan)
    # Renderer isolation: supply the same sampled trajectory at half the frame rate.
    plan = plan.model_copy(update={"fps": 12, "contact": plan.contact.model_copy(
        update={"frame": 24, "reaction_frame": 28}
    )})
    compiled["frames"] = compiled["frames"][::2]
    compiled["frame_count"] = 48
    for index, sample in enumerate(compiled["frames"]):
        sample["frame"] = index
    result = render_motion_media(plan, compiled, tmp_path)
    assert result["avg_frame_rate"] == "12/1"
    assert result["reference_video"]["avg_frame_rate"] == "24/1"
    assert int(result["reference_video"]["nb_frames"]) == 96
    assert float(result["reference_video"]["duration"]) == 4


@pytest.mark.asyncio
async def test_missing_asset_stops_immediately_before_budget_or_api(workspace):
    generator = VideoGenerator(str(workspace.path))
    generator.reserve_paid_take = Mock(side_effect=AssertionError("must not reserve"))
    generator.api.generate = AsyncMock(side_effect=AssertionError("must not submit"))
    storyboard = workspace.load_storyboard()
    storyboard["shots"][0]["motion_reference"]["asset_uri"] = None
    with pytest.raises(ValueError, match="素材 ID"):
        await generator._generate_single_shot(storyboard["shots"][0], None, None, storyboard)
    generator.reserve_paid_take.assert_not_called()
    generator.api.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_policy_safe_retry_preserves_motion_input(workspace):
    generator = VideoGenerator(str(workspace.path))
    generator.reserve_paid_take = Mock(return_value="fake-reservation")
    calls = []
    async def generate(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return {"status": "failed", "error_type": "moderation", "error_locus": "input_text"}
        kwargs["on_submitted"]("fake-safe-task")
        return {"status": "pending", "provider_task_id": "fake-safe-task"}
    generator.api.generate = AsyncMock(side_effect=generate)
    storyboard = workspace.load_storyboard()
    result = await generator._generate_single_shot(storyboard["shots"][0], None, None, storyboard)
    assert result.prompt_profile == "policy_safe"
    assert len(calls) == 2
    assert all(call["reference_video"] == "asset://test-motion" for call in calls)
    assert all("teal skeleton is attacker" in call["prompt"] for call in calls)
