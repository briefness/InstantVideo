"""Offline end-to-end trial preparation, real ledger recovery and human review."""

import copy
import fcntl
import json
import socket
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import config
from pipeline.generator import VideoGenerator
from pipeline.motion_experiment import experiment_preflight, experiment_report, load_experiment, prepare_experiment
from pipeline.motion_plan import MotionPlan
from pipeline.motion_review import export_review_bundle, summarize_reviews
from tools.motion_experiment import execute_experiment, main
from tools.motion_previs import build_motion_previs
from tools.seedance_api import SeedanceAPI

MODEL = "doubao-seedance-2-0-mini-260615"
EXAMPLE = Path(__file__).resolve().parents[1] / "examples/contact_previs"


@pytest.mark.asyncio
async def test_real_generator_carries_staging_to_both_arms_without_retake(experiment, monkeypatch):
    submissions = []
    async def generate(self, **kwargs):
        task_id = f"offline-staging-{len(submissions)}"
        submissions.append(kwargs)
        kwargs["on_submitted"](task_id)
        return {"status": "failed", "provider_task_id": task_id, "error": "offline fixture"}
    monkeypatch.setattr(SeedanceAPI, "generate", generate)
    monkeypatch.setattr("pipeline.generator.asyncio.sleep", AsyncMock())
    assert (await execute_experiment(experiment))["status"] == "finished"
    assert len(submissions) == 2
    for request in submissions:
        assert "attacker=profile toward screen-right" in request["prompt"]
        assert "defender=profile toward screen-left" in request["prompt"]
        assert "camera remains fixed" in request["prompt"]
    await execute_experiment(experiment)
    assert len(submissions) == 2


def test_exported_review_carries_same_staging_as_generation(experiment, review_bundle):
    from pipeline.visual_invariants import compile_visual_invariants
    _, rows = load_experiment(experiment)
    shot = rows[0][1].load_storyboard()["shots"][0]
    document = json.loads((review_bundle / "reviewer/ratings.json").read_text())
    assert document["version"] == "motion-review-v3"
    assert document["contract"]["visual_invariants"] == compile_visual_invariants(shot).review_projection
    assert all(row["visual_invariants_ok"] is None for row in document["ratings"])


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    storyboard = json.loads((EXAMPLE / "storyboard.json").read_text())
    plan = MotionPlan.model_validate_json((EXAMPLE / "motion.json").read_text())
    path = build_motion_previs(storyboard, plan, tmp_path_factory.mktemp("trials") / "package")
    return storyboard, path


@pytest.fixture(autouse=True)
def offline(monkeypatch, use_test_model):
    def forbid(*args, **kwargs):
        raise AssertionError("network forbidden")
    monkeypatch.setattr(socket, "getaddrinfo", forbid)
    monkeypatch.setattr(socket, "create_connection", forbid)
    monkeypatch.setattr(SeedanceAPI, "_init_clients", lambda self: None)
    use_test_model(MODEL)
    monkeypatch.setattr(config, "SEMANTIC_REVIEW_ENABLED", False)
    monkeypatch.setattr(config, "ARK_API_KEY", "test-only")
    monkeypatch.setattr(config, "ARK_API_KEY_SEEDANCE", "test-only")


@pytest.fixture
def experiment(package, tmp_path):
    storyboard, path = package
    return prepare_experiment(storyboard, path, tmp_path / "experiment", model=MODEL,
                              repetitions=1, paid_take_budget=2, asset_uri="asset://offline-test")


def test_zero_budget_pairs_preserve_controls(package, tmp_path):
    source, path = package
    root = prepare_experiment(source, path, tmp_path / "zero", model=MODEL)
    manifest, loaded = load_experiment(root)
    assert len(loaded) == 6 and manifest["paid_take_budget"] == 0
    assert [s["arm"] for s, _ in loaded] == ["baseline", "motion_reference", "motion_reference", "baseline", "baseline", "motion_reference"]
    assert all(w.options.paid_take_budget == 0 for _, w in loaded)
    check = experiment_preflight(root)
    assert check["status"] == "blocked" and check["network_checked"] is False
    assert experiment_report(root)["generation_quality"] == "not_evaluated"
    assert all(row["status"] == "prepared" for row in experiment_report(root)["samples"])
    assert main(["run", "--experiment", str(root)]) == 2
    with pytest.raises(FileExistsError):
        prepare_experiment(source, path, root, model=MODEL)


@pytest.mark.parametrize("repetitions,budget", [(0, 0), (4, 0), (True, 0), (1, 1), (2, 3), (1, -1)])
def test_invalid_budget_before_directory_creation(package, tmp_path, repetitions, budget):
    source, path = package
    with pytest.raises(ValueError):
        prepare_experiment(source, path, tmp_path / "invalid", model=MODEL,
                           repetitions=repetitions, paid_take_budget=budget)
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("change", ["storyboard", "options", "escape", "duplicate", "id", "arm", "pair", "model", "workflow"])
def test_frozen_inputs_reject_tampering(experiment, change):
    manifest, loaded = load_experiment(experiment)
    sample, workspace = loaded[0]
    if change == "storyboard":
        document = workspace.load_storyboard()
        document["shots"][0]["prompt_en"] += " changed"
        workspace.save_storyboard(document)
    elif change == "options":
        manifest["samples"][0]["options"]["paid_take_budget"] = 9
    elif change == "escape":
        manifest["samples"][0]["workspace"] = "../outside"
    elif change == "duplicate":
        manifest["samples"][1]["workspace"] = sample["workspace"]
    elif change == "id":
        manifest["samples"][0]["sample_id"] = "../x"
    elif change == "arm":
        manifest["samples"][0]["arm"] = "motion_reference"
    elif change == "pair":
        manifest["samples"][0]["pair"] = True
    elif change == "model":
        manifest["model"] = "unknown-model"
    else:
        manifest["remote_semantic_review"] = True
    (experiment / "experiment_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_experiment(experiment)


@pytest.mark.asyncio
async def test_zero_budget_and_unknown_submission_never_initialize_provider(package, tmp_path, experiment, monkeypatch):
    def forbid(*args, **kwargs):
        raise AssertionError("must not initialize provider")
    monkeypatch.setattr(SeedanceAPI, "_init_clients", forbid)
    source, path = package
    zero = prepare_experiment(source, path, tmp_path / "zero", model=MODEL, repetitions=1)
    assert (await execute_experiment(zero))["status"] == "blocked"
    _, loaded = load_experiment(experiment)
    loaded[0][1].reserve_paid_take(1)
    assert (await execute_experiment(experiment))["status"] == "blocked"


@pytest.mark.asyncio
async def test_exclusive_execution_lock(experiment):
    with (experiment / ".execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="执行进程"):
            await execute_experiment(experiment)


@pytest.mark.asyncio
async def test_crash_after_provider_confirmation_stops_batch(experiment, monkeypatch):
    _, loaded = load_experiment(experiment)
    workspace = loaded[0][1]
    reservation = workspace.reserve_paid_take(1)
    workspace.confirm_paid_take_submission(reservation, "acknowledged-before-crash")
    def forbid(*args, **kwargs):
        raise AssertionError("untracked acknowledgement must stop before provider init")
    monkeypatch.setattr(SeedanceAPI, "_init_clients", forbid)
    assert (await execute_experiment(experiment))["status"] == "blocked"


@pytest.mark.asyncio
async def test_pending_resume_uses_same_task_and_no_second_submission(experiment, monkeypatch):
    submissions, polls = [], []
    async def generate(self, **kwargs):
        submissions.append(kwargs)
        kwargs["on_submitted"]("pending-original")
        return {"status": "pending", "provider_task_id": "pending-original"}
    async def poll(self, task_id, **kwargs):
        polls.append(task_id)
        return {"status": "pending", "provider_task_id": task_id}
    monkeypatch.setattr(SeedanceAPI, "generate", generate)
    monkeypatch.setattr(SeedanceAPI, "poll_task", poll)
    assert (await execute_experiment(experiment))["status"] == "pending"
    assert (await execute_experiment(experiment))["status"] == "pending"
    assert len(submissions) == 1 and polls == ["pending-original"]
    assert sum(row["submitted_tasks"] for row in experiment_report(experiment)["samples"]) == 1


@pytest.mark.asyncio
async def test_failed_paid_tasks_retained_without_retakes(experiment, monkeypatch):
    submissions = []
    async def generate(self, **kwargs):
        task_id = f"failed-{len(submissions)}"
        submissions.append(kwargs)
        kwargs["on_submitted"](task_id)
        return {"status": "failed", "provider_task_id": task_id, "error": "test failure"}
    monkeypatch.setattr(SeedanceAPI, "generate", generate)
    monkeypatch.setattr("pipeline.generator.asyncio.sleep", AsyncMock())
    assert (await execute_experiment(experiment))["status"] == "finished"
    assert len(submissions) == 2
    assert "reference_video" not in submissions[0]
    assert submissions[1]["reference_video"] == "asset://offline-test"
    assert (await execute_experiment(experiment))["status"] == "finished"
    assert len(submissions) == 2
    rows = experiment_report(experiment)["samples"]
    assert all(row["submitted_tasks"] == 1 and row["status"] == "failed" for row in rows)


@pytest.mark.asyncio
@pytest.mark.parametrize("technical_pass", [True, False])
async def test_downloaded_raw_takes_survive_execution_and_repeat(experiment, package, monkeypatch, technical_pass):
    submissions = []
    async def generate(self, **kwargs):
        task_id = f"local-fixture-{len(submissions)}"
        submissions.append(kwargs)
        kwargs["on_submitted"](task_id)
        return {"status": "succeeded", "provider_task_id": task_id, "video_url": "https://unused.example.com"}
    async def download(self, url, destination):
        Path(destination).write_bytes((package[1] / "reference.mp4").read_bytes())
    monkeypatch.setattr(SeedanceAPI, "generate", generate)
    monkeypatch.setattr(SeedanceAPI, "download_video", download)
    monkeypatch.setattr("pipeline.generator.check_video_quality", lambda path: {
        "quality_score": 90 if technical_pass else 0, "pass": technical_pass, "issues": [],
    })
    assert (await execute_experiment(experiment))["status"] == "finished"
    assert (await execute_experiment(experiment))["status"] == "finished"
    assert len(submissions) == 2
    _, loaded = load_experiment(experiment)
    for _, workspace in loaded:
        state = workspace.manifest.shots["1"]
        assert len(state.take_history) == 1
        assert state.semantic_accepted is None
        assert state.prompt_attempts[0].fingerprint
        assert not workspace.manifest.final_path
    review = experiment / "test-review"
    assert export_review_bundle(experiment, review)["clips"] == 2


@pytest.fixture
def review_bundle(experiment, package, tmp_path):
    # Local previs media is a test fixture, never a claimed generated result.
    _, loaded = load_experiment(experiment)
    for sample, workspace in loaded:
        generator = VideoGenerator(str(workspace.path))
        shot = workspace.load_storyboard()["shots"][0]
        clip = workspace.path / "shots/shot_001.mp4"
        clip.write_bytes((package[1] / "reference.mp4").read_bytes())
        generator._write_generation_provenance(1, generator._generation_provenance(shot, [], None), "fixture")
        workspace.record_shot(shot_id=1, status="failed", local_path=str(clip),
                              provider_task_id="fixture", semantic_accepted=False)
    directory = tmp_path / "review"
    result = export_review_bundle(experiment, directory)
    assert result["clips"] == 2
    return directory


def test_review_export_preserves_rejected_takes_and_hides_labels(experiment, review_bundle):
    ratings_path = review_bundle / "reviewer/ratings.json"
    ratings = json.loads(ratings_path.read_text())
    assert "contract" in ratings and "motion_reference" not in ratings_path.read_text()
    assert "sample_id" not in ratings_path.read_text()
    assert summarize_reviews(experiment, review_bundle)["decision"] == "not_evaluated"
    ratings["reviewer_id"] = "reviewer_a"
    for rating in ratings["ratings"]:
        rating.update(assessment="reviewed", contact_frame=48, reaction_frame=55,
                      identity_ok=True, scope_ok=True, endpoint_ok=True, causal_link_ok=True, visual_invariants_ok=True)
    ratings_path.write_text(json.dumps(ratings))
    result = summarize_reviews(experiment, review_bundle)
    assert result["decision"] == "pilot_observations_only"
    assert all(counts == {"planned": 1, "reviewed": 1, "passed": 1} for counts in result["arms"].values())


@pytest.mark.parametrize("change", ["frame", "flags", "unknown", "duplicate_rating", "clip", "arm", "sample", "traversal", "duplicate_take", "count"])
def test_review_rejects_bad_observations_and_mapping(experiment, review_bundle, change):
    ratings_path = review_bundle / "reviewer/ratings.json"
    map_path = review_bundle / "private_mapping.json"
    ratings = json.loads(ratings_path.read_text())
    mapping = json.loads(map_path.read_text())
    entry = mapping["mapping"][0]
    rating = ratings["ratings"][0]
    ratings["reviewer_id"] = "reviewer_a"
    rating.update(assessment="reviewed", causal_link_ok=True, identity_ok=True, scope_ok=True, visual_invariants_ok=True,
                  endpoint_ok=True, contact_frame=1, reaction_frame=2)
    if change == "frame":
        rating["reaction_frame"] = rating["frames"]
    elif change == "flags":
        rating["scope_ok"] = "true"
    elif change == "unknown":
        rating["review_id"] = "0" * 32
    elif change == "duplicate_rating":
        ratings["ratings"].append(copy.deepcopy(rating))
    elif change == "clip":
        (review_bundle / "reviewer" / f"{entry['review_id']}.mp4").write_bytes(b"corrupt")
    elif change == "arm":
        entry["arm"] = "motion_reference" if entry["arm"] == "baseline" else "baseline"
    elif change == "sample":
        entry["sample_id"] = "0" * 32
    elif change == "traversal":
        entry["review_id"] = "../../outside"
    elif change == "duplicate_take":
        mapping["mapping"].append(copy.deepcopy(entry))
    else:
        entry["frames"] = 900
    ratings_path.write_text(json.dumps(ratings))
    map_path.write_text(json.dumps(mapping))
    with pytest.raises(ValueError):
        summarize_reviews(experiment, review_bundle)


def test_empty_review_does_not_claim_quality(experiment, tmp_path):
    directory = tmp_path / "empty-review"
    assert export_review_bundle(experiment, directory)["clips"] == 0
    result = summarize_reviews(experiment, directory)
    assert result["decision"] == "not_evaluated"
    assert all(v["planned"] == 1 and v["reviewed"] == 0 for v in result["arms"].values())


def test_dual_review_reports_disagreement_without_promoting_take(experiment, review_bundle, tmp_path):
    first = json.loads((review_bundle / "reviewer/ratings.json").read_text())
    first["reviewer_id"] = "independent_a"
    for row in first["ratings"]:
        row.update(assessment="reviewed", identity_ok=True, scope_ok=True, endpoint_ok=True, visual_invariants_ok=True,
                   causal_link_ok=True, contact_frame=48, reaction_frame=55)
    second = copy.deepcopy(first)
    second["reviewer_id"] = "independent_b"
    second["ratings"][0]["causal_link_ok"] = False
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    for path, doc in zip(paths, (first, second)):
        path.write_text(json.dumps(doc))
    report = summarize_reviews(experiment, review_bundle, ratings_files=paths)
    assert report["judgments"]["disputed"] == 1
    assert report["judgments"]["pass"] == 1
    assert report["agreement"]["verdict_agreement"] == 0.5
    assert sum(count["passed"] for count in report["arms"].values()) == 1
    assert all(not w.accepted_shot_artifacts() for _, w in load_experiment(experiment)[1])
    assert main(["report", "--experiment", str(experiment), "--review-dir", str(review_bundle),
                 "--ratings-file", str(paths[0]), "--ratings-file", str(paths[1])]) == 0


def test_unjudgeable_review_not_conflated_with_unreviewed(experiment, review_bundle):
    path = review_bundle / "reviewer/ratings.json"
    doc = json.loads(path.read_text())
    doc["reviewer_id"] = "a"
    doc["ratings"][0].update(assessment="unjudgeable", notes="手掌被遮挡")
    path.write_text(json.dumps(doc))
    report = summarize_reviews(experiment, review_bundle)
    assert report["judgments"]["unjudgeable"] == 1
    assert report["judgments"]["unreviewed"] == 1
    assert sum(count["passed"] for count in report["arms"].values()) == 0


def test_legacy_review_keeps_explicit_version_and_cannot_enter_consensus(experiment, review_bundle):
    path = review_bundle / "reviewer/ratings.json"
    doc = json.loads(path.read_text())
    doc.pop("version")
    for row in doc["ratings"]:
        row.pop("causal_link_ok")
        row.update(identity_ok=True, scope_ok=True, endpoint_ok=True, contact_frame=48, reaction_frame=55)
    path.write_text(json.dumps(doc))
    report = summarize_reviews(experiment, review_bundle)
    assert report["assessment_versions"] == ["legacy-v1"]
    assert report["judgments"]["pass"] == 2
    with pytest.raises(ValueError, match="新版"):
        summarize_reviews(experiment, review_bundle, ratings_files=[path, path])


@pytest.mark.asyncio
async def test_unknown_creation_keeps_budget_and_disables_sdk_retry(experiment, monkeypatch):
    from types import SimpleNamespace
    clients, calls = [], []
    def initialize(api):
        api.ark_client = SimpleNamespace(max_retries=2)
        clients.append(api.ark_client)
    async def generate(api, **kwargs):
        calls.append(kwargs)
        return {"status": "failed", "error_type": "submission_unknown", "error": "unknown submission"}
    monkeypatch.setattr(SeedanceAPI, "_init_clients", initialize)
    monkeypatch.setattr(SeedanceAPI, "generate", generate)
    with pytest.raises(RuntimeError):
        await execute_experiment(experiment)
    assert len(calls) == 1 and clients[0].max_retries == 0
    assert (await execute_experiment(experiment))["status"] == "blocked"
    assert len(calls) == 1
    rows = experiment_report(experiment)["samples"]
    assert sum(row["unresolved_reservations"] for row in rows) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [None, 408, 500, 503])
async def test_api_creation_unknown_is_not_classified_as_safe_rejection(status):
    from types import SimpleNamespace
    from unittest.mock import Mock
    error = RuntimeError("do not expose transport internals")
    error.status_code = status
    api = object.__new__(SeedanceAPI)
    create = Mock(side_effect=error)
    api.ark_client = SimpleNamespace(content_generation=SimpleNamespace(tasks=SimpleNamespace(create=create)))
    result = await api.generate("test", model=MODEL, duration=4)
    assert result["error_type"] == "submission_unknown"
    assert "transport internals" not in result["error"]
    assert create.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", ["authentication", "permission_denied"])
async def test_configuration_rejection_stops_first_request_and_releases_budget(experiment, monkeypatch, error_type):
    calls = []
    async def generate(api, **kwargs):
        calls.append(kwargs)
        return {"status": "failed", "error_type": error_type, "error": "configuration rejected"}
    monkeypatch.setattr(SeedanceAPI, "generate", generate)
    with pytest.raises(RuntimeError):
        await execute_experiment(experiment)
    assert len(calls) == 1
    _, loaded = load_experiment(experiment)
    ledger = loaded[0][1].manifest.paid_take_budget.reservations
    assert len(ledger) == 1 and ledger[0].status == "released"
    assert not loaded[1][1].manifest.paid_take_budget.reservations


@pytest.mark.asyncio
@pytest.mark.parametrize("status,expected", [(401, "authentication"), (403, "permission_denied")])
async def test_api_auth_rejection_is_explicit(status, expected):
    from types import SimpleNamespace
    from unittest.mock import Mock
    error = RuntimeError("configuration rejected")
    error.status_code = status
    api = object.__new__(SeedanceAPI)
    api.ark_client = SimpleNamespace(content_generation=SimpleNamespace(tasks=SimpleNamespace(create=Mock(side_effect=error))))
    assert (await api.generate("test", model=MODEL, duration=4))["error_type"] == expected
