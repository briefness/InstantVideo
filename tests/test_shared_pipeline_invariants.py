"""Offline regressions across orchestration, Take identity and rejection evidence."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

import config
from pipeline.failure_evidence import semantic_review_failure_evidence
from pipeline.generator import VideoGenerator
from pipeline.models import RunOptions, TakeRecordState
from pipeline.orchestrator import VideoPipeline
from pipeline.production_plan import apply_production_plan, build_production_plan
from pipeline.run_state import PaidTakeBudgetExhaustedError, RunWorkspace, _merge_take_history
from pipeline.semantic_review import SemanticReview
from pipeline.causality import ACTION_CONTRACT_VERSION


@pytest.fixture(autouse=True)
def no_external_http(monkeypatch):
    import httpx
    monkeypatch.setattr(httpx.Client, "send", lambda *_a, **_k: pytest.fail("offline HTTP forbidden"))
    async def reject(*_a, **_k):
        pytest.fail("offline async HTTP forbidden")
    monkeypatch.setattr(httpx.AsyncClient, "send", reject)


@pytest.mark.parametrize("field,code", [
    ("action_timing_valid", "contract.action_timing_invalid"),
    ("requested_outcome_valid", "contract.requested_outcome_invalid"),
    ("action_sequence_valid", "contract.action_sequence_invalid"),
])
def test_objective_rejections_keep_stable_failure_codes(field, code):
    review = SimpleNamespace(accepted=False, **{field: False})
    evidence = semantic_review_failure_evidence(review)
    assert [item["code"] for item in evidence] == [code]
    assert evidence[0]["category"] == "contract"


class AtVideoStage(RuntimeError):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("frozen", [True, False])
async def test_music_never_retimes_a_frozen_plan_but_keeps_legacy_behavior(tmp_path, monkeypatch, frozen):
    story = {"title": "Product or narrative montage", "mood": "cinematic",
        "story_arc": {"goal": "show the product", "stakes": "understand the design",
                      "turning_point": "reveal the mechanism", "resolution": "show the result"},
        "shots": [
        {"shot_id": i, "duration": 5, "prompt_en": "A clear cinematic scene",
         "primary_action": "show the subject", "scene_id": f"scene_{i}",
         "scene_description": "A product on a studio table",
         "narrative_beat": {"state_before": f"state_{i-1}", "state_change": "revealed",
                            "state_after": f"state_{i}"}}
        for i in (1, 2, 3)]}
    if frozen:
        apply_production_plan(story, build_production_plan("balanced", 15))
    else:
        story.pop("story_arc")
        for shot in story["shots"]:
            shot.pop("narrative_beat")
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr("pipeline.orchestrator.generate_storyboard", lambda **_: story)
    monkeypatch.setattr("pipeline.orchestrator.beat_analyzer.analyze_beats", lambda _: {
        "bpm": 120., "beat_interval": .5, "beat_times": [i * .5 for i in range(120)],
        "total_duration": 60.})
    class ObserveGenerator:
        def __init__(self, *_a, **_k):
            pass
        async def generate_all(self, generated):
            assert [shot["duration"] for shot in generated["shots"]] == ([5] * 3 if frozen else [15] * 3)
            raise AtVideoStage
    monkeypatch.setattr("pipeline.orchestrator.VideoGenerator", ObserveGenerator)
    with pytest.raises(AtVideoStage):
        await VideoPipeline(music_path="offline-track.wav", paid_take_budget=0).run("15秒产品或剧情短片")


@pytest.mark.parametrize("field,new", [
    ("prompt_fingerprint", "b" * 64),
    ("compiled_contract_fingerprint", "c" * 64),
    ("compiled_contract_version", "different-version"),
    ("model_used", "different-model"),
    ("resolution_used", "720p"),
])
def test_existing_take_cannot_change_submission_identity(field, new):
    original = TakeRecordState(take_id="1:shots/rejected_1.mp4", disposition="rejected",
        local_path="shots/rejected_1.mp4", prompt_fingerprint="a" * 64,
        compiled_contract_fingerprint="a" * 64, compiled_contract_version="action-contract-v2",
        model_used="submitted-model", resolution_used="480p")
    with pytest.raises(ValueError, match="来源"):
        _merge_take_history([original], original.model_copy(update={field: new}))


def test_take_rereview_preserves_origin_and_can_update_acceptance():
    original = TakeRecordState(take_id="1:shots/rejected_1.mp4", disposition="rejected",
        local_path="shots/rejected_1.mp4", prompt_fingerprint="a" * 64,
        compiled_contract_fingerprint="a" * 64, compiled_contract_version="original-version")
    reviewed = TakeRecordState(take_id=original.take_id, disposition="accepted",
        local_path=original.local_path, semantic_accepted=True,
        accepted_contract_fingerprint="b" * 64, accepted_contract_version="current-version")
    merged = _merge_take_history([original], reviewed)
    assert len(merged) == 1
    assert merged[0].disposition == "accepted"
    assert merged[0].accepted_contract_fingerprint == "b" * 64
    assert merged[0].compiled_contract_version == "original-version"
    assert merged[0].prompt_fingerprint == "a" * 64
    assert original.disposition == "rejected"
    legacy = original.model_copy(update={"prompt_fingerprint": None})
    assert _merge_take_history([legacy], original)[0].prompt_fingerprint == "a" * 64


@pytest.mark.asyncio
@pytest.mark.parametrize("second_pending", [False, True])
async def test_retake_submission_callback_cannot_relabel_the_previous_take(tmp_path, monkeypatch, second_pending):
    workspace = RunWorkspace.create(tmp_path, RunOptions(request="通用连续动作", paid_take_budget=2))
    shot = {"shot_id": 1, "duration": 5, "prompt_en": "A machine moves a payload into place.",
            "scene_description": "Factory"}
    story = {"resolution": "480p", "aspect_ratio": "16:9", "shots": [shot]}
    workspace.save_storyboard(story)
    snapshots = []
    def progress(result):
        payload = asdict(result)
        # 与正式编排相同：完整传递本条结果的状态、素材和来源。
        payload.pop("character_ref_path")
        workspace.record_shot(**payload)
        snapshots.append(workspace.manifest.shots["1"].model_dump())
    generator = VideoGenerator(str(workspace.path), on_progress=progress,
        reserve_paid_take=workspace.reserve_paid_take,
        confirm_paid_take_submission=workspace.confirm_paid_take_submission,
        reconcile_paid_take=workspace.reconcile_paid_take)
    class API:
        supports_last_frame = True
        calls = 0
        async def generate(self, **kwargs):
            self.calls += 1
            task = f"offline-task-{self.calls}"
            kwargs["on_submitted"](task)
            if self.calls == 2 and second_pending:
                return {"status": "failed", "provider_task_id": task,
                        "error_type": "poll_timeout", "error": "Timeout (600s)"}
            return {"status": "succeeded", "provider_task_id": task, "video_url": task}
        async def download_video(self, url, destination):
            Path(destination).write_bytes(url.encode())
    class Reviewer:
        calls = 0
        def review(self, *_a, **_k):
            self.calls += 1
            accepted = self.calls == 2
            return SemanticReview(accepted=accepted, required_entities_visible={},
                action_geometry_valid=accepted, primary_action_completed=accepted,
                observed_end_state={"location": "factory", "subject": "machine",
                                    "action_phase": "payload settled"},
                failure_reason="payload did not move" if not accepted else "")
    generator.api = API()
    generator.semantic_reviewer = Reviewer()
    monkeypatch.setattr("pipeline.generator.check_video_quality",
                        lambda _: {"pass": True, "quality_score": 100, "issues": []})
    monkeypatch.setattr(generator, "_extract_local_tail_frame", lambda *_: str(tmp_path / "tail.jpg"))
    result = await generator._generate_single_shot(shot, None, None, story)
    assert result.status == ("running" if second_pending else "success")
    assert generator.api.calls == 2
    first_fingerprint = result.prompt_attempts[0]["fingerprint"]
    second_fingerprint = result.prompt_attempts[1]["fingerprint"]
    assert first_fingerprint != second_fingerprint
    first_records = [take for snapshot in snapshots for take in snapshot["take_history"]
                     if take["local_path"].endswith("rejected_1.mp4")]
    assert first_records and all(take["prompt_fingerprint"] == first_fingerprint for take in first_records)
    history = workspace.manifest.shots["1"].take_history
    assert len(history) == (1 if second_pending else 2)
    assert history[0].semantic_accepted is False
    resumed = RunWorkspace.resume(workspace.path)
    reservations = resumed.manifest.paid_take_budget.reservations
    assert len(reservations) == 2
    assert [item.provider_task_id for item in reservations] == ["offline-task-1", "offline-task-2"]
    assert all(item.status == "reconciled" for item in reservations)
    with pytest.raises(PaidTakeBudgetExhaustedError):
        resumed.reserve_paid_take(1)
    if second_pending:
        pending = resumed.resumable_pending_tasks()[1]
        assert pending["task_id"] == "offline-task-2"
        assert pending["prompt_fingerprint"] == second_fingerprint
    else:
        assert history[1].prompt_fingerprint == second_fingerprint
        assert history[1].semantic_accepted is True


@pytest.mark.asyncio
async def test_poll_without_repeated_task_id_keeps_the_known_submission(tmp_path):
    shot = {"shot_id": 1, "duration": 5, "prompt_en": "An industrial machine is waiting."}
    generator = VideoGenerator(str(tmp_path),
        reserve_paid_take=lambda _: pytest.fail("poll must not reserve another Take"))
    fingerprint = "a" * 64
    generator.resume_tasks = {1: {
        "task_id": "known-task", "prompt_profile": "normal", "prompt_fingerprint": fingerprint,
        "compiled_contract_version": ACTION_CONTRACT_VERSION,
        "compiled_contract_fingerprint": generator._compiled_contract_fingerprint(shot),
    }}
    provenance = generator._generation_provenance(
        shot, [], None, prompt_profile="normal", prompt_fingerprint=fingerprint)
    generator._write_generation_provenance(1, provenance, "known-task")
    class PollOnly:
        async def poll_task(self, task_id, **_):
            assert task_id == "known-task"
            return {"status": "failed", "error_type": "poll_timeout", "error": "still pending"}
    generator.api = PollOnly()
    result = await generator._generate_single_shot(
        shot, None, None, {"resolution": "480p", "shots": [shot]})
    assert result.status == "running"
    assert result.provider_task_id == "known-task"


@pytest.mark.asyncio
@pytest.mark.parametrize("budget,semantic_rejection", [(0, False), (1, False), (1, True)])
async def test_budget_denial_does_not_publish_an_unsubmitted_take(tmp_path, monkeypatch, budget, semantic_rejection):
    workspace = RunWorkspace.create(tmp_path, RunOptions(request="workshop", paid_take_budget=budget))
    shot = {"shot_id": 1, "duration": 5, "scene_description": "workshop",
            "prompt_en": "A machine moves its workpiece."}
    story = {"resolution": "480p", "shots": [shot]}
    workspace.save_storyboard(story)
    def record(result):
        fields = asdict(result)
        fields.pop("character_ref_path")
        workspace.record_shot(**fields)
    generator = VideoGenerator(str(workspace.path), on_progress=record,
        reserve_paid_take=workspace.reserve_paid_take,
        confirm_paid_take_submission=workspace.confirm_paid_take_submission,
        reconcile_paid_take=workspace.reconcile_paid_take)
    class TerminalProvider:
        supports_last_frame = True
        calls = 0
        async def generate(self, **kwargs):
            self.calls += 1
            kwargs["on_submitted"]("terminal-original")
            if semantic_rejection:
                return {"status": "succeeded", "provider_task_id": "terminal-original", "video_url": "offline"}
            return {"status": "failed", "provider_task_id": "terminal-original",
                    "error_type": "copyright_policy", "error": "terminal rejection"}
        async def download_video(self, url, destination):
            Path(destination).write_bytes(b"rejected offline material")
    generator.api = TerminalProvider()
    if semantic_rejection:
        generator.semantic_reviewer = SimpleNamespace(review=lambda *_a, **_k: SemanticReview(
            accepted=False, required_entities_visible={}, action_geometry_valid=False,
            primary_action_completed=False, observed_end_state={}, failure_reason="no visible workpiece movement"))
        monkeypatch.setattr("pipeline.generator.check_video_quality",
            lambda _: {"pass": True, "quality_score": 100, "issues": []})
    result = await generator._generate_single_shot(shot, None, None, story)
    restored = RunWorkspace.resume(workspace.path)
    state = restored.manifest.shots["1"]
    assert result.status == "failed"
    assert generator.api.calls == budget
    assert len(state.prompt_attempts) == budget
    assert len(restored.manifest.paid_take_budget.reservations) == budget
    assert not restored.resumable_pending_tasks()
    assert len(state.take_history) == int(semantic_rejection)
    if budget:
        assert result.provider_task_id == state.provider_task_id == "terminal-original"
        assert result.prompt_fingerprint == state.prompt_attempts[0].fingerprint
        assert state.prompt_attempts[0].outcome == ("succeeded" if semantic_rejection else "failed")
        if semantic_rejection:
            assert state.take_history[0].prompt_fingerprint == result.prompt_fingerprint
            assert result.semantic_accepted is False
        else:
            assert result.provider_error_type == "copyright_policy"
    else:
        assert result.provider_task_id is None
