"""Exercise ordered event acceptance through the real reviewer and a local SDK double."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.causality import compile_action_contract
from pipeline.semantic_review import SemanticReviewUnavailableError, SemanticTakeReviewer
from test_action_sequence import sequence_story


@pytest.fixture(autouse=True)
def offline_frames(monkeypatch):
    import httpx
    def forbidden(*_a, **_k):
        pytest.fail("sequence review must never use external HTTP")
    monkeypatch.setattr(httpx.Client, "send", forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "send", forbidden)
    monkeypatch.setattr("pipeline.semantic_review.get_video_duration", lambda _: 5.)
    monkeypatch.setattr("pipeline.semantic_review.extract_frame",
                        lambda _video, output, timestamp=None: Path(output).write_bytes(f"frame {timestamp}".encode()))


def event_response():
    return {
        "accepted": True, "required_entities_visible": [True, True],
        "action_geometry_valid": True, "primary_action_completed": True,
        "effect_path_valid": True, "reaction_causality_valid": True,
        "action_timing_valid": True, "requested_outcome_valid": True,
        "narrative_state_change_valid": True, "blocking_valid": True,
        "visual_invariants_valid": True, "boundary_continuity_valid": True,
        "identity_continuity_valid": True, "identity_crop_boxes": [],
        "observed_end_state": {"location": "workshop", "subject": "two machines",
                               "action_phase": "transfer completed"},
        "failure_reason": "",
        "action_event_evidence": [
            {"event_id": event_id, "effect_visible": True, "target_reaction_visible": True,
             "causal_order_valid": True, "scope_valid": True, "result_visible": True,
             "time_window_valid": True} for event_id in (1, 2)],
    }


class LocalCompletions:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop",
            message=SimpleNamespace(content=json.dumps(self.response), refusal=None))])


def reviewer_for(tmp_path, response):
    video = tmp_path / "candidate.mp4"
    video.write_bytes(b"offline sequence candidate")
    completions = LocalCompletions(response)
    reviewer = SemanticTakeReviewer(tmp_path,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)), model="offline-vision")
    return video, reviewer, completions


def process_shot(mode):
    # 同一事件模型表达搬运、喷涂、区域加热和连杆传动，而非某个武打主题。
    story = json.loads(json.dumps(sequence_story()).replace("attacker", "machine_a").replace("defender", "machine_b"))
    shot = story["shots"][0]
    action, result = {
        "direct_contact": ("moves the crate by contact", "crate shifts onto the platform"),
        "directed_path": ("sprays coating along a visible path", "panel becomes coated"),
        "area_effect": ("heats the defined region", "workpiece changes color inside the region"),
        "indirect_effect": ("drives the visible linkage", "linked platform rises"),
    }[mode]
    shot["scene_description"] = "Two workshop machines perform a controlled operation in turn."
    shot["primary_action"] = action
    shot["production_slot"] = {"required_outcome": result}
    for beat in shot["action_beats"][:2]:
        beat["action"] = action
        beat["visible_result"] = result
        beat["interaction_geometry"].update(interaction_mode=mode,
            source="active machine actuator", effect_region="receiver workpiece",
            effect_motion="propagate" if mode == "indirect_effect" else "static")
    shot["interaction_geometry"] = deepcopy(shot["action_beats"][1]["interaction_geometry"])
    return shot


@pytest.mark.parametrize("mode", ["direct_contact", "directed_path", "area_effect", "indirect_effect"])
@pytest.mark.parametrize("first_event_passes", [True, False])
def test_final_result_cannot_replace_a_missing_intermediate_event(tmp_path, mode, first_event_passes):
    shot = process_shot(mode)
    response = event_response()
    response["action_event_evidence"][0]["target_reaction_visible"] = first_event_passes
    video, reviewer, sdk = reviewer_for(tmp_path, response)
    review = reviewer.review(str(video), shot)
    assert review.requested_outcome_valid is True
    assert review.accepted is first_event_passes
    assert review.action_sequence_valid is first_event_passes
    if not first_event_passes:
        assert "事件1" in review.failure_reason
    assert len(sdk.calls) == 1
    events = compile_action_contract(shot).review_projection["action_events"]
    assert len(events) == 2
    contract_text = sdk.calls[0]["messages"][1]["content"][0]["text"]
    assert mode in contract_text and "machine_a" in contract_text and "machine_b" in contract_text
    sample_labels = [item["text"] for item in sdk.calls[0]["messages"][1]["content"]
                     if item["type"] == "text" and item["text"].startswith("Current ")]
    for time in ("1.000s", "1.200s", "3.000s", "3.200s"):
        assert any(time in label for label in sample_labels)


@pytest.mark.parametrize("malformation", ["absent", "missing_event", "reordered", "missing_field", "duplicate", "wrong_boolean"])
def test_incomplete_or_mismatched_event_evidence_cannot_be_accepted(tmp_path, malformation):
    response = event_response()
    evidence = response["action_event_evidence"]
    if malformation == "absent":
        response.pop("action_event_evidence")
    elif malformation == "missing_event":
        evidence.pop(0)
    elif malformation == "reordered":
        evidence.reverse()
    elif malformation == "missing_field":
        evidence[0].pop("scope_valid")
    elif malformation == "duplicate":
        evidence[1]["event_id"] = 1
    else:
        evidence[0]["causal_order_valid"] = "true"
    video, reviewer, sdk = reviewer_for(tmp_path, response)
    with pytest.raises(SemanticReviewUnavailableError, match="无效结构"):
        reviewer.review(str(video), process_shot("direct_contact"))
    assert len(sdk.calls) == 2  # Only the same-video structural repair, no generation.


@pytest.mark.parametrize("change", ["none", "missing_evidence", "failed_event", "reordered"])
def test_cached_success_rechecks_raw_event_evidence(tmp_path, change):
    video, reviewer, sdk = reviewer_for(tmp_path, event_response())
    shot = process_shot("indirect_effect")
    assert reviewer.review(str(video), shot).accepted is True
    cache = next((tmp_path / "semantic_reviews").glob("*.json"))
    stored = json.loads(cache.read_text())
    assert stored["review"]["accepted"] is True
    if change == "missing_evidence":
        stored.pop("action_event_evidence")
    elif change == "failed_event":
        stored["action_event_evidence"][0]["time_window_valid"] = False
    elif change == "reordered":
        stored["action_event_evidence"].reverse()
    cache.write_text(json.dumps(stored))
    restored = SemanticTakeReviewer(tmp_path, client=reviewer.client, model="offline-vision")
    review = restored.review(str(video), shot)
    assert review.accepted is (change == "none")
    assert review.action_sequence_valid is (change == "none")
    assert len(sdk.calls) == 1
