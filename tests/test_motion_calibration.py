"""Real local synthetic media, independent geometric checks and simulated forms only."""

import json
import math
import shutil
import socket
from pathlib import Path

import pytest

from pipeline.motion_assessment import contract_digest
from pipeline.motion_plan import MotionPlan, compile_motion_previs
from tools.motion_calibration import CASES, build_calibration, calibration_frames, score_calibration

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/contact_previs"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network forbidden")
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    destination = tmp_path_factory.mktemp("calibration") / "bundle"
    result = build_calibration(destination)
    assert result["clips"] == 6 and result["remote_calls"] == 0
    return destination


def test_rendered_bundle_is_unrated_and_answers_are_private(bundle):
    result = score_calibration(bundle)
    assert result["reviewers"][0]["answered"] == 0
    assert result["generation_quality"] == "not_evaluated"
    ratings = (bundle / "reviewer/ratings.json").read_text()
    assert all(case not in ratings for case in CASES)
    assert len(list((bundle / "reviewer").glob("*.mp4"))) == 6
    assert not (bundle / "manifest.json").exists()
    with pytest.raises(ValueError):
        build_calibration(bundle)


def test_constructed_motion_perturbations_measured_independently():
    source = json.loads((EXAMPLE / "storyboard.json").read_text())
    plan = MotionPlan.model_validate_json((EXAMPLE / "motion.json").read_text())
    frames = compile_motion_previs(source["shots"][0], "16:9", plan)["frames"]
    snapshot = json.dumps(frames)
    miss = calibration_frames(frames, "missed_contact")
    distances = [math.dist(s["entities"]["attacker"]["lead_hand"], s["entities"]["defender"]["lead_hand"]) for s in miss]
    assert min(distances) > 0.15
    early = calibration_frames(frames, "premature_reaction")
    assert early[24]["entities"]["defender"]["pelvis"][0] > frames[24]["entities"]["defender"]["pelvis"][0] + 0.09
    assert early[48]["entities"] == frames[48]["entities"]
    frozen = calibration_frames(frames, "no_reaction")
    assert all(s["entities"]["defender"] == frames[0]["entities"]["defender"] for s in frozen)
    swapped = calibration_frames(frames, "roles_swapped")
    assert swapped[48]["entities"]["attacker"] == frames[48]["entities"]["defender"]
    assert json.dumps(frames) == snapshot


def test_all_pass_shortcut_is_exposed_by_negative_controls(bundle, tmp_path):
    document = json.loads((bundle / "reviewer/ratings.json").read_text())
    document["reviewer_id"] = "simulated_always_pass"
    for rating in document["ratings"]:
        rating.update(assessment="reviewed", identity_ok=True, scope_ok=True, endpoint_ok=True, visual_invariants_ok=True,
                      causal_link_ok=True, contact_frame=48, reaction_frame=55)
    path = tmp_path / "simulated.json"
    path.write_text(json.dumps(document))
    report = score_calibration(bundle, [path])
    assert report["reviewers"][0]["false_acceptances"] == 5
    assert report["reviewers"][0]["correct"] == 1
    assert report["independent_human_validation"] == "not_evaluated"


def test_synthetic_oracle_form_only_checks_plumbing(bundle, tmp_path):
    manifest = json.loads((bundle / "calibration_manifest.json").read_text())
    truth = {row["review_id"]: row["expected"] for row in manifest["mapping"]}
    document = json.loads((bundle / "reviewer/ratings.json").read_text())
    document["reviewer_id"] = "simulated_oracle"
    assert document["contract_sha256"] == contract_digest(manifest["contract"])
    for rating in document["ratings"]:
        expected = truth[rating["review_id"]]
        if expected == "unjudgeable":
            rating.update(assessment="unjudgeable", notes="测试用：接触不可见")
        else:
            rating.update(assessment="reviewed", identity_ok=True, scope_ok=True, endpoint_ok=True, visual_invariants_ok=True,
                          causal_link_ok=expected == "pass", contact_frame=48, reaction_frame=55)
    path = tmp_path / "oracle.json"
    path.write_text(json.dumps(document))
    assert score_calibration(bundle, [path])["reviewers"][0]["correct"] == 6
    document["reviewer_id"] = "simulated_second"
    other = tmp_path / "other.json"
    other.write_text(json.dumps(document))
    report = score_calibration(bundle, [path, other])
    assert report["agreement"]["verdict_agreement"] == 1
    assert report["independent_human_validation"] == "not_evaluated"


@pytest.mark.parametrize("change", ["path", "expected", "duplicate", "missing", "bytes", "frames"])
def test_calibration_tampering_rejected(bundle, tmp_path, change):
    target = tmp_path / "copy"
    shutil.copytree(bundle, target)
    path = target / "calibration_manifest.json"
    manifest = json.loads(path.read_text())
    entry = manifest["mapping"][0]
    if change == "path":
        entry["review_id"] = "../outside"
    elif change == "expected":
        entry["expected"] = "fail"
    elif change == "duplicate":
        manifest["mapping"][1] = entry
    elif change == "missing":
        manifest["mapping"].pop()
    elif change == "bytes":
        (target / "reviewer" / f"{entry['review_id']}.mp4").write_bytes(b"different")
    else:
        entry["frames"] = 95
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        score_calibration(target)


def test_encoding_failure_does_not_publish_or_leave_staging(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("test encoder failure")
    monkeypatch.setattr("tools.motion_calibration._encode_motion_frames", fail)
    with pytest.raises(RuntimeError):
        build_calibration(tmp_path / "incomplete")
    assert not list(tmp_path.iterdir())


def test_publish_failure_removes_only_own_partial_files(tmp_path, monkeypatch):
    import os
    original = os.link
    calls = []
    def fail_second(source, destination):
        calls.append(destination)
        if len(calls) == 2:
            raise OSError("test publication failure")
        original(source, destination)
    marker = tmp_path / "preserved.txt"
    marker.write_text("preserve")
    monkeypatch.setattr("tools.motion_calibration.os.link", fail_second)
    with pytest.raises(OSError):
        build_calibration(tmp_path / "incomplete")
    assert list(tmp_path.iterdir()) == [marker]
    assert marker.read_text() == "preserve"
