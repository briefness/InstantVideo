# Seedance Video Pipeline

This project turns a story request into a sequence of generated shots, then accepts only footage that can safely drive the next shot and the final edit.

## Language

**Action Contract**:
The authoritative description of one shot's physical cause, visible progression, permitted scope, and required endpoint. Planning, prompting, review, and retake instructions must compile from this contract.
_Avoid_: Combat rules, prompt rules, review rules

**Planned State**:
The intended visible state before or after a shot, fixed before generation and never rewritten to pretend that generated footage matched the plan.
_Avoid_: Expected truth, canonical state

**Observed State**:
The visible state measured from one generated take, including uncertainty and deviations from the Planned State.
_Avoid_: Generated plan, assumed state

**Canonical State**:
The accepted Observed State that may anchor later shots. Rejected or unreviewed takes never update it.
_Avoid_: Latest state, planned state

**Take**:
One paid generation result for a shot, retained with its prompt, review, and lineage whether it is accepted or rejected.
_Avoid_: Attempt, clip result

**Phase Endpoint**:
The visible state that completes the current Action Contract phase, including a preparation endpoint that does not imply impact or narrative payoff.
_Avoid_: Outcome, visible result

**Non-Physical Cue**:
A visible indicator such as aiming guidance, charge light, signal, or targeting overlay that communicates preparation without physically changing a target.
_Avoid_: Effect, impact

**Physical Effect**:
A visible contact, emitted action, propagated force, or environmental interaction capable of physically changing a target.
_Avoid_: Cue, preparation

**Narrative Outcome**:
The visible story consequence produced by an active interaction or preserved in its aftermath.
_Avoid_: Phase endpoint, camera change

