# Seedance Video Pipeline

New flexible creation uses `story-context-v6-candidate-results`. The schedule
binder marks result claims `review-required-v1` before any lowering. Every claim
is retained as a candidate, regardless of the author's optional legacy importance
annotation. `CandidateResolution` permits states without importance; old
`EntityResolution` and relative result types keep their prior interpretation when
not explicitly pending. Source maps preserve the original annotation. Existing
independent result admission alone chooses obligations versus performance
directions, and final provider preflight rejects unreviewed candidates before
reserving a paid take. This removes an early author-label gate; it neither invents
missing target states nor approves the real failed film. Raw responses and old
accepted contracts are not migrated.

Whole-film preview is now an explicit persisted delivery mode, separate from
submission policy. `RunOptions.delivery_mode` defaults to `reviewed` for new and
old workspaces; `preview` selects the existing technical-only generator path
without changing global configuration. CLI and Web expose the same option and
resume preserves it. Preview uses actual tail frames, never fabricated observed
states, and exports `preview.mp4` plus `delivery_report.json` with content marked
`not_reviewed`. The observed-film-rhythm gate remains in the reviewed path; it
cannot measure preview clips without observations. Budget, media readiness,
technical QA, provenance and recovery checks still apply. This separates complete
delivery from content acceptance; it does not establish cinematic motion quality
or approve any previously rejected take.

Visual review v24 derives cross-shot applicability from supplied previous-frame
evidence. With no previous frame, the provider schema omits boundary judgments;
the adapter uses neutral conjunction values internally and records observation_scope.
These values are not observations of successful continuity. With a previous frame,
null still means unavailable and false remains failure. Original model responses
stay in their immutable diagnostic files. A real first-shot replay still stops on
unresolvable temporal evidence; removing an inapplicable check does not approve it.

New reviewed continuations with authored staging use `observed-entry-v7-path-guidance`.
For an interaction entry, the continuation response is stored separately as
`observed_entry_path`; it does not overwrite the first authored action or results.
The existing staging projection carries this runtime suggestion and fingerprints
it for generation and acceptance cache identity. Policy-safe prompts omit it
explicitly. Planning previews remain pre-observation projections; runtime guidance
can therefore change their staging fingerprint without changing the action plan.
Saved v6 continuations retain their original reconstruction behavior, while
transition entries retain their existing adaptation behavior. This protects
information ownership; it does not prove a model's path suggestion is sensible.

New event-authoring requests distinguish creation types from storage-compatible
types. `AuthoredStoryArc` requires nonblank goal/stakes/turning_point/resolution;
`CreationShot` uses `AuthoredContinuityState` to require nonblank
location/subject/action_phase/camera at both ends of a shot. Schema generation
and local response parsing use those same creation types. These were already
downstream requirements, not new quotas or plot rules. Historical `StoryArcSpec`,
`ContinuityStateSpec`, and `AuthoredShot` retain their existing defaults and
compilation behavior. Decorative and inapplicable state fields remain optional.
New response validation may stop before a derived authoring summary exists;
original `planning_*_response.json` and decoded response artifacts are saved
before validation and remain the authoritative failure evidence.

Flexible authoring's spatial maps share the action identity registry:
`camera.screen_positions` keys are character/object IDs and `blocking` keys are
character IDs. The contextual schema derives both from the frozen catalog.
Before another shot is requested, scene binding uses the execution contract's
shared entity-reference traversal (with the authored action list as its input),
then registers referenced scene objects even when they appear only in staging.
Free position/orientation/composition prose is retained. Unknown keys remain
errors; captured failures are never renamed or stripped to pass. This closes a
producer/consumer reference gap, without altering timing, semantic acceptance,
legacy schemas or saved plans. Schema fingerprints invalidate incompatible
planning checkpoints; no automatic migration or resubmission is permitted.

New flexible authoring uses `story-context-v5-entity-results`. Each action's
`resolution.states` maps frozen entity IDs to lists of state/importance claims.
Result ownership does not depend on whether that entity is the actor or target
of the current action. The shared result compiler projects these claims into
the existing outcomes/scene_changes contract and retains source paths, repeated
claims and contradictions. The existing required target result and 16-total /
14-scene claim limits still apply; no new action or timing quota is introduced.
Legacy role-relative resolutions remain supported for historical compilation,
but are not accepted as a new scheduled response. Captured failed plans are not
migrated or repaired. Semantic review remains necessary: an entity key cannot
prove the associated natural-language statement is true.

As in `story-context-v4-staging`, initial framing comes from
the existing slot, while rich camera, blocking and per-shot mood remain generation
guidance (`generation_direction=authored-staging-v1`). Normal prompts preserve
that guidance; policy-safe provenance names its omitted free-text fields. The
same source fingerprint protects pending/accepted reuse. Initial blocking is no
longer overwritten using the final event's participants. Plan-only automatically
saves both request projections in `generation_preview.json`, without fabricating
future references or observed states.

`blocking.action_target` is an entity ID, not action prose. Its creation schema
and local scene binding share the existing frozen entity registry. Real
captures previously passed because compilation replaced invalid targets; they
now remain negative regression evidence. Do not edit those captures or treat
their old `planning_ready` state as validation of the new authoring protocol.

New flexible shots also select `media_selection=planned-prefix-v1`. Vision,
identity midpoint and continuation tail use the video-stream prefix retained by
the existing exporter, not audio padding or a subsequently discarded ending.
Original media remains intact; missing markers preserve legacy semantics.

As in `story-context-v3-entry-scope`, optional entry has two
complete response alternatives: omit `opening` and the first interaction cannot
link to an in-shot predecessor; provide a nonempty `opening` (`first` plus optional
`following`) and the first interaction may link to it. Both prohibit first-event
overlap. This survives provider schema projection without relying on array
`minItems`. Cross-shot movement remains in `start_state.open_motion`; no boolean is
silently reset and no captured failed response is repaired or migrated.

As introduced in `story-context-v2-scheduled-performance`, the existing
Production Plan now owns required interaction keys and terminal result display in
each response, rather than asking a variable action array to rediscover the slot's
minimum work. `authoring_schedule.py` derives this interface from the existing
slot and action types, then unfolds authored opening, required interactions,
optional following units and final hold into the unchanged `AuthoredShot` list.
It preserves creative text, ordering, relative weights, participants and links;
it neither creates missing actions nor caps interactions at the minimum count.
Whole-film timing remains owned by the existing compiler. A populated interaction
field is not semantic evidence: independent planning review still judges the cause,
reaction and requested result. Historical imports and exact authoring keep their
array representation; incompatible checkpoint inputs stop without migration or
automatic resubmission.

As in `story-context-v1`, story identity registers physical
`locations` once, and each shot selects a registered `scene_id`. Camera changes do
not require invented per-shot location names. Each registered scene object has one
`prop_roles` declaration (visible, persistent, or both); the compiler projects it
into the existing `key_props`/`continuity_props` fields. Character identity stays
separate. The catalog and projected lists share the existing 40-object bound.
The new schema drops unreachable legacy definitions and is sent through the
structured-output field once, not repeated in the prompt. Action/state prose,
semantic checks and the execution contract remain unchanged. Exact authoring and
historical imports retain their previous representation; failed raw responses are
never deduplicated or rewritten to migrate them into this protocol.

All live intent, authoring, visual-review and continuation JSON boundaries now
reject duplicate object keys. Visual review also requires exactly one completed
`stop` response; complete-looking JSON with an unfinished envelope is unavailable,
not an accepted verdict or a reason to pay for a retake. Historical review caches
retain their original evaluator semantics.
Visual and continuation responses are captured before validation in separate
diagnostic directories. They are never read as accepted verdicts or executable
entry plans. Explicit re-review preserves prior raw responses and existing media;
these diagnostics do not create permission for another paid video submission.

Scene-binding recovery distinguishes a captured `pending` candidate from an
`applied` judgment. Negative or locked-shot-rejected candidates survive recovery
without another model call or lock bypass. A pre-request record covers unknown
transport outcomes; legacy successful caches stay compatible. This uses the
existing workspace lock, not a new workflow engine.

New flexible event authoring binds adjacent motion through `link_previous` rather
than a second free-text `carry` cause. The author writes the concrete maneuver in
`action` and selects continuity; the compiler owns the immediate predecessor and
projects a directional link into the existing Action Contract. A first unit has no
in-shot predecessor; cross-shot motion remains in `start_state.open_motion`.
Overlap still requires an explicitly linked interaction with shared participants.
New formal responses must satisfy the same schema locally even if a provider
ignores it. Historical imports and exact authoring retain `carry` unchanged; no
saved plan or failed response is migrated. Checkpoint input fingerprints distinguish
the new request. This removes contradictory duplicated handoff prose, not all
possible semantic errors in the remaining creative actions or generated footage.

Flexible outcome visual review now omits suggested event timestamps and the
authored activity schedule at the external observation boundary. Actual labeled
frames, ordered participants/results, creative brief and measured-duration
requirements remain. The shared Action Contract, saved plan and exact/reference
choreography retain their original schedule. Reviewer version v23 separates this
input protocol from old cached judgments; historical footage is not reclassified.

Submission scheduling uses the workspace's read-only `check_paid_take` before
preparing a new attempt or announcing a semantic retake. The atomic reservation
rechecks the same policy under the manifest lock, so preflight is not a spending
authorization and cannot race past the ledger. Polling an existing task skips
this check. Local recovery records its mode in `recovery_actions`, restores known
generation provenance and never substitutes a cache label for a provider model.
New sidecars also retain model and resolution; missing historical fields remain
unknown. These changes improve evidence and lifecycle correctness, not proven
cinematic performance. The real 20261101_140601_984099 Shot 1 remains rejected.

Explicit user endings now remain source-owned during event-story identity creation:
the identity response schema binds story_arc.resolution to the verbatim ending
evidence, and local validation enforces the same contract before requesting shots.
No-ending briefs retain creative resolution freedom. New final planning reviews
replace the bare outcome boolean with requested-outcome-v1: a relation (entailed,
contradicted, undetermined), references to actual final-shot result/state text, and
an explanation. User requirements and story summaries cannot prove themselves.
The existing admission call and internal verdict remain; no extra model calls or
retries were added. Versioned raw cache and evidence retain the source requirement;
historical caches keep their original contract without migration or re-review.
This prevents requirement rewriting and unsupported protocol approval. It does not
prove that a model's evidence interpretation is correct or that footage satisfies
the plan. Real video quality still requires observed-video validation.

This project turns a story request into a sequence of generated shots, then accepts only footage that can safely drive the next shot and the final edit.

New formal runs persist `planning_checkpoint.json` before each creative step and
after its complete return. Intent, identity and individual event shots can replay
after interruption; legacy draft/correction use coarser checkpoints. Replayed data
still passes the ordinary typed compiler, readiness and independent semantic review.
A returned step is not an accepted plan. Inputs, model/configuration and record
integrity must match; failed or outcome-unknown steps never retry on resume.
The existing workspace execution lock and an atomic planning record own progress.
Partial recovery applies only before a storyboard/shot ledger has been established;
a missing frozen storyboard cannot be silently regenerated. Historical diagnostic
files without this protocol are not adopted or rewritten as executable checkpoints.

Schema-bound calls through the shared storyboard transport now stream responses.
An incremental object-key guard closes a response on duplicate decoded keys in
the same object; partial raw content is preserved when a diagnostic sink exists.
The final JSON parser also rejects duplicate keys, including legacy wrapped JSON,
instead of allowing last-key-wins to overwrite a judgment. Missing stream finish,
refusal and transport failures cannot become accepted content or trigger a retry.
The formal authoring client disables SDK-level HTTP retries as well.
This is response integrity and early cancellation, not a repair of model output
or proof of provider reliability. The real text-only trial `20261101_021739_192672`
failed on a repeating first-shot response; its original evidence remains unchanged.

Authored action compilation separates character identity membership from required
visual participation. Every registered interaction target, including scene objects,
survives into `required_visible_entities`; objects are not promoted to characters
or face-reference subjects. Unused props and incidental decoration are not added
to that required set. Event geometry consumes authored body blocking, falling back
to existing camera positions; absent positions remain unspecified, never inferred
from prose. This projection happens at new authoring compilation, not during
historical workspace migration or review recovery.

New result-necessity reviews use `planning-semantics-v8-evidence-sources`: the program
binds every action and candidate result to a fixed response key before requesting
semantic judgments. The model does not generate source IDs or repeat source prose
to identify claims. A unit without candidate results needs no result admission;
a unit with candidates requires a judgment for every claim. Cause, classification,
whole-performance continuity and requested-outcome checks still apply. Raw wire
responses are cached before validation, including invalid/negative responses;
resume never implicitly retries them. Each classification selects a source from its
own enumerated domain: user text, readable forward continuity fields, or current
action/response text. Replaceable direction uses a null reference and explanation.
The program resolves user references to exact original text; the model provides
its reasoning separately. Composition distinguishes one necessity category from
mixed or uncertain categories; multiple details alone do not imply a mixed claim.
Mixed/uncertain claims remain rejected, never split or reclassified automatically.
The same source catalog governs the response schema and decoding, including the
next shot across review batches. It binds references, not semantic truth.
Unversioned, v6 and v7 historical caches retain their original payload fingerprint and
validation rather than being migrated or repaired. The 2026-11-01 text-only trial
114005 failed review: a prose/object/out-of-range citation was accepted by the old
provider schema but rejected locally. The review also incorrectly treated
incapacitation as satisfying a requested death. The citation fix does not establish
semantic accuracy; the original trial remains failed and no video was submitted.
The later `20261101_144550_063903` failure cited authored results as user evidence
and marked those claims mixed. Its immutable replay remains rejected. V8 prevents
cross-domain references by construction, but cannot prove that a selected source
semantically supports a judgment. No new reviewer, retry, topic rule or video call
is introduced by this change.

Authored visual direction is preserved as optional `visual_description` through
typed persistence and normal provider compilation. Legacy `prompt_en` and
`scene_description` remain excluded from explicit-action prompt compilation;
policy-safe recompilation still excludes free-form visual prose. New visual
direction binds both submission provenance and acceptance context, without becoming
an action verdict. Absent fields are not injected into historical storyboards.
New reviewed-result continuation uses `observed-entry-v6-reviewed-results` to
project admitted outcomes for every entry type, including transitions and
compatibility-only entries. Existing cached versions reconstruct their original
input fingerprint and are not migrated or silently regenerated.

New v8 plans freeze explicit total duration and generated-shot count with verbatim
request evidence. The configured planning range is 4–600 seconds, each generated
shot 4–15 seconds; infeasible combinations stop instead of silently shortening the
film or changing the requested count. Legacy plan versions retain their allocation.
Long v8 semantic admission uses bounded batches of complete shots with prior-shot
context, final-batch outcome checks and fingerprinted resumable evidence.

## Language

**Action Contract**:
The authoritative description of one shot's physical cause, visible progression, permitted scope, and required endpoint. Planning, prompting, review, and retake instructions must compile from this contract.
_Avoid_: Combat rules, prompt rules, review rules

**Action Event**:
One ordered physical interaction inside an Action Contract, with its own actor, target, geometry, time window, effect/reaction times, and required visible result. Different events may exchange actor roles. Every event must be evidenced; a final pose cannot replace intermediate events.
An explicit response may be impact, evasion, or deflection; not every attack must hit. Adjacent events can declare a concrete flow from the preceding movement and overlap in time while their starts, ends and effects remain ordered. Overlap counts only once toward action duration. Flow is observed continuity of movement, not mandatory turn-taking or extra event count.
New v6 plans distinguish exact choreography from performance-guided choreography.
Exact reproduction and motion references retain strict paths/windows. Performance
guidance allows route and pacing variation while retaining every ordered cause,
participant, response, scope and required state change. Measured observation times,
not authored windows, establish interaction occupancy and result-hold capacity.
The original creative brief is independently checked; flexibility cannot remove a
user requirement. Existing workspaces have no new policy fields injected on load.
New flexible authoring uses `resolution`: a required `target_state`, nullable
`actor_state`, and explicitly classified `scene_changes`. The compiler binds the
primary owners to the event's roles and projects them into `outcomes` (entity/state).
Authoring may contain multiple statements about the same owner. Structural lowering
preserves each statement and its required/incidental obligation one-to-one, including
duplicates and contradictions; it neither merges prose nor deduplicates by entity.
New natural-language flexible plans mark authored results as candidates
(`review-required-v1`). The existing independent planning review classifies every
candidate as a user requirement, forward continuity dependency, minimal functional
response, or replaceable performance direction. Source paths and reasons are kept;
user evidence must quote the original request and dependency pointers must resolve
forward. Mixed/uncertain obligations or broken whole-performance continuity stop
before video submission. No extra author/reviewer retry stage is added.
The admitted execution copy (`reviewed-results-v1`) keeps every original statement
and stores direction paths. Generation retains them as suggestions; required state
projections, visual review, and observed-state continuation share the same filter.
Every cause/target response, user brief, final outcome, identity, camera and film
timing check remains binding. End poses do not independently recreate removed
state obligations. The original frozen storyboard remains the cache input;
admission is reproducibly derived on resume and recorded in `result_admission.json`.
Historical workspaces without this marker retain their original contract/verdict.
Classification remains a model judgment: protocol tests do not establish semantic
accuracy or prove fluent generated motion. The current 2D motion module remains
limited to single-contact fixed-camera unarmed blocking; text-guided runs are not
skeletal control.
Required primary claims enter `outcomes`, other required/optional claims enter
`scene_changes` (including optional primary details). `planning_lowering.json` records
source paths. Independent review still judges consistency and completeness. Historical
outcomes without the explicit scene_changes protocol retain unique-owner validation.
Creative prose shares a bounded description budget rather than short-label limits;
IDs, cardinality, numbers and execution constraints retain separate strict bounds.
The existing story-identity call freezes `scene_entities`; target and scene-result
IDs share that enum with the registered characters. Per-shot prop registrations
are derived from these frozen references, without fuzzy aliases or another call.
Registered scene objects/other characters may change in the same event; required
changes join the review obligations, while incidental details are optional and
must not hide a user requirement or a later shot's dependency. This distinction
is checked by independent planning review, not inferred from prose by the compiler.
Historical authored `outcomes` retain their strict ownership checks; no auto-migration.
The performance-v2 projection keeps suggested execution out of result admission; the
planning reviewer must explicitly attest outcome completeness/separation. Timing
uses sampled active intervals and sampling bounds. Uncertain timing interrupts
review with the original media retained, rather than authorizing a paid retake.
Legacy performance-v1 and exact/reference workspaces retain their original contract.
New result-bearing interaction entries use observed-entry-v5-bound-results:
continuation only asks for compatibility and an approach action; the program keeps
the fixed results rather than asking the model to echo them. Earlier continuation
protocols remain readable and retain their original fingerprints.
New v7 dense flexible plans retain the same minimum interaction total at film scope,
allowing per-shot weights to serve entry, progression and payoff. Weights express
preferences: the complete-film compiler allocates any missing interaction
milliseconds from available passive slack while preserving shots, actions and hold
floors. Prefix timings remain provisional; infeasible aggregate capacity still stops.
Planned totals and complete outcome evidence contracts are checked before generation. Before export,
raw observations bound to accepted media and contracts are recomputed over retained
edit windows; missing, uncertain or insufficient film evidence stops without retakes.
_Avoid_: Extra prompt keyword, entire shot duration

**Authored Action Unit**:
The event planner's private semantic input: one cause, its complete visible reaction,
one actor/target pair, and one effect description. Transition and hold units cannot
carry effects. The compiler derives execution phases, repeated geometry and exact
time windows before the existing Action Contract is validated. Relative weights
express pacing intent; compiled timestamps are planned constraints, not measured
motion or a promise of model adherence. Contradictory authoring stops without whole
storyboard rewrites. Legacy persisted contracts remain readable.

**Planned State**:
The intended visible state before or after a shot, fixed before generation and never rewritten to pretend that generated footage matched the plan.
Formal resume reads the frozen storyboard without manifest observations for planning
admission and scene binding. Execution restores accepted observations from the take
ledger in order; runtime progress must not invalidate the unchanged plan's review.
_Avoid_: Expected truth, canonical state

**Observed State**:
The visible state measured from one generated take, including uncertainty and deviations from the Planned State.
The v22 visual review protocol uses true/false/null observations: explicit unknown
pauses review and preserves the original take. Missing live identity/continuity
judgments cannot inherit legacy true defaults. Supplied references must exist and
identity images are bound to character IDs independently in generation and review.
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

**Paid Take Budget**:
The run-level authorization for new provider submissions. Reserving a new Take consumes authorization; polling an existing provider task does not.
New entrypoints freeze a completion-first policy: retries may spend only surplus
capacity after reserving one opportunity for each unsubmitted shot. Single-pass
allows at most one submitted Take per shot. Unknown submissions remain charged to
the local cap; released, definitely unsubmitted reservations do not. Old options
without a policy retain legacy behavior. This allocates opportunities, not success.
_Avoid_: Retry count, quality threshold

**Narrative Carrier**:
A concrete, filmable visual choice that expresses a Narrative Beat through readable behavior, coverage, a declared detail, or a visible endpoint.
_Avoid_: Mood, camera decoration

**Canonical Anchor**:
An accepted identity reference explicitly eligible to reset accumulated output-reference drift at a planned intentional cut.
_Avoid_: Previous tail, arbitrary keyframe

**Reference Chain Depth**:
The number of consecutive accepted output-tail handoffs since the last independent generation or Canonical Anchor.
_Avoid_: Shot number, retry count

# Execution projection audit (2026-11-01)

New complete flexible authored films persist `generation_rhythm=allocated-v1`.
The shared Action Contract now carries the already allocated interaction-time
union into normal and policy-safe generation as a pacing target. It is neither
a new per-shot gate nor an observation hint: whole-film admission still uses
actual independent video evidence. Missing markers preserve historical prompt
and contract fingerprints; exact/reference choreography is unchanged.
New scheduled camera choices come from the execution registry; unsupported raw
values stop at the response boundary without aliases, repairs or retries.
See the execution-projection section of `docs/adr/0001-single-action-contract-owner.md`.
This audit is offline only; no new hosted-model, video or upload authorization.
