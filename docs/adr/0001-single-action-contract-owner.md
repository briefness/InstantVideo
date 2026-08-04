# Keep one owner for action semantics

Planning, Seedance prompt compilation, semantic review, and retake diagnosis must compile from the Action Contract Module in `pipeline/causality.py`. We deepen the existing Module instead of adding a workflow framework because duplicated interpretations caused semantic drift, while a new DAG, multi-agent runtime, or provider abstraction would add a shallow interface without another real adapter.

## Consequences

Legacy storyboard fields remain accepted at the persistence seam, but downstream Modules may not independently redefine action phases or evidence fields. Planned State, Observed State, and Canonical State remain separate, and only an accepted Take may advance Canonical State.
