# Writing Agent phase

Files:
- `src/models/schemas.py`
- `src/agents/writing_agent.py`
- `tests/test_writing_agent.py`

Implementation notes:
- `FinalReport.quality` is optional during the provisional Writing stage and is
  intentionally `None`. The Critic/Orchestrator must assign the final quality.
- Citation parsing accepts only `[n]` / `[n][m]`; comma-form citations such as
  `[1, 3]` are rejected.
- Writer drafts preserve Analysis Finding/Gap/Conflict structure and order.
- `references` are deterministically derived from report-wide cited source IDs.
- The Writer performs no quality assignment and cannot repair unsupported
  Analysis Findings.
