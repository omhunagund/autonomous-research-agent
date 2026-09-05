# Orchestrator Phase

Implemented the Critic-driven workflow control layer and Research correction path.

## Locked behavior implemented

- Research -> Analysis -> Writing -> Critic pipeline.
- Critic PASS finalizes immediately with **High** quality for a first-pass success.
- Critic failures route by category priority: **Research > Writing > Analysis**.
- Research correction groups all `[RESEARCH]` issues for the same exact current sub-question into one targeted-query planning call, producing 1-3 complementary queries.
- Critic Research issues must contain the exact current sub-question in quotes; matching uses normalized exact equality.
- Each correction query reuses the initial search contract: 5 candidates -> LLM selects top 3; 1-4 candidates -> use all; 0 -> record the retrieval limitation.
- Existing usable sources are preserved; new sources use global URL deduplication and continue citation IDs.
- Failed candidate URLs are not retried during the same correction pass.
- Research correction always flows through fresh Analysis -> Writing -> Critic reevaluation. The Orchestrator does not infer correction success from source counts.
- Current limitations are replaced for affected sub-questions and removed when at least three usable sources are currently available.
- Writing correction reruns Writing only, preserving Analysis findings/confidence/gaps/conflicts.
- Maximum of 2 Critic-driven correction cycles.
- Analysis-only Critic failures finalize Low quality and preserve `[ANALYSIS]` issues verbatim.
- After the correction limit is exhausted, the latest `[RESEARCH]` / `[WRITING]` issues are stored as unresolved issues and the current report finalizes Low.
- A correction that eventually passes finalizes Medium quality.

## Validation

The added test file covers exact issue mapping, grouped correction planning, the 1-3 query contract, validation retries, top-3 selection for five candidates, source preservation, and limitation removal.

The package was syntax-checked with `py_compile`. Full pytest execution was not possible in this packaging environment because the project's runtime dependencies (for example `langchain_core`) are not installed here; run the repository's normal test command inside the project's Python 3.11 virtual environment before committing.
