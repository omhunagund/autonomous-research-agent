# Orchestrator phase schema fix

Replace `src/models/schemas.py` in the project with the supplied file.

This restores the Writing Agent's internal structured draft schemas and the
Critic/Orchestrator FinalReport fields required by the current implementation:
- WrittenFindingDraft
- WrittenEvidenceDraft
- WrittenGapDraft
- WrittenConflictDraft
- InternalReportDraft
- gap_explanations
- conflict_explanations
- analysis_issues
- unresolved_issues
- nullable provisional `quality`
