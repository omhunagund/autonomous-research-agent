from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CritiqueCheck(str, Enum):
    PASS = "pass"
    FAIL = "fail"


class QualityLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class GapType(str, Enum):
    RESEARCH_GAP = "research_gap"
    EVIDENCE_GAP = "evidence_gap"


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


class SubQuestionPlan(BaseModel):
    sub_questions: list[str]


class SearchSelection(BaseModel):
    selected_indices: list[int]


class Source(BaseModel):
    citation_id: int
    title: str
    url: str
    retrieved_at: datetime
    snippet: str
    search_queries: list[str]
    content: str


class Finding(BaseModel):
    claim: str
    supporting_sources: list[int]
    confidence: ConfidenceLevel


class Conflict(BaseModel):
    description: str
    related_sources: list[int]


class Gap(BaseModel):
    type: GapType
    description: str
    related_sub_question: str | None = None
    related_claim: str | None = None


class AgreementLevel(str, Enum):
    AGREE = "agree"
    PARTIAL = "partial"
    UNCLEAR = "unclear"


class InternalFindingAssessment(BaseModel):
    claim: str
    supporting_sources: list[int]
    independent_sources: list[int]
    agreement: AgreementLevel
    conflict: bool


class InternalAnalysis(BaseModel):
    finding_assessments: list[InternalFindingAssessment]
    conflicts: list[Conflict]
    gaps: list[Gap]


class Analysis(BaseModel):
    findings: list[Finding]
    conflicts: list[Conflict]
    gaps: list[Gap]


class WrittenFindingDraft(BaseModel):
    text: str
    citation_ids: list[int]


class WrittenEvidenceDraft(BaseModel):
    text: str
    citation_ids: list[int]
    related_finding_indices: list[int]


class WrittenGapDraft(BaseModel):
    text: str


class WrittenConflictDraft(BaseModel):
    text: str


class InternalReportDraft(BaseModel):
    executive_summary: str
    finding_drafts: list[WrittenFindingDraft]
    evidence_drafts: list[WrittenEvidenceDraft]
    gap_drafts: list[WrittenGapDraft]
    conflict_drafts: list[WrittenConflictDraft]
    cited_source_ids: list[int]


class Critique(BaseModel):
    faithfulness: CritiqueCheck
    coverage: CritiqueCheck
    recency: CritiqueCheck
    balance: CritiqueCheck
    verdict: CritiqueCheck
    issues: list[str]


class FinalReport(BaseModel):
    topic: str
    executive_summary: str
    key_findings: list[Finding]
    supporting_evidence: list[str]
    gaps: list[Gap]
    gap_explanations: list[str]
    conflicts: list[Conflict]
    conflict_explanations: list[str]
    quality: QualityLevel | None = None
    references: list[Source]


class StoredReport(BaseModel):
    report_id: str
    topic: str
    created_at: datetime
    quality: QualityLevel
    report: FinalReport


class MemoryMatch(BaseModel):
    report_id: str
    topic: str
    key_findings: list[Finding]
    gaps: list[Gap]
    conflicts: list[Conflict]


class MemoryContext(BaseModel):
    matches: list[MemoryMatch]


class ResearchLimitation(BaseModel):
    sub_question: str
    usable_source_count: int
    failed_candidate_count: int
    description: str


class ResearchState(BaseModel):
    user_topic: str
    memory_context: MemoryContext | None
    sub_questions: list[str]
    sources: list[Source]
    findings: list[Finding]
    conflicts: list[Conflict]
    gaps: list[Gap]
    research_limitations: list[ResearchLimitation]
    draft: FinalReport | None
    critique: Critique | None
    retry_count: int
    revision_target: str | None
    final_report: FinalReport | None
