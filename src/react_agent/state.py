"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated


class ReviewStatus(Enum):
    """Status values for review workflow states."""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ImpactLevel(Enum):
    """Impact levels for severity and importance assessment."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    MINIMAL = "MINIMAL"


class ImpactDirection(Enum):
    """Direction of impact on rental valuation."""
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"


class MarketPosition(Enum):
    """Market position indicating stakeholder advantage."""
    TENANT_FAVORABLE = "TENANT_FAVORABLE"
    LANDLORD_FAVORABLE = "LANDLORD_FAVORABLE"
    MARKET_STANDARD = "MARKET_STANDARD"
    BALANCED = "BALANCED"


@dataclass
class CrossReference:
    """Standard structure for citations and references in lease documents.

    Used for tracking exact sources of lease provisions and legal language.
    """

    clause: str = field(default="")
    """Document section reference (e.g., "14.2", "Schedule A.3")."""

    exact_quote: str = field(default="")
    """Verbatim text from the source document."""

    document: Optional[str] = field(default=None)
    """Source document identifier when multiple documents are involved."""

    page_number: Optional[int] = field(default=None)
    """Page reference for physical document location."""


@dataclass
class ValidationItem:
    """Individual validation check with status and description.

    Used in quality validation checklists throughout the review process.
    """

    check_name: str = field(default="")
    """Brief description of the validation check."""

    status: bool = field(default=False)
    """Pass/fail status of the validation check."""

    description: str = field(default="")
    """Detailed description or reasoning for the validation result."""


@dataclass
class QualityValidation:
    """Standard structure for quality checks and validation tracking.

    Provides comprehensive validation framework for review processes.
    """

    validation_items: List[ValidationItem] = field(default_factory=list)
    """List of individual validation checks performed."""

    overall_quality_score: float = field(default=0.0)
    """Aggregate quality score from 0-100."""


@dataclass
class ConfidenceMetrics:
    """Confidence scoring and uncertainty quantification.

    Provides graduated confidence assessment instead of binary confidence.
    """

    confidence_score: float = field(default=0.0)
    """Confidence level as percentage (0-100)."""

    uncertainty_range: Dict[str, float] = field(default_factory=dict)
    """Uncertainty bounds with 'lower' and 'upper' keys."""

    confidence_factors: List[str] = field(default_factory=list)
    """Factors contributing to confidence assessment."""

    uncertainty_sources: List[str] = field(default_factory=list)
    """Sources of uncertainty affecting confidence."""


@dataclass
class ReviewFlag:
    """Individual flag for legal uncertainties, ambiguities, or issues.

    Represents specific issues that require attention or legal counsel.
    """

    flag_id: str = field(default="")
    """Unique identifier for the flag."""

    flag_type: str = field(default="")
    """Type of flag (e.g., 'LEGAL_UNCERTAINTY', 'AMBIGUITY', 'ONEROUS_PROVISION')."""

    severity: ImpactLevel = field(default=ImpactLevel.MEDIUM)
    """Severity level of the flagged issue."""

    title: str = field(default="")
    """Brief title describing the flagged issue."""

    description: str = field(default="")
    """Detailed description of the issue and reasoning."""

    cross_references: List[CrossReference] = field(default_factory=list)
    """References to relevant lease clauses and documents."""

    recommendations: List[str] = field(default_factory=list)
    """Recommended actions or next steps."""

    requires_legal_counsel: bool = field(default=False)
    """Whether this flag requires legal expert review."""


@dataclass
class LeaseTermAnalysis:
    """Analysis of individual lease terms and provisions.

    Captures detailed assessment of lease provisions and their valuation impact.
    """

    term_name: str = field(default="")
    """Name or description of the lease term."""

    cross_references: List[CrossReference] = field(default_factory=list)
    """References to lease clauses and exact language."""

    impact_direction: ImpactDirection = field(default=ImpactDirection.NEUTRAL)
    """Direction of impact on rental valuation."""

    impact_magnitude: ImpactLevel = field(default=ImpactLevel.MEDIUM)
    """Magnitude of impact on rental valuation."""

    market_position: MarketPosition = field(
        default=MarketPosition.MARKET_STANDARD)
    """Position relative to market standards."""

    analysis_notes: str = field(default="")
    """Detailed analysis and reasoning."""

    related_terms: List[str] = field(default_factory=list)
    """Other lease terms that interact with this provision."""


@dataclass
class OneroousProvisionCluster:
    """Cluster of interconnected onerous provisions.

    Represents groups of lease terms that cumulatively impact rental value.
    """

    cluster_id: str = field(default="")
    """Unique identifier for the cluster."""

    cluster_name: str = field(default="")
    """Descriptive name for the cluster."""

    primary_terms: List[str] = field(default_factory=list)
    """Main lease terms in the cluster."""

    supporting_terms: List[str] = field(default_factory=list)
    """Supporting or related terms."""

    cumulative_impact: ImpactLevel = field(default=ImpactLevel.MEDIUM)
    """Combined impact of all terms in cluster."""

    interaction_description: str = field(default="")
    """Description of how terms interact."""

    flags: List[ReviewFlag] = field(default_factory=list)
    """Flags raised for terms in this cluster."""


@dataclass
class ReviewerFindings:
    """Complete findings from the REVIEWER AGENT analysis.

    Comprehensive results of the review process including all assessments.
    """

    review_status: ReviewStatus = field(default=ReviewStatus.PENDING)
    """Current status of the review process."""

    review_timestamp: Optional[datetime] = field(default=None)
    """When the review was completed."""

    confidence_metrics: ConfidenceMetrics = field(
        default_factory=ConfidenceMetrics)
    """Overall confidence assessment for the analysis."""

    flags: List[ReviewFlag] = field(default_factory=list)
    """All flags raised during review."""

    lease_term_analyses: List[LeaseTermAnalysis] = field(default_factory=list)
    """Detailed analysis of individual lease terms."""

    onerous_clusters: List[OneroousProvisionCluster] = field(
        default_factory=list)
    """Identified clusters of interconnected onerous provisions."""

    quality_validation: QualityValidation = field(
        default_factory=QualityValidation)
    """Quality validation results."""

    recommendations: List[str] = field(default_factory=list)
    """High-level recommendations from the review."""

    requires_legal_review: bool = field(default=False)
    """Whether any findings require legal expert consultation."""


@dataclass
class AnalysisComparison:
    """Comparison between primary analysis and reviewer findings.

    Tracks differences, agreements, and discrepancies between analyses.
    """

    primary_confidence: float = field(default=0.0)
    """Confidence score from primary analysis."""

    reviewer_confidence: float = field(default=0.0)
    """Confidence score from reviewer analysis."""

    confidence_delta: float = field(default=0.0)
    """Difference between primary and reviewer confidence."""

    agreement_areas: List[str] = field(default_factory=list)
    """Areas where primary and reviewer analyses agree."""

    discrepancy_areas: List[str] = field(default_factory=list)
    """Areas where analyses disagree."""

    new_findings: List[str] = field(default_factory=list)
    """Issues identified by reviewer but missed in primary analysis."""

    escalation_required: bool = field(default=False)
    """Whether discrepancies require escalation or further review."""


@dataclass
class ReviewMetrics:
    """Metrics and statistics for the review process.

    Tracks counts, scores, and performance indicators for the review.
    """

    flags_raised_count: int = field(default=0)
    """Total number of flags raised."""

    high_severity_flags_count: int = field(default=0)
    """Number of high or critical severity flags."""

    legal_counsel_required_count: int = field(default=0)
    """Number of flags requiring legal counsel."""

    terms_analyzed_count: int = field(default=0)
    """Total number of lease terms analyzed."""

    clusters_identified_count: int = field(default=0)
    """Number of onerous provision clusters identified."""

    review_depth_score: float = field(default=0.0)
    """Score indicating thoroughness of review (0-100)."""

    processing_time_seconds: Optional[float] = field(default=None)
    """Time taken to complete the review."""


@dataclass
class InputState:
    """Defines the input state for the agent, representing a narrower interface to the outside world.

    This class is used to define the initial state and structure of incoming data.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )
    """
    Messages tracking the primary execution state of the agent.

    Typically accumulates a pattern of:
    1. HumanMessage - user input
    2. AIMessage with .tool_calls - agent picking tool(s) to use to collect information
    3. ToolMessage(s) - the responses (or errors) from the executed tools
    4. AIMessage without .tool_calls - agent responding in unstructured format to the user
    5. HumanMessage - user responds with the next conversational turn

    Steps 2-5 may repeat as needed.

    The `add_messages` annotation ensures that new messages are merged with existing ones,
    updating by ID to maintain an "append-only" state unless a message with the same ID is provided.
    """


@dataclass
class State(InputState):
    """Represents the complete state of the agent, extending InputState with additional attributes.

    This class can be used to store any information needed throughout the agent's lifecycle.
    """

    is_last_step: IsLastStep = field(default=False)
    """
    Indicates whether the current step is the last one before the graph raises an error.

    This is a 'managed' variable, controlled by the state machine rather than user code.
    It is set to 'True' when the step count reaches recursion_limit - 1.
    """

    # REVIEWER AGENT STATE EXTENSIONS
    # These fields support the enhanced review and validation workflow

    primary_analysis_results: Dict[str, Any] = field(default_factory=dict)
    """
    Results from the primary lease analysis agent.

    Stores the initial analysis findings before review validation.
    Structure matches the primary analysis output format.
    """

    reviewer_findings: ReviewerFindings = field(
        default_factory=ReviewerFindings)
    """
    Complete findings and assessments from the REVIEWER AGENT.

    Includes confidence metrics, flags, detailed term analyses,
    onerous provision clusters, and quality validation results.
    """

    analysis_comparison: AnalysisComparison = field(
        default_factory=AnalysisComparison)
    """
    Comparison between primary analysis and reviewer findings.

    Tracks agreements, discrepancies, confidence deltas, and
    escalation requirements for quality assurance.
    """

    review_metrics: ReviewMetrics = field(default_factory=ReviewMetrics)
    """
    Performance metrics and statistics for the review process.

    Includes counts of flags, severity levels, processing times,
    and review depth scoring for monitoring and optimization.
    """

    review_configuration: Dict[str, Any] = field(default_factory=dict)
    """
    Configuration settings specific to the review process.

    Includes review depth settings, confidence thresholds,
    flag sensitivity parameters, and other reviewer-specific options.
    """

    # Additional attributes can be added here as needed.
    # Common examples include:
    # retrieved_documents: List[Document] = field(default_factory=list)
    # extracted_entities: Dict[str, Any] = field(default_factory=dict)
    # api_connections: Dict[str, Any] = field(default_factory=dict)
