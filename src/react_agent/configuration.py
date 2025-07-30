"""Configuration for the ReAct agent."""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Optional

from langchain_core.runnables import RunnableConfig


class ReviewDepthLevel(str, Enum):
    """Available review depth levels."""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


class CalibrationStrategy(str, Enum):
    """Available confidence calibration strategies."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


@dataclass(kw_only=True)
class Configuration:
    """Configuration for the agent."""

    model: str = field(
        default="anthropic/claude-3-5-sonnet-20240620",
        metadata={
            "description": "Model to use for the agent. Should be in the format: provider/model-name."},
    )
    system_prompt: str = field(
        default="You are a helpful assistant.",
        metadata={"description": "System prompt for the agent."},
    )
    max_steps: int = field(
        default=25,
        metadata={
            "description": "Maximum number of steps to take in the agent's execution."},
    )

    # Reviewer Agent Configuration
    review_depth_level: str = field(
        default="standard",
        metadata={
            "description": "Controls the thoroughness of review analysis. Options: basic, standard, comprehensive."}
    )
    enable_legal_uncertainty_detection: bool = field(
        default=True,
        metadata={
            "description": "Enable detection and flagging of legal uncertainties."}
    )
    enable_ambiguity_detection: bool = field(
        default=True,
        metadata={
            "description": "Enable detection and flagging of contractual ambiguities."}
    )
    enable_onerous_clustering: bool = field(
        default=True,
        metadata={
            "description": "Enable clustering of onerous provisions for cumulative impact analysis."}
    )

    # Confidence Calibration Configuration
    confidence_threshold_high: float = field(
        default=80.0,
        metadata={
            "description": "Threshold percentage (0-100) for high confidence classification."}
    )
    confidence_threshold_medium: float = field(
        default=60.0,
        metadata={
            "description": "Threshold percentage (0-100) for medium confidence classification."}
    )
    uncertainty_range_factor: float = field(
        default=0.15,
        metadata={
            "description": "Factor (0.0-1.0) for calculating confidence uncertainty ranges."}
    )
    confidence_calibration_strategy: str = field(
        default="balanced",
        metadata={
            "description": "Strategy for calibrating confidence scores. Options: conservative, balanced, aggressive."}
    )

    # Flag Sensitivity Parameters
    flag_sensitivity_legal: str = field(
        default="medium",
        metadata={
            "description": "Sensitivity level for legal uncertainty detection. Options: low, medium, high."}
    )
    flag_sensitivity_ambiguity: str = field(
        default="medium",
        metadata={
            "description": "Sensitivity level for ambiguity detection. Options: low, medium, high."}
    )
    max_flags_per_analysis: int = field(
        default=20,
        metadata={
            "description": "Maximum number of flags to generate per analysis to prevent overwhelming output."}
    )

    # Performance and Processing Settings
    reviewer_timeout_seconds: int = field(
        default=120,
        metadata={
            "description": "Maximum time in seconds for reviewer agent processing."}
    )
    enable_parallel_analysis: bool = field(
        default=False,
        metadata={
            "description": "Enable parallel processing of multiple review analyses."}
    )
    max_terms_per_cluster: int = field(
        default=8,
        metadata={
            "description": "Maximum number of terms to include in a single onerous provision cluster."}
    )

    # Quality Control Settings
    minimum_confidence_for_recommendation: float = field(
        default=70.0,
        metadata={
            "description": "Minimum confidence percentage required before making recommendations."}
    )
    require_legal_review_threshold: float = field(
        default=75.0,
        metadata={
            "description": "Confidence threshold below which legal review is automatically recommended."}
    )

    # Advanced Review Features
    enable_confidence_calibration: bool = field(
        default=True,
        metadata={
            "description": "Enable sophisticated confidence calibration across multiple analyzers."}
    )
    enable_cross_validation: bool = field(
        default=False,
        metadata={
            "description": "Enable cross-validation between different review approaches."}
    )
    cluster_interaction_depth: int = field(
        default=2,
        metadata={
            "description": "Depth level (1-3) for analyzing provision cluster interactions."}
    )

    @classmethod
    def from_context(cls, context: Optional[RunnableConfig] = None) -> "Configuration":
        """Create a Configuration instance from the given context.

        Args:
            context: The context to get the configuration from.

        Returns:
            A Configuration instance.
        """
        if context and context.get("configurable"):
            return cls(**context["configurable"])
        return cls()

    def __post_init__(self):
        """Post-initialization validation of configuration values."""
        # Validate API keys
        required_vars = ["TAVILY_API_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if self.model.startswith("openai"):
            if not os.getenv("OPENAI_API_KEY"):
                missing_vars.append("OPENAI_API_KEY")
        elif self.model.startswith("anthropic"):
            if not os.getenv("ANTHROPIC_API_KEY"):
                missing_vars.append("ANTHROPIC_API_KEY")

        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}. Please check your .env file.")

        # Validate confidence thresholds
        if not (0.0 <= self.confidence_threshold_high <= 100.0):
            raise ValueError(
                "confidence_threshold_high must be between 0.0 and 100.0")
        if not (0.0 <= self.confidence_threshold_medium <= 100.0):
            raise ValueError(
                "confidence_threshold_medium must be between 0.0 and 100.0")
        if self.confidence_threshold_medium >= self.confidence_threshold_high:
            raise ValueError(
                "confidence_threshold_medium must be less than confidence_threshold_high")

        # Validate uncertainty range factor
        if not (0.0 <= self.uncertainty_range_factor <= 1.0):
            raise ValueError(
                "uncertainty_range_factor must be between 0.0 and 1.0")

        # Validate calibration strategy
        valid_strategies = [strategy.value for strategy in CalibrationStrategy]
        if self.confidence_calibration_strategy not in valid_strategies:
            raise ValueError(
                f"confidence_calibration_strategy must be one of: {', '.join(valid_strategies)}")

        # Validate review depth level
        valid_depths = [depth.value for depth in ReviewDepthLevel]
        if self.review_depth_level not in valid_depths:
            raise ValueError(
                f"review_depth_level must be one of: {', '.join(valid_depths)}")

        # Validate sensitivity levels
        valid_sensitivity = ["low", "medium", "high"]
        if self.flag_sensitivity_legal not in valid_sensitivity:
            raise ValueError(
                f"flag_sensitivity_legal must be one of: {', '.join(valid_sensitivity)}")
        if self.flag_sensitivity_ambiguity not in valid_sensitivity:
            raise ValueError(
                f"flag_sensitivity_ambiguity must be one of: {', '.join(valid_sensitivity)}")

        # Validate numeric ranges
        if self.max_flags_per_analysis < 1:
            raise ValueError("max_flags_per_analysis must be at least 1")
        if self.reviewer_timeout_seconds < 10:
            raise ValueError("reviewer_timeout_seconds must be at least 10")
        if self.max_terms_per_cluster < 2:
            raise ValueError("max_terms_per_cluster must be at least 2")
        if not (1 <= self.cluster_interaction_depth <= 3):
            raise ValueError(
                "cluster_interaction_depth must be between 1 and 3")

        # Validate confidence-related settings
        if not (0.0 <= self.minimum_confidence_for_recommendation <= 100.0):
            raise ValueError(
                "minimum_confidence_for_recommendation must be between 0.0 and 100.0")
        if not (0.0 <= self.require_legal_review_threshold <= 100.0):
            raise ValueError(
                "require_legal_review_threshold must be between 0.0 and 100.0")
