"""Sophisticated confidence calibration for lease analysis review.

This module provides advanced confidence calibration that combines multiple
analyzer confidence scores using configurable strategies, content complexity
analysis, and detailed confidence reasoning.
"""

import re
import math
import statistics
from typing import List, Dict, Any, Tuple, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from react_agent.state import ConfidenceMetrics
from react_agent.configuration import Configuration


class CalibrationStrategy(Enum):
    """Available confidence calibration strategies."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"  
    AGGRESSIVE = "aggressive"


@dataclass
class ContentComplexity:
    """Represents content complexity analysis results."""
    
    text_length_score: float = field(default=0.0)
    """Score based on text length (0.0-1.0)."""
    
    legal_terminology_density: float = field(default=0.0)
    """Density of legal terms in content (0.0-1.0)."""
    
    provision_interaction_complexity: float = field(default=0.0)
    """Complexity of provision interactions (0.0-1.0)."""
    
    cross_reference_complexity: float = field(default=0.0)
    """Complexity based on cross-references (0.0-1.0)."""
    
    overall_complexity: float = field(default=0.0)
    """Overall complexity score (0.0-1.0)."""


@dataclass
class AnalyzerAgreement:
    """Represents agreement analysis between multiple analyzers."""
    
    confidence_variance: float = field(default=0.0)
    """Variance in confidence scores (0.0+)."""
    
    agreement_score: float = field(default=0.0)
    """Agreement level between analyzers (0.0-1.0)."""
    
    outlier_count: int = field(default=0)
    """Number of outlier confidence scores."""
    
    consensus_bonus: float = field(default=0.0)
    """Bonus for high agreement (-0.2 to +0.2)."""


@dataclass
class CalibrationDetails:
    """Detailed breakdown of confidence calibration process."""
    
    strategy_used: CalibrationStrategy = field(default=CalibrationStrategy.BALANCED)
    """Calibration strategy applied."""
    
    input_confidences: List[float] = field(default_factory=list)
    """Original confidence scores from analyzers."""
    
    content_complexity: ContentComplexity = field(default_factory=ContentComplexity)
    """Content complexity analysis results."""
    
    analyzer_agreement: AnalyzerAgreement = field(default_factory=AnalyzerAgreement)
    """Analyzer agreement analysis."""
    
    base_confidence: float = field(default=0.0)
    """Base confidence before calibration."""
    
    complexity_adjustment: float = field(default=0.0)
    """Adjustment based on content complexity."""
    
    agreement_adjustment: float = field(default=0.0)
    """Adjustment based on analyzer agreement."""
    
    final_confidence: float = field(default=0.0)
    """Final calibrated confidence score."""
    
    calibration_reasoning: List[str] = field(default_factory=list)
    """Step-by-step calibration reasoning."""


class ConfidenceCalibrator:
    """Advanced confidence calibration with multiple strategies and complexity analysis."""
    
    def __init__(self, configuration: Configuration):
        """Initialize the confidence calibrator with configuration.
        
        Args:
            configuration: Reviewer configuration settings
        """
        self.configuration = configuration
        self.strategy = self._determine_strategy()
        self.legal_terms = self._initialize_legal_terms()
        
    def _determine_strategy(self) -> CalibrationStrategy:
        """Determine calibration strategy from configuration.
        
        Returns:
            CalibrationStrategy enum value
        """
        strategy_name = getattr(self.configuration, 'confidence_calibration_strategy', 'balanced')
        
        try:
            return CalibrationStrategy(strategy_name.lower())
        except ValueError:
            return CalibrationStrategy.BALANCED
    
    def _initialize_legal_terms(self) -> List[str]:
        """Initialize list of legal terms for complexity analysis.
        
        Returns:
            List of legal terms and phrases
        """
        return [
            # Contract terminology
            'whereas', 'notwithstanding', 'hereinafter', 'heretofore', 'aforementioned',
            'covenant', 'indemnify', 'warranty', 'guarantee', 'breach', 'default',
            
            # Real estate legal terms
            'lessor', 'lessee', 'premises', 'demised', 'appurtenant', 'easement',
            'subletting', 'assignment', 'alienation', 'encumbrance', 'lien',
            
            # Financial/rent terms
            'escalation', 'abatement', 'proration', 'holdover', 'percentage rent',
            'base year', 'expense stop', 'cam charges', 'operating expenses',
            
            # Legal process terms
            'arbitration', 'mediation', 'jurisdiction', 'venue', 'force majeure',
            'condemnation', 'eminent domain', 'compliance', 'zoning', 'permits',
            
            # Performance terms
            'substantial completion', 'tenant improvements', 'work letter',
            'punch list', 'certificate of occupancy', 'commencement date'
        ]
    
    def calibrate_confidence(self, 
                           confidence_metrics: List[ConfidenceMetrics], 
                           analysis_content: str = "") -> Tuple[ConfidenceMetrics, CalibrationDetails]:
        """Calibrate confidence using sophisticated analysis and strategy.
        
        Args:
            confidence_metrics: List of confidence metrics from different analyzers
            analysis_content: Original analysis content for complexity assessment
            
        Returns:
            Tuple of (calibrated ConfidenceMetrics, CalibrationDetails)
        """
        if not confidence_metrics:
            # Return default confidence with explanation
            default_confidence = ConfidenceMetrics(
                confidence_score=50.0,
                uncertainty_range={"lower": 25.0, "upper": 75.0},
                confidence_factors=["No analyzer results available"],
                uncertainty_sources=["Missing analysis input"]
            )
            
            default_details = CalibrationDetails(
                final_confidence=50.0,
                calibration_reasoning=["Default confidence applied - no analyzer results"]
            )
            
            return default_confidence, default_details
        
        # Extract confidence scores
        input_scores = [cm.confidence_score for cm in confidence_metrics]
        
        # Step 1: Analyze content complexity
        complexity = self._analyze_content_complexity(analysis_content)
        
        # Step 2: Analyze analyzer agreement
        agreement = self._analyze_analyzer_agreement(input_scores)
        
        # Step 3: Calculate base confidence using strategy
        base_confidence = self._calculate_base_confidence(input_scores, self.strategy)
        
        # Step 4: Apply complexity and agreement adjustments
        complexity_adj = self._calculate_complexity_adjustment(complexity, base_confidence)
        agreement_adj = self._calculate_agreement_adjustment(agreement, base_confidence)
        
        # Step 5: Calculate final calibrated confidence
        final_confidence = self._apply_calibration_bounds(
            base_confidence + complexity_adj + agreement_adj
        )
        
        # Step 6: Generate calibrated confidence metrics
        calibrated_metrics = self._generate_calibrated_metrics(
            final_confidence, confidence_metrics, complexity, agreement
        )
        
        # Step 7: Create detailed calibration breakdown
        calibration_details = CalibrationDetails(
            strategy_used=self.strategy,
            input_confidences=input_scores,
            content_complexity=complexity,
            analyzer_agreement=agreement,
            base_confidence=base_confidence,
            complexity_adjustment=complexity_adj,
            agreement_adjustment=agreement_adj,
            final_confidence=final_confidence,
            calibration_reasoning=self._generate_calibration_reasoning(
                base_confidence, complexity_adj, agreement_adj, final_confidence
            )
        )
        
        return calibrated_metrics, calibration_details
    
    def _analyze_content_complexity(self, content: str) -> ContentComplexity:
        """Analyze content complexity factors.
        
        Args:
            content: Analysis content to evaluate
            
        Returns:
            ContentComplexity object with analysis results
        """
        if not content:
            return ContentComplexity()
        
        # Text length complexity (longer content = higher complexity)
        length_score = min(1.0, len(content) / 5000.0)  # Normalize to typical lease length
        
        # Legal terminology density
        legal_term_count = sum(1 for term in self.legal_terms if term.lower() in content.lower())
        legal_density = min(1.0, legal_term_count / max(1, len(content.split()) / 100))
        
        # Provision interaction complexity (based on cross-references)
        cross_ref_patterns = [
            r'\bclause\s+\d+',
            r'\bsection\s+\d+',
            r'\bparagraph\s+\d+',
            r'\bsee\s+clause',
            r'\bas\s+defined\s+in',
            r'\bsubject\s+to\s+clause'
        ]
        
        cross_ref_count = sum(len(re.findall(pattern, content, re.IGNORECASE)) 
                             for pattern in cross_ref_patterns)
        cross_ref_complexity = min(1.0, cross_ref_count / max(1, len(content.split()) / 200))
        
        # Provision interaction complexity (based on conditional language)
        conditional_patterns = [
            r'\bprovided\s+that',
            r'\bsubject\s+to',
            r'\bunless\s+otherwise',
            r'\bexcept\s+as',
            r'\bnotwithstanding',
            r'\bin\s+the\s+event'
        ]
        
        conditional_count = sum(len(re.findall(pattern, content, re.IGNORECASE)) 
                               for pattern in conditional_patterns)
        interaction_complexity = min(1.0, conditional_count / max(1, len(content.split()) / 150))
        
        # Calculate overall complexity
        overall = statistics.mean([length_score, legal_density, interaction_complexity, cross_ref_complexity])
        
        return ContentComplexity(
            text_length_score=length_score,
            legal_terminology_density=legal_density,
            provision_interaction_complexity=interaction_complexity,
            cross_reference_complexity=cross_ref_complexity,
            overall_complexity=overall
        )
    
    def _analyze_analyzer_agreement(self, confidence_scores: List[float]) -> AnalyzerAgreement:
        """Analyze agreement between analyzer confidence scores.
        
        Args:
            confidence_scores: List of confidence scores to analyze
            
        Returns:
            AnalyzerAgreement object with analysis results
        """
        if len(confidence_scores) < 2:
            return AnalyzerAgreement(agreement_score=1.0)  # Perfect agreement with single score
        
        # Calculate variance
        variance = statistics.variance(confidence_scores)
        
        # Calculate agreement score (inverse of normalized variance)
        max_possible_variance = 50.0 ** 2  # Max variance when scores are 0 and 100
        agreement_score = max(0.0, 1.0 - (variance / max_possible_variance))
        
        # Detect outliers (scores more than 1.5 * IQR from median)
        sorted_scores = sorted(confidence_scores)
        q1 = statistics.median(sorted_scores[:len(sorted_scores)//2])
        q3 = statistics.median(sorted_scores[len(sorted_scores)//2:])
        iqr = q3 - q1
        
        outlier_count = 0
        if iqr > 0:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_count = sum(1 for score in confidence_scores 
                               if score < lower_bound or score > upper_bound)
        
        # Calculate consensus bonus/penalty
        if agreement_score > 0.8:
            consensus_bonus = 0.1 * agreement_score  # Bonus for high agreement
        elif agreement_score < 0.3:
            consensus_bonus = -0.15 * (1 - agreement_score)  # Penalty for disagreement
        else:
            consensus_bonus = 0.0
        
        return AnalyzerAgreement(
            confidence_variance=variance,
            agreement_score=agreement_score,
            outlier_count=outlier_count,
            consensus_bonus=consensus_bonus
        )
    
    def _calculate_base_confidence(self, scores: List[float], strategy: CalibrationStrategy) -> float:
        """Calculate base confidence using specified strategy.
        
        Args:
            scores: List of confidence scores
            strategy: Calibration strategy to use
            
        Returns:
            Base confidence score
        """
        if not scores:
            return 50.0
        
        if strategy == CalibrationStrategy.CONSERVATIVE:
            # Use minimum score with slight upward adjustment
            base = min(scores)
            if len(scores) > 1:
                base += (statistics.mean(scores) - base) * 0.2
            return base
            
        elif strategy == CalibrationStrategy.AGGRESSIVE:
            # Use maximum score with slight downward adjustment  
            base = max(scores)
            if len(scores) > 1:
                base -= (base - statistics.mean(scores)) * 0.2
            return base
            
        else:  # BALANCED
            # Use weighted average favoring median
            if len(scores) == 1:
                return scores[0]
            
            median_score = statistics.median(scores)
            mean_score = statistics.mean(scores)
            
            # Weight: 60% median, 40% mean
            return 0.6 * median_score + 0.4 * mean_score
    
    def _calculate_complexity_adjustment(self, complexity: ContentComplexity, base_confidence: float) -> float:
        """Calculate confidence adjustment based on content complexity.
        
        Args:
            complexity: Content complexity analysis
            base_confidence: Base confidence score
            
        Returns:
            Adjustment value to apply to confidence
        """
        # Higher complexity should generally reduce confidence
        complexity_factor = complexity.overall_complexity
        
        # Adjustment scale: 0-20% reduction for high complexity
        max_adjustment = base_confidence * 0.2
        adjustment = -complexity_factor * max_adjustment
        
        # Special considerations for very high legal density
        if complexity.legal_terminology_density > 0.8:
            adjustment -= 5.0  # Additional penalty for very dense legal content
            
        # Special considerations for high cross-reference complexity
        if complexity.cross_reference_complexity > 0.7:
            adjustment -= 3.0  # Additional penalty for complex cross-references
        
        return max(-20.0, adjustment)  # Cap maximum negative adjustment
    
    def _calculate_agreement_adjustment(self, agreement: AnalyzerAgreement, base_confidence: float) -> float:
        """Calculate confidence adjustment based on analyzer agreement.
        
        Args:
            agreement: Analyzer agreement analysis
            base_confidence: Base confidence score
            
        Returns:
            Adjustment value to apply to confidence
        """
        # Start with consensus bonus/penalty
        adjustment = agreement.consensus_bonus * base_confidence
        
        # Additional penalty for outliers
        if agreement.outlier_count > 0:
            outlier_penalty = agreement.outlier_count * 2.0
            adjustment -= outlier_penalty
        
        # High variance penalty (beyond what consensus already covers)
        if agreement.confidence_variance > 400:  # 20-point standard deviation
            variance_penalty = min(10.0, (agreement.confidence_variance - 400) / 100)
            adjustment -= variance_penalty
        
        return max(-15.0, min(10.0, adjustment))  # Cap adjustment range
    
    def _apply_calibration_bounds(self, confidence: float) -> float:
        """Apply bounds to keep calibrated confidence in reasonable range.
        
        Args:
            confidence: Raw calibrated confidence
            
        Returns:
            Bounded confidence score
        """
        return max(15.0, min(95.0, confidence))
    
    def _generate_calibrated_metrics(self, 
                                   final_confidence: float,
                                   original_metrics: List[ConfidenceMetrics],
                                   complexity: ContentComplexity,
                                   agreement: AnalyzerAgreement) -> ConfidenceMetrics:
        """Generate calibrated confidence metrics object.
        
        Args:
            final_confidence: Final calibrated confidence score
            original_metrics: Original confidence metrics from analyzers
            complexity: Content complexity analysis
            agreement: Analyzer agreement analysis
            
        Returns:
            Calibrated ConfidenceMetrics object
        """
        # Calculate uncertainty range based on strategy and agreement
        base_uncertainty = self.configuration.uncertainty_range_factor
        
        # Adjust uncertainty based on agreement and complexity
        if agreement.agreement_score < 0.5:
            uncertainty_multiplier = 1.5  # Higher uncertainty for disagreement
        elif agreement.agreement_score > 0.8:
            uncertainty_multiplier = 0.8  # Lower uncertainty for high agreement
        else:
            uncertainty_multiplier = 1.0
        
        # Adjust for complexity
        if complexity.overall_complexity > 0.7:
            uncertainty_multiplier *= 1.3
        
        adjusted_uncertainty = base_uncertainty * uncertainty_multiplier
        
        # Calculate bounds
        uncertainty_range = final_confidence * adjusted_uncertainty
        lower_bound = max(10.0, final_confidence - uncertainty_range)
        upper_bound = min(98.0, final_confidence + uncertainty_range)
        
        # Combine confidence factors from all analyzers
        all_factors = []
        for metrics in original_metrics:
            all_factors.extend(metrics.confidence_factors)
        
        # Add calibration-specific factors
        calibration_factors = [
            f"Calibrated using {self.strategy.value} strategy",
            f"Content complexity: {complexity.overall_complexity:.2f}",
            f"Analyzer agreement: {agreement.agreement_score:.2f}"
        ]
        
        # Combine uncertainty sources
        all_sources = []
        for metrics in original_metrics:
            all_sources.extend(metrics.uncertainty_sources)
        
        # Add calibration-specific sources
        if complexity.overall_complexity > 0.6:
            all_sources.append("High content complexity detected")
        if agreement.agreement_score < 0.6:
            all_sources.append("Analyzer disagreement identified")
        if agreement.outlier_count > 0:
            all_sources.append(f"{agreement.outlier_count} outlier confidence scores")
        
        return ConfidenceMetrics(
            confidence_score=final_confidence,
            uncertainty_range={"lower": lower_bound, "upper": upper_bound},
            confidence_factors=all_factors + calibration_factors,
            uncertainty_sources=list(set(all_sources))  # Remove duplicates
        )
    
    def _generate_calibration_reasoning(self, 
                                      base_confidence: float,
                                      complexity_adj: float, 
                                      agreement_adj: float,
                                      final_confidence: float) -> List[str]:
        """Generate step-by-step calibration reasoning.
        
        Args:
            base_confidence: Base confidence before adjustments
            complexity_adj: Complexity adjustment applied
            agreement_adj: Agreement adjustment applied
            final_confidence: Final calibrated confidence
            
        Returns:
            List of reasoning steps
        """
        reasoning = [
            f"Applied {self.strategy.value} strategy for base confidence: {base_confidence:.1f}%"
        ]
        
        if abs(complexity_adj) > 0.5:
            if complexity_adj < 0:
                reasoning.append(f"Reduced confidence by {abs(complexity_adj):.1f}% due to content complexity")
            else:
                reasoning.append(f"Increased confidence by {complexity_adj:.1f}% due to low complexity")
        
        if abs(agreement_adj) > 0.5:
            if agreement_adj < 0:
                reasoning.append(f"Reduced confidence by {abs(agreement_adj):.1f}% due to analyzer disagreement")
            else:
                reasoning.append(f"Increased confidence by {agreement_adj:.1f}% due to analyzer consensus")
        
        reasoning.append(f"Final calibrated confidence: {final_confidence:.1f}%")
        
        return reasoning


def calibrate_confidence(confidence_metrics: List[ConfidenceMetrics], 
                        analysis_content: str, 
                        configuration: Configuration) -> Tuple[ConfidenceMetrics, CalibrationDetails]:
    """Main entry point for sophisticated confidence calibration.
    
    Args:
        confidence_metrics: List of confidence metrics from different analyzers
        analysis_content: Original analysis content for complexity assessment
        configuration: Reviewer configuration settings
        
    Returns:
        Tuple of (calibrated ConfidenceMetrics, CalibrationDetails for explanation)
    """
    calibrator = ConfidenceCalibrator(configuration)
    return calibrator.calibrate_confidence(confidence_metrics, analysis_content) 