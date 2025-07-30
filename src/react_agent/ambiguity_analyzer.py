"""Sophisticated ambiguity detection for lease analysis review.

This module provides advanced pattern-based detection of ambiguous contractual language,
unclear definitions, and provisions with multiple reasonable interpretations.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass

from react_agent.state import (
    ReviewFlag, 
    CrossReference, 
    ImpactLevel, 
    ConfidenceMetrics
)
from react_agent.configuration import Configuration


@dataclass
class AmbiguityPattern:
    """Represents a pattern for detecting contractual ambiguities."""
    
    pattern: str
    """Regex pattern to match ambiguous text."""
    
    flag_type: str
    """Type of ambiguity this pattern detects."""
    
    severity: ImpactLevel
    """Default severity level for matches."""
    
    description: str
    """Description of what this pattern detects."""
    
    recommendation: str
    """Standard recommendation for clarifying this ambiguity."""


class AmbiguityDetector:
    """Advanced ambiguity detection with pattern matching and clarity scoring."""
    
    def __init__(self, configuration: Configuration):
        """Initialize the ambiguity detector with configuration.
        
        Args:
            configuration: Reviewer configuration settings
        """
        self.configuration = configuration
        self.patterns = self._initialize_patterns()
        self.sensitivity_multiplier = self._get_sensitivity_multiplier()
    
    def _initialize_patterns(self) -> List[AmbiguityPattern]:
        """Initialize predefined patterns for ambiguity detection.
        
        Returns:
            List of AmbiguityPattern objects for different ambiguity types
        """
        return [
            # Vague Terminology Without Clear Criteria
            AmbiguityPattern(
                pattern=r'\b(?:reasonable|appropriate|satisfactory|adequate|substantial|material|significant)\b(?!\s+(?:as|is|means|shall be|defined))',
                flag_type='VAGUE_TERMINOLOGY',
                severity=ImpactLevel.MEDIUM,
                description='Vague terminology without clear definition or criteria',
                recommendation='Define specific criteria, thresholds, or objective standards for subjective terms'
            ),
            
            # Missing Key Definitions
            AmbiguityPattern(
                pattern=r'\b(?:substantial completion|material default|material change|material adverse effect|good standing|commercially reasonable efforts)\b',
                flag_type='MISSING_DEFINITION',
                severity=ImpactLevel.HIGH,
                description='Important terms referenced without clear definition',
                recommendation='Add specific definitions section or clause defining key terms'
            ),
            
            # Unclear Conditional Language
            AmbiguityPattern(
                pattern=r'\b(?:if|when|unless|provided that|subject to)\b.*\b(?:necessary|required|appropriate|reasonable|feasible)\b',
                flag_type='UNCLEAR_CONDITION',
                severity=ImpactLevel.MEDIUM,
                description='Conditional language with subjective or unclear triggers',
                recommendation='Specify objective conditions and clear triggering events'
            ),
            
            # Contradictory or Conflicting Language
            AmbiguityPattern(
                pattern=r'\b(?:notwithstanding|except|however|provided however)\b.*\b(?:above|foregoing|herein|previously)\b',
                flag_type='CONTRADICTORY_LANGUAGE',
                severity=ImpactLevel.HIGH,
                description='Language that may create conflicts or contradictions',
                recommendation='Review for internal conflicts and establish clear priority rules'
            ),
            
            # Imprecise Time References
            AmbiguityPattern(
                pattern=r'\b(?:promptly|immediately|as soon as|within a reasonable time|timely|without delay)\b',
                flag_type='IMPRECISE_TIMING',
                severity=ImpactLevel.MEDIUM,
                description='Imprecise time references that could lead to disputes',
                recommendation='Specify exact timeframes, deadlines, or calendar periods'
            ),
            
            # Vague Quantity or Measurement References
            AmbiguityPattern(
                pattern=r'\b(?:sufficient|adequate|appropriate|reasonable)(?:\s+(?:amount|quantity|number|size|area))\b',
                flag_type='VAGUE_QUANTITY',
                severity=ImpactLevel.MEDIUM,
                description='Vague quantity or measurement references without clear standards',
                recommendation='Provide specific numerical values, ranges, or measurement criteria'
            ),
            
            # Subjective Performance Standards
            AmbiguityPattern(
                pattern=r'\b(?:good faith|best efforts|commercially reasonable|industry standard|professional manner|workmanlike manner)\b',
                flag_type='SUBJECTIVE_STANDARD',
                severity=ImpactLevel.MEDIUM,
                description='Subjective performance standards without objective criteria',
                recommendation='Define specific performance metrics, benchmarks, or evaluation methods'
            ),
            
            # Unclear Scope or Extent
            AmbiguityPattern(
                pattern=r'\b(?:including but not limited to|such as|and other|similar|related|applicable)\b',
                flag_type='UNCLEAR_SCOPE',
                severity=ImpactLevel.LOW,
                description='Language that creates uncertainty about scope or extent',
                recommendation='Provide comprehensive lists or clear boundaries for scope'
            ),
            
            # Multiple Possible Interpretations
            AmbiguityPattern(
                pattern=r'\b(?:may|might|could|should|would)\s+(?:be deemed|be considered|include|apply)\b',
                flag_type='MULTIPLE_INTERPRETATIONS',
                severity=ImpactLevel.MEDIUM,
                description='Language that allows multiple reasonable interpretations',
                recommendation='Use definitive language and eliminate interpretative ambiguity'
            ),
            
            # Undefined Relationship or Hierarchy
            AmbiguityPattern(
                pattern=r'\b(?:in connection with|relating to|with respect to|concerning|regarding)\b.*\b(?:obligations|rights|responsibilities)\b',
                flag_type='UNDEFINED_RELATIONSHIP',
                severity=ImpactLevel.MEDIUM,
                description='Unclear relationships between obligations, rights, or responsibilities',
                recommendation='Clarify specific relationships and hierarchies between provisions'
            )
        ]
    
    def _get_sensitivity_multiplier(self) -> float:
        """Get sensitivity multiplier based on configuration.
        
        Returns:
            Multiplier for adjusting detection sensitivity
        """
        sensitivity = self.configuration.flag_sensitivity_ambiguity.lower()
        return {
            'conservative': 1.2,  # Flag more ambiguities
            'balanced': 1.0,      # Standard detection
            'aggressive': 0.8     # Flag fewer ambiguities
        }.get(sensitivity, 1.0)
    
    def detect_ambiguities(self, analysis_content: str) -> Tuple[List[ReviewFlag], ConfidenceMetrics]:
        """Detect ambiguities in lease analysis content.
        
        Args:
            analysis_content: The primary lease analysis text to review
            
        Returns:
            Tuple of (list of ReviewFlag objects, ConfidenceMetrics for the detection)
        """
        if not analysis_content:
            return [], ConfidenceMetrics(confidence_score=0.0)
        
        flags = []
        pattern_matches = []
        
        # Apply each pattern to the content
        for pattern_obj in self.patterns:
            matches = list(re.finditer(pattern_obj.pattern, analysis_content, re.IGNORECASE))
            
            for match in matches:
                # Calculate confidence score for this match
                confidence = self._calculate_match_confidence(match, analysis_content, pattern_obj)
                
                # Apply sensitivity threshold
                if confidence * self.sensitivity_multiplier >= 45.0:  # Slightly lower threshold for ambiguity
                    flag = self._create_flag_from_match(match, analysis_content, pattern_obj, confidence)
                    flags.append(flag)
                    pattern_matches.append((pattern_obj, match, confidence))
        
        # Remove duplicate flags based on overlapping text regions
        flags = self._deduplicate_flags(flags)
        
        # Apply configuration limits
        max_flags = self.configuration.max_flags_per_analysis
        if len(flags) > max_flags:
            # Sort by severity and confidence, keep top flags
            flags.sort(key=lambda f: (f.severity.value, -45.0), reverse=True)  # Simplified sorting
            flags = flags[:max_flags]
        
        # Calculate overall confidence metrics
        overall_confidence = self._calculate_overall_confidence(flags, pattern_matches, analysis_content)
        
        return flags, overall_confidence
    
    def _calculate_match_confidence(self, match: re.Match, content: str, pattern_obj: AmbiguityPattern) -> float:
        """Calculate confidence score for a pattern match.
        
        Args:
            match: The regex match object
            content: Full analysis content
            pattern_obj: The pattern that matched
            
        Returns:
            Confidence score (0-100)
        """
        base_confidence = 65.0  # Base confidence for ambiguity matches (slightly lower than legal)
        
        # Adjust based on match context
        matched_text = match.group(0)
        
        # Longer matches tend to be more significant
        length_factor = min(1.1, len(matched_text) / 40.0)
        
        # Adjust based on pattern type - ambiguity-specific scoring
        pattern_confidence = {
            'MISSING_DEFINITION': 80.0,
            'CONTRADICTORY_LANGUAGE': 75.0,
            'VAGUE_TERMINOLOGY': 70.0,
            'UNCLEAR_CONDITION': 68.0,
            'MULTIPLE_INTERPRETATIONS': 65.0,
            'SUBJECTIVE_STANDARD': 60.0,
            'IMPRECISE_TIMING': 58.0,
            'VAGUE_QUANTITY': 55.0,
            'UNDEFINED_RELATIONSHIP': 52.0,
            'UNCLEAR_SCOPE': 50.0
        }.get(pattern_obj.flag_type, base_confidence)
        
        # Calculate final confidence
        final_confidence = pattern_confidence * length_factor
        return min(90.0, max(25.0, final_confidence))  # Clamp between 25-90%
    
    def _create_flag_from_match(self, match: re.Match, content: str, pattern_obj: AmbiguityPattern, confidence: float) -> ReviewFlag:
        """Create a ReviewFlag object from a pattern match.
        
        Args:
            match: The regex match object
            content: Full analysis content
            pattern_obj: The pattern that matched
            confidence: Calculated confidence score
            
        Returns:
            ReviewFlag object with detailed information
        """
        matched_text = match.group(0)
        
        # Extract surrounding context
        start_pos = max(0, match.start() - 80)
        end_pos = min(len(content), match.end() + 80)
        context = content[start_pos:end_pos].strip()
        
        # Create cross-reference
        cross_ref = CrossReference(
            clause="Ambiguity Pattern Match",
            exact_quote=matched_text,
            document="Primary Analysis"
        )
        
        # Generate flag ID
        flag_id = f"AMBIGUITY_{pattern_obj.flag_type}_{match.start()}_{len(matched_text)}"
        
        # Create detailed recommendations
        recommendations = [
            pattern_obj.recommendation,
            f"Review context: '{context[:80]}...' for clarification opportunities",
            "Consider adding definitions or more specific language"
        ]
        
        # Determine if legal counsel is required (generally lower threshold than legal uncertainty)
        requires_counsel = pattern_obj.severity == ImpactLevel.HIGH
        
        return ReviewFlag(
            flag_id=flag_id,
            flag_type=pattern_obj.flag_type,
            severity=pattern_obj.severity,
            title=f"{pattern_obj.flag_type.replace('_', ' ').title()} Detected",
            description=f"{pattern_obj.description}\n\nAmbiguous text: '{matched_text}'\nContext: {context[:150]}...",
            cross_references=[cross_ref],
            recommendations=recommendations,
            requires_legal_counsel=requires_counsel
        )
    
    def _deduplicate_flags(self, flags: List[ReviewFlag]) -> List[ReviewFlag]:
        """Remove duplicate flags that cover overlapping text regions.
        
        Args:
            flags: List of flags to deduplicate
            
        Returns:
            Deduplicated list of flags
        """
        if len(flags) <= 1:
            return flags
        
        # Simple deduplication based on flag type and similar descriptions
        seen_combinations = set()
        deduplicated = []
        
        for flag in flags:
            # Create a key based on flag type and first 40 chars of description
            key = (flag.flag_type, flag.description[:40])
            
            if key not in seen_combinations:
                seen_combinations.add(key)
                deduplicated.append(flag)
        
        return deduplicated
    
    def _calculate_overall_confidence(self, flags: List[ReviewFlag], pattern_matches: List, content: str) -> ConfidenceMetrics:
        """Calculate overall confidence metrics for the ambiguity detection.
        
        Args:
            flags: List of generated flags
            pattern_matches: List of pattern matches with confidence scores
            content: Original analysis content
            
        Returns:
            ConfidenceMetrics object with overall assessment
        """
        if not flags:
            return ConfidenceMetrics(
                confidence_score=88.0,  # High confidence when no ambiguities found
                uncertainty_range={"lower": 82.0, "upper": 92.0},
                confidence_factors=["No contractual ambiguities detected", "Language clarity analysis completed"],
                uncertainty_sources=[]
            )
        
        # Calculate confidence based on number and severity of flags
        severity_weights = {
            ImpactLevel.CRITICAL: -18.0,
            ImpactLevel.HIGH: -12.0,
            ImpactLevel.MEDIUM: -8.0,
            ImpactLevel.LOW: -4.0,
            ImpactLevel.MINIMAL: -1.0
        }
        
        base_confidence = 75.0  # Slightly lower base for ambiguity detection
        for flag in flags:
            base_confidence += severity_weights.get(flag.severity, -8.0)
        
        # Ensure reasonable bounds
        final_confidence = max(25.0, min(80.0, base_confidence))
        
        # Calculate uncertainty range
        uncertainty_factor = self.configuration.uncertainty_range_factor
        lower_bound = max(15.0, final_confidence - (final_confidence * uncertainty_factor))
        upper_bound = min(90.0, final_confidence + (final_confidence * uncertainty_factor))
        
        # Generate confidence factors and uncertainty sources
        confidence_factors = [
            f"Analyzed {len(self.patterns)} ambiguity patterns",
            f"Processed {len(content)} characters of analysis content",
            "Language clarity assessment completed"
        ]
        
        uncertainty_sources = [
            f"{len(flags)} ambiguities detected",
            f"{len([f for f in flags if f.severity == ImpactLevel.HIGH])} high-impact ambiguities",
            f"{len([f for f in flags if f.requires_legal_counsel])} items may need clarification"
        ] if flags else []
        
        return ConfidenceMetrics(
            confidence_score=final_confidence,
            uncertainty_range={"lower": lower_bound, "upper": upper_bound},
            confidence_factors=confidence_factors,
            uncertainty_sources=uncertainty_sources
        )


def analyze_ambiguities(analysis_content: str, configuration: Configuration) -> Tuple[List[ReviewFlag], ConfidenceMetrics]:
    """Main entry point for ambiguity detection.
    
    Args:
        analysis_content: The primary lease analysis text to review
        configuration: Reviewer configuration settings
        
    Returns:
        Tuple of (list of ReviewFlag objects, ConfidenceMetrics for the analysis)
    """
    detector = AmbiguityDetector(configuration)
    return detector.detect_ambiguities(analysis_content) 