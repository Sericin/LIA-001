"""Sophisticated legal uncertainty detection for lease analysis review.

This module provides advanced pattern-based detection of legal uncertainties,
ambiguities, and potential issues that may require legal counsel.
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
class LegalPattern:
    """Represents a pattern for detecting legal uncertainties."""
    
    pattern: str
    """Regex pattern to match problematic text."""
    
    flag_type: str
    """Type of legal uncertainty this pattern detects."""
    
    severity: ImpactLevel
    """Default severity level for matches."""
    
    description: str
    """Description of what this pattern detects."""
    
    recommendation: str
    """Standard recommendation for this type of uncertainty."""


class LegalUncertaintyDetector:
    """Advanced legal uncertainty detection with pattern matching and confidence scoring."""
    
    def __init__(self, configuration: Configuration):
        """Initialize the legal uncertainty detector with configuration.
        
        Args:
            configuration: Reviewer configuration settings
        """
        self.configuration = configuration
        self.patterns = self._initialize_patterns()
        self.sensitivity_multiplier = self._get_sensitivity_multiplier()
    
    def _initialize_patterns(self) -> List[LegalPattern]:
        """Initialize predefined patterns for legal uncertainty detection.
        
        Returns:
            List of LegalPattern objects for different uncertainty types
        """
        return [
            # External Law References
            LegalPattern(
                pattern=r'\b(?:statute|regulation|ordinance|code|law|act)\b.*\b(?:requires?|mandates?|prohibits?|governs?)\b',
                flag_type='EXTERNAL_LAW_REFERENCE',
                severity=ImpactLevel.HIGH,
                description='Reference to external law that may affect interpretation',
                recommendation='Consult legal counsel to verify current law and compliance requirements'
            ),
            
            # Conditional Language with Unclear Triggers
            LegalPattern(
                pattern=r'\b(?:if|when|unless|provided that|subject to)\b.*\b(?:reasonabl[ey]|appropriat[ely]|satisfactor[ily]|adequat[ely])\b',
                flag_type='VAGUE_CONDITIONAL',
                severity=ImpactLevel.MEDIUM,
                description='Conditional language with subjective or undefined standards',
                recommendation='Define objective standards or criteria for conditional terms'
            ),
            
            # Undefined Key Terms
            LegalPattern(
                pattern=r'\b(?:reasonable|appropriate|satisfactory|adequate|substantial|material|significant)\b.*\b(?:time|notice|manner|condition|change|improvement)\b',
                flag_type='UNDEFINED_STANDARDS',
                severity=ImpactLevel.MEDIUM,
                description='Subjective standards without clear definition',
                recommendation='Request specific definitions or objective criteria for subjective terms'
            ),
            
            # Conflicting or Contradictory Language
            LegalPattern(
                pattern=r'\b(?:notwithstanding|except|however|but|nevertheless)\b.*\b(?:foregoing|above|preceding)\b',
                flag_type='POTENTIAL_CONFLICT',
                severity=ImpactLevel.HIGH,
                description='Language that may contradict other lease provisions',
                recommendation='Review entire document for conflicts and seek clarification on priority'
            ),
            
            # Unusual or Non-Standard Provisions
            LegalPattern(
                pattern=r'\b(?:in perpetuity|forever|indefinitely|without limit|absolute|unconditional)\b',
                flag_type='UNUSUAL_PROVISION',
                severity=ImpactLevel.HIGH,
                description='Unusual or potentially problematic absolute language',
                recommendation='Consider whether absolute terms are necessary and enforceable'
            ),
            
            # Vague Performance Standards
            LegalPattern(
                pattern=r'\b(?:best efforts|reasonable efforts|commercially reasonable|industry standard|customary)\b',
                flag_type='VAGUE_PERFORMANCE_STANDARD',
                severity=ImpactLevel.MEDIUM,
                description='Performance standards that lack specific criteria',
                recommendation='Define specific performance metrics and evaluation criteria'
            ),
            
            # References to "Market" Without Definition
            LegalPattern(
                pattern=r'\bmarket\s+(?:rate|rent|value|terms|conditions)\b(?!\s+(?:as|means|shall|is|defined))',
                flag_type='UNDEFINED_MARKET_REFERENCE',
                severity=ImpactLevel.MEDIUM,
                description='Market-based terms without clear definition or methodology',
                recommendation='Specify market determination methodology and comparable properties'
            ),
            
            # Force Majeure or Casualty Language
            LegalPattern(
                pattern=r'\b(?:force majeure|act of god|casualty|destruction|fire|flood|earthquake)\b',
                flag_type='CASUALTY_PROVISION',
                severity=ImpactLevel.MEDIUM,
                description='Force majeure or casualty provisions requiring careful review',
                recommendation='Review scope and allocation of risks for casualty events'
            ),
            
            # Sole Discretion Language
            LegalPattern(
                pattern=r'\b(?:sole|absolute|complete|full|unfettered)\s+discretion\b',
                flag_type='SOLE_DISCRETION',
                severity=ImpactLevel.HIGH,
                description='Unilateral discretionary power without objective standards',
                recommendation='Request objective criteria or mutual agreement for discretionary decisions'
            ),
            
            # "Regardless of" Risk Shifting
            LegalPattern(
                pattern=r'\bregardless\s+of\s+(?:cause|condition|age|fault|negligence|source|timing)\b',
                flag_type='RISK_SHIFTING',
                severity=ImpactLevel.HIGH,
                description='Broad risk allocation regardless of fault or causation',
                recommendation='Limit risk allocation to appropriate circumstances and causes'
            ),
            
            # "For Any Reason or No Reason" Language
            LegalPattern(
                pattern=r'\bfor\s+any\s+reason\s+or\s+no\s+reason\b',
                flag_type='UNLIMITED_DISCRETION',
                severity=ImpactLevel.CRITICAL,
                description='Unlimited discretionary power without any standards',
                recommendation='Establish reasonable standards and limitations on discretionary power'
            ),
            
            # Immediate Termination/Action Language
            LegalPattern(
                pattern=r'\b(?:immediately\s+upon|without\s+notice|without\s+cure|instant(?:ly)?)\b',
                flag_type='IMMEDIATE_ACTION',
                severity=ImpactLevel.HIGH,
                description='Provisions allowing immediate action without notice or cure period',
                recommendation='Negotiate reasonable notice periods and opportunities to cure'
            ),
            
            # Waiver of Claims/Rights
            LegalPattern(
                pattern=r'\b(?:waives?|waiver\s+of)\s+(?:all\s+)?(?:claims?|rights?|remedies?)\b',
                flag_type='RIGHTS_WAIVER',
                severity=ImpactLevel.HIGH,
                description='Broad waiver of legal rights or claims',
                recommendation='Limit waivers to specific, reasonable circumstances'
            ),
            
            # "At Tenant's Expense" Language
            LegalPattern(
                pattern=r'\bat\s+(?:tenant\'?s?|lessee\'?s?)\s+(?:sole\s+)?expense\b',
                flag_type='TENANT_EXPENSE_BURDEN',
                severity=ImpactLevel.MEDIUM,
                description='Cost allocation placing financial burden on tenant',
                recommendation='Review whether cost allocation is reasonable and customary'
            ),
            
            # Penalty or Liquidated Damages
            LegalPattern(
                pattern=r'\b(?:penalty|liquidated\s+damages?|additional\s+rent|premium)\b.*\b(?:\d+%|percent|times|multiple)\b|plus\s+\d+%\s+penalty|\d+%\s+penalty',
                flag_type='PENALTY_PROVISIONS',
                severity=ImpactLevel.HIGH,
                description='Penalty or liquidated damages provisions',
                recommendation='Ensure penalties are reasonable and reflect actual anticipated damages'
            ),
            
            # Environmental Liability
            LegalPattern(
                pattern=r'\b(?:environmental|hazardous|toxic|contamination)\b.*\b(?:liable?|responsible|shall\s+pay)\b',
                flag_type='ENVIRONMENTAL_LIABILITY',
                severity=ImpactLevel.HIGH,
                description='Environmental liability provisions',
                recommendation='Limit environmental liability to tenant-caused contamination'
            )
        ]
    
    def _get_sensitivity_multiplier(self) -> float:
        """Get sensitivity multiplier based on configuration.
        
        Returns:
            Multiplier for adjusting detection sensitivity
        """
        sensitivity = self.configuration.flag_sensitivity_legal.lower()
        return {
            'conservative': 1.3,  # Flag more uncertainties
            'balanced': 1.0,      # Standard detection
            'aggressive': 0.7     # Flag fewer uncertainties
        }.get(sensitivity, 1.0)
    
    def detect_legal_uncertainties(self, analysis_content: str) -> Tuple[List[ReviewFlag], ConfidenceMetrics]:
        """Detect legal uncertainties in lease analysis content.
        
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
                if confidence * self.sensitivity_multiplier >= 40.0:  # Minimum threshold
                    flag = self._create_flag_from_match(match, analysis_content, pattern_obj, confidence)
                    flags.append(flag)
                    pattern_matches.append((pattern_obj, match, confidence))
        
        # Remove duplicate flags based on overlapping text regions
        flags = self._deduplicate_flags(flags)
        
        # Apply configuration limits
        max_flags = self.configuration.max_flags_per_analysis
        if len(flags) > max_flags:
            # Sort by severity and confidence, keep top flags
            flags.sort(key=lambda f: (f.severity.value, -50.0), reverse=True)  # Simplified sorting
            flags = flags[:max_flags]
        
        # Calculate overall confidence metrics
        overall_confidence = self._calculate_overall_confidence(flags, pattern_matches, analysis_content)
        
        return flags, overall_confidence
    
    def _calculate_match_confidence(self, match: re.Match, content: str, pattern_obj: LegalPattern) -> float:
        """Calculate confidence score for a pattern match.
        
        Args:
            match: The regex match object
            content: Full analysis content
            pattern_obj: The pattern that matched
            
        Returns:
            Confidence score (0-100)
        """
        base_confidence = 70.0  # Base confidence for pattern matches
        
        # Adjust based on match context
        matched_text = match.group(0)
        
        # Longer matches tend to be more significant
        length_factor = min(1.2, len(matched_text) / 50.0)
        
        # Adjust based on pattern type
        pattern_confidence = {
            'EXTERNAL_LAW_REFERENCE': 85.0,
            'POTENTIAL_CONFLICT': 80.0,
            'UNUSUAL_PROVISION': 75.0,
            'VAGUE_CONDITIONAL': 70.0,
            'UNDEFINED_STANDARDS': 65.0,
            'VAGUE_PERFORMANCE_STANDARD': 60.0,
            'UNDEFINED_MARKET_REFERENCE': 70.0,
            'CASUALTY_PROVISION': 55.0,
            # New patterns with high confidence scores
            'SOLE_DISCRETION': 90.0,
            'RISK_SHIFTING': 85.0,
            'UNLIMITED_DISCRETION': 95.0,
            'IMMEDIATE_ACTION': 85.0,
            'RIGHTS_WAIVER': 80.0,
            'TENANT_EXPENSE_BURDEN': 75.0,
            'PENALTY_PROVISIONS': 85.0,
            'ENVIRONMENTAL_LIABILITY': 80.0
        }.get(pattern_obj.flag_type, base_confidence)
        
        # Calculate final confidence
        final_confidence = pattern_confidence * length_factor
        return min(95.0, max(30.0, final_confidence))  # Clamp between 30-95%
    
    def _create_flag_from_match(self, match: re.Match, content: str, pattern_obj: LegalPattern, confidence: float) -> ReviewFlag:
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
        start_pos = max(0, match.start() - 100)
        end_pos = min(len(content), match.end() + 100)
        context = content[start_pos:end_pos].strip()
        
        # Create cross-reference
        cross_ref = CrossReference(
            clause="Pattern Match",
            exact_quote=matched_text,
            document="Primary Analysis"
        )
        
        # Generate flag ID
        flag_id = f"LEGAL_{pattern_obj.flag_type}_{match.start()}_{len(matched_text)}"
        
        # Create detailed recommendations
        recommendations = [
            pattern_obj.recommendation,
            f"Review context: '{context[:100]}...' if needed"
        ]
        
        # Determine if legal counsel is required
        requires_counsel = pattern_obj.severity in [ImpactLevel.CRITICAL, ImpactLevel.HIGH]
        
        return ReviewFlag(
            flag_id=flag_id,
            flag_type=pattern_obj.flag_type,
            severity=pattern_obj.severity,
            title=f"{pattern_obj.flag_type.replace('_', ' ').title()} Detected",
            description=f"{pattern_obj.description}\n\nMatched text: '{matched_text}'\nContext: {context[:200]}...",
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
            # Create a key based on flag type and first 50 chars of description
            key = (flag.flag_type, flag.description[:50])
            
            if key not in seen_combinations:
                seen_combinations.add(key)
                deduplicated.append(flag)
        
        return deduplicated
    
    def _calculate_overall_confidence(self, flags: List[ReviewFlag], pattern_matches: List, content: str) -> ConfidenceMetrics:
        """Calculate overall confidence metrics for the legal uncertainty detection.
        
        Args:
            flags: List of generated flags
            pattern_matches: List of pattern matches with confidence scores
            content: Original analysis content
            
        Returns:
            ConfidenceMetrics object with overall assessment
        """
        if not flags:
            return ConfidenceMetrics(
                confidence_score=90.0,  # High confidence when no issues found
                uncertainty_range={"lower": 85.0, "upper": 95.0},
                confidence_factors=["No legal uncertainties detected", "Pattern analysis completed"],
                uncertainty_sources=[]
            )
        
        # Calculate confidence based on number and severity of flags
        severity_weights = {
            ImpactLevel.CRITICAL: -20.0,
            ImpactLevel.HIGH: -15.0,
            ImpactLevel.MEDIUM: -10.0,
            ImpactLevel.LOW: -5.0,
            ImpactLevel.MINIMAL: -2.0
        }
        
        base_confidence = 80.0
        for flag in flags:
            base_confidence += severity_weights.get(flag.severity, -10.0)
        
        # Ensure reasonable bounds
        final_confidence = max(20.0, min(85.0, base_confidence))
        
        # Calculate uncertainty range
        uncertainty_factor = self.configuration.uncertainty_range_factor
        lower_bound = max(10.0, final_confidence - (final_confidence * uncertainty_factor))
        upper_bound = min(95.0, final_confidence + (final_confidence * uncertainty_factor))
        
        # Generate confidence factors and uncertainty sources
        confidence_factors = [
            f"Analyzed {len(self.patterns)} legal uncertainty patterns",
            f"Processed {len(content)} characters of analysis content"
        ]
        
        uncertainty_sources = [
            f"{len(flags)} legal uncertainties detected",
            f"{len([f for f in flags if f.requires_legal_counsel])} flags require legal counsel"
        ] if flags else []
        
        return ConfidenceMetrics(
            confidence_score=final_confidence,
            uncertainty_range={"lower": lower_bound, "upper": upper_bound},
            confidence_factors=confidence_factors,
            uncertainty_sources=uncertainty_sources
        )


def analyze_legal_uncertainties(analysis_content: str, configuration: Configuration) -> Tuple[List[ReviewFlag], ConfidenceMetrics]:
    """Main entry point for legal uncertainty detection.
    
    Args:
        analysis_content: The primary lease analysis text to review
        configuration: Reviewer configuration settings
        
    Returns:
        Tuple of (list of ReviewFlag objects, ConfidenceMetrics for the analysis)
    """
    detector = LegalUncertaintyDetector(configuration)
    return detector.detect_legal_uncertainties(analysis_content) 