"""Sophisticated onerous provision clustering for lease analysis review.

This module provides advanced clustering of interconnected lease provisions
that cumulatively impact rental value through relationship mapping and
cumulative effect analysis.
"""

import re
from typing import List, Dict, Any, Tuple, Optional, Set
from datetime import datetime
from dataclasses import dataclass

from react_agent.state import (
    OneroousProvisionCluster,
    ReviewFlag, 
    CrossReference, 
    ImpactLevel, 
    ConfidenceMetrics
)
from react_agent.configuration import Configuration


@dataclass
class ProvisionPattern:
    """Represents a pattern for detecting specific types of lease provisions."""
    
    pattern: str
    """Regex pattern to match provision text."""
    
    provision_type: str
    """Type of provision this pattern detects."""
    
    impact_weight: float
    """Base impact weight for this provision type (0.0-1.0)."""
    
    description: str
    """Description of what this provision type represents."""
    
    interaction_multiplier: Dict[str, float]
    """Multipliers for interactions with other provision types."""


@dataclass
class ProvisionMatch:
    """Represents a matched provision with its metadata."""
    
    provision_type: str
    """Type of the matched provision."""
    
    matched_text: str
    """The actual text that matched."""
    
    start_pos: int
    """Start position in the content."""
    
    end_pos: int
    """End position in the content."""
    
    impact_weight: float
    """Impact weight for this provision."""
    
    context: str
    """Surrounding context."""


class ClusteringAnalyzer:
    """Advanced onerous provision clustering with relationship mapping and cumulative impact analysis."""
    
    def __init__(self, configuration: Configuration):
        """Initialize the clustering analyzer with configuration.
        
        Args:
            configuration: Reviewer configuration settings
        """
        self.configuration = configuration
        self.patterns = self._initialize_patterns()
        self.max_cluster_size = configuration.max_terms_per_cluster
        self.cluster_depth = configuration.cluster_interaction_depth
    
    def _initialize_patterns(self) -> List[ProvisionPattern]:
        """Initialize predefined patterns for provision detection.
        
        Returns:
            List of ProvisionPattern objects for different provision types
        """
        return [
            # Rent and Financial Obligations
            ProvisionPattern(
                pattern=r'\b(?:base\s+rent|minimum\s+rent|additional\s+rent|percentage\s+rent|escalation|rent\s+increase|CAM\s+charges|operating\s+expenses|real\s+estate\s+taxes|insurance|capital\s+improvements|increases.*discretion|market\s+adjustments)\b',
                provision_type='RENT_OBLIGATIONS',
                impact_weight=0.8,
                description='Rent and financial payment obligations',
                interaction_multiplier={
                    'USE_RESTRICTIONS': 1.3,
                    'COMPLIANCE_OBLIGATIONS': 1.2,
                    'MAINTENANCE_REQUIREMENTS': 1.2
                }
            ),
            
            # Use and Activity Restrictions
            ProvisionPattern(
                pattern=r'\b(?:permitted use|prohibited|exclusive|non-compete|hours of operation|noise restrictions|signage restrictions)\b',
                provision_type='USE_RESTRICTIONS',
                impact_weight=0.7,
                description='Use restrictions and operational limitations',
                interaction_multiplier={
                    'RENT_OBLIGATIONS': 1.3,
                    'COMPLIANCE_OBLIGATIONS': 1.4,
                    'OPERATIONAL_RESTRICTIONS': 1.5
                }
            ),
            
            # Maintenance and Repair Obligations
            ProvisionPattern(
                pattern=r'\b(?:tenant.*(?:maintenance|repair)|repair.*(?:obligations|responsibilities)|HVAC.*(?:maintenance|repair)|structural.*(?:repairs?|maintenance)|common\s+area\s+maintenance|utilities|maintenance.*repairs?|responsible.*(?:ALL|maintenance|repairs?)|plumbing|electrical|roof|exterior\s+walls|parking\s+areas|landscaping|building\s+systems)\b',
                provision_type='MAINTENANCE_REQUIREMENTS',
                impact_weight=0.6,
                description='Maintenance and repair responsibilities',
                interaction_multiplier={
                    'RENT_OBLIGATIONS': 1.2,
                    'COMPLIANCE_OBLIGATIONS': 1.3,
                    'INSURANCE_REQUIREMENTS': 1.2
                }
            ),
            
            # Legal and Regulatory Compliance
            ProvisionPattern(
                pattern=r'\b(?:compliance\s+with.*(?:laws?|regulations?|ordinances?)|permits?\s+required|zoning\s+compliance|ADA\s+compliance|environmental.*(?:regulations?|liability|contamination)|health\s+codes?|responsible.*compliance|bring.*into\s+compliance|applicable\s+codes)\b',
                provision_type='COMPLIANCE_OBLIGATIONS',
                impact_weight=0.9,
                description='Legal and regulatory compliance requirements',
                interaction_multiplier={
                    'USE_RESTRICTIONS': 1.4,
                    'RENT_OBLIGATIONS': 1.2,
                    'MAINTENANCE_REQUIREMENTS': 1.3
                }
            ),
            
            # Insurance and Risk Management
            ProvisionPattern(
                pattern=r'\b(?:insurance requirements|liability coverage|property insurance|workers compensation|indemnification|hold harmless)\b',
                provision_type='INSURANCE_REQUIREMENTS',
                impact_weight=0.5,
                description='Insurance and liability obligations',
                interaction_multiplier={
                    'MAINTENANCE_REQUIREMENTS': 1.2,
                    'COMPLIANCE_OBLIGATIONS': 1.1,
                    'OPERATIONAL_RESTRICTIONS': 1.1
                }
            ),
            
            # Operational and Performance Restrictions
            ProvisionPattern(
                pattern=r'\b(?:business hours|customer limits|parking restrictions|delivery restrictions|waste disposal|security requirements)\b',
                provision_type='OPERATIONAL_RESTRICTIONS',
                impact_weight=0.6,
                description='Operational and performance limitations',
                interaction_multiplier={
                    'USE_RESTRICTIONS': 1.5,
                    'COMPLIANCE_OBLIGATIONS': 1.2,
                    'MAINTENANCE_REQUIREMENTS': 1.1
                }
            ),
            
            # Default and Termination Provisions
            ProvisionPattern(
                pattern=r'\b(?:default.*(?:immediate|without\s+notice|terminate)|rent.*(?:48\s+hours|late|delay)|penalty.*(?:25%|percent)|remaining.*rent.*(?:term|penalty)|material\s+default|cure\s+period)\b',
                provision_type='DEFAULT_TERMINATION',
                impact_weight=0.9,
                description='Default and termination provisions',
                interaction_multiplier={
                    'RENT_OBLIGATIONS': 1.4,
                    'ASSIGNMENT_RESTRICTIONS': 1.3,
                    'CONTROL_PROVISIONS': 1.3
                }
            ),
            
            # Assignment and Subletting Restrictions
            ProvisionPattern(
                pattern=r'\b(?:assignment.*(?:restrictions?|prohibited|consent)|sublet.*(?:restrictions?|prohibited|consent)|transfer.*restrictions?|consent.*(?:withheld|discretion)|absolute\s+discretion|any\s+reason\s+or\s+no\s+reason|landlord.*approval|written\s+consent)\b',
                provision_type='ASSIGNMENT_RESTRICTIONS',
                impact_weight=0.7,
                description='Assignment and transfer limitations',
                interaction_multiplier={
                    'USE_RESTRICTIONS': 1.2,
                    'DEFAULT_TERMINATION': 1.3,
                    'CONTROL_PROVISIONS': 1.4
                }
            ),
            
            # Landlord Control and Discretion
            ProvisionPattern(
                pattern=r'\b(?:landlord.*(?:discretion|sole\s+discretion|absolute\s+discretion|deems\s+necessary|determines?)|waives?\s+all\s+claims|tenant.*expense|improvements.*landlord.*property|without\s+(?:notice|liability)|regardless\s+of)\b',
                provision_type='CONTROL_PROVISIONS',
                impact_weight=0.8,
                description='Landlord control and discretionary provisions',
                interaction_multiplier={
                    'RENT_OBLIGATIONS': 1.3,
                    'ASSIGNMENT_RESTRICTIONS': 1.4,
                    'DEFAULT_TERMINATION': 1.3,
                    'MAINTENANCE_REQUIREMENTS': 1.2
                }
            ),
            
            # Financial Guarantees and Security
            ProvisionPattern(
                pattern=r'\b(?:security deposit|letter of credit|guaranty|financial statements|credit requirements|personal guarantee)\b',
                provision_type='FINANCIAL_GUARANTEES',
                impact_weight=0.8,
                description='Financial security and guarantee requirements',
                interaction_multiplier={
                    'RENT_OBLIGATIONS': 1.2,
                    'ASSIGNMENT_RESTRICTIONS': 1.3,
                    'COMPLIANCE_OBLIGATIONS': 1.1
                }
            )
        ]
    
    def analyze_clusters(self, analysis_content: str) -> Tuple[List[OneroousProvisionCluster], ConfidenceMetrics]:
        """Analyze lease content for onerous provision clusters.
        
        Args:
            analysis_content: The primary lease analysis text to review
            
        Returns:
            Tuple of (list of OneroousProvisionCluster objects, ConfidenceMetrics for the analysis)
        """
        if not analysis_content:
            return [], ConfidenceMetrics(confidence_score=0.0)
        
        # Step 1: Identify all provisions
        provision_matches = self._identify_provisions(analysis_content)
        
        if not provision_matches:
            return [], ConfidenceMetrics(
                confidence_score=85.0,
                uncertainty_range={"lower": 80.0, "upper": 90.0},
                confidence_factors=["No onerous provisions detected", "Clustering analysis completed"],
                uncertainty_sources=[]
            )
        
        # Step 2: Group provisions into clusters based on relationships
        clusters = self._create_clusters(provision_matches)
        
        # Step 3: Calculate cumulative impacts for each cluster
        enhanced_clusters = self._calculate_cumulative_impacts(clusters, analysis_content)
        
        # Step 4: Filter and limit clusters based on configuration
        filtered_clusters = self._filter_clusters(enhanced_clusters)
        
        # Step 5: Calculate overall confidence metrics
        confidence_metrics = self._calculate_clustering_confidence(filtered_clusters, provision_matches, analysis_content)
        
        return filtered_clusters, confidence_metrics
    
    def _identify_provisions(self, content: str) -> List[ProvisionMatch]:
        """Identify all provisions in the content using pattern matching.
        
        Args:
            content: Text content to analyze
            
        Returns:
            List of ProvisionMatch objects
        """
        matches = []
        
        for pattern_obj in self.patterns:
            regex_matches = list(re.finditer(pattern_obj.pattern, content, re.IGNORECASE))
            
            for match in regex_matches:
                # Extract context around the match
                start_context = max(0, match.start() - 60)
                end_context = min(len(content), match.end() + 60)
                context = content[start_context:end_context].strip()
                
                provision_match = ProvisionMatch(
                    provision_type=pattern_obj.provision_type,
                    matched_text=match.group(0),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    impact_weight=pattern_obj.impact_weight,
                    context=context
                )
                matches.append(provision_match)
        
        return matches
    
    def _create_clusters(self, provision_matches: List[ProvisionMatch]) -> List[List[ProvisionMatch]]:
        """Group provisions into clusters based on proximity and type relationships.
        
        Args:
            provision_matches: List of identified provisions
            
        Returns:
            List of clusters, where each cluster is a list of related provisions
        """
        if not provision_matches:
            return []
        
        # Sort by position for proximity analysis
        sorted_matches = sorted(provision_matches, key=lambda x: x.start_pos)
        
        clusters = []
        current_cluster = [sorted_matches[0]]
        
        for i in range(1, len(sorted_matches)):
            current_match = sorted_matches[i]
            last_match = current_cluster[-1]
            
            # Check if provisions should be in the same cluster
            should_cluster = self._should_cluster_provisions(current_match, last_match, current_cluster)
            
            if should_cluster and len(current_cluster) < self.max_cluster_size:
                current_cluster.append(current_match)
            else:
                # Start new cluster if current one has multiple provisions
                if len(current_cluster) > 1:
                    clusters.append(current_cluster)
                current_cluster = [current_match]
        
        # Add the last cluster if it has multiple provisions
        if len(current_cluster) > 1:
            clusters.append(current_cluster)
        
        return clusters
    
    def _should_cluster_provisions(self, match1: ProvisionMatch, match2: ProvisionMatch, current_cluster: List[ProvisionMatch]) -> bool:
        """Determine if two provisions should be clustered together.
        
        Args:
            match1: First provision
            match2: Second provision  
            current_cluster: Current cluster being built
            
        Returns:
            Boolean indicating if provisions should be clustered
        """
        # Check proximity (within reasonable distance)
        distance = abs(match1.start_pos - match2.start_pos)
        proximity_threshold = 300  # characters
        
        if distance > proximity_threshold:
            return False
        
        # Check type interaction strength
        pattern1 = next((p for p in self.patterns if p.provision_type == match1.provision_type), None)
        if not pattern1:
            return False
        
        interaction_multiplier = pattern1.interaction_multiplier.get(match2.provision_type, 0.0)
        
        # Cluster if there's a meaningful interaction
        return interaction_multiplier > 1.0
    
    def _calculate_cumulative_impacts(self, clusters: List[List[ProvisionMatch]], content: str) -> List[OneroousProvisionCluster]:
        """Calculate cumulative impacts for each cluster and create cluster objects.
        
        Args:
            clusters: List of provision clusters
            content: Original content for cross-references
            
        Returns:
            List of OneroousProvisionCluster objects with impact calculations
        """
        enhanced_clusters = []
        
        for i, cluster in enumerate(clusters):
            if len(cluster) < 2:
                continue  # Skip single-provision clusters
            
            # Calculate base impact
            base_impact = sum(match.impact_weight for match in cluster) / len(cluster)
            
            # Calculate interaction multipliers
            interaction_bonus = self._calculate_interaction_bonus(cluster)
            
            # Determine cumulative impact level
            cumulative_score = base_impact * (1 + interaction_bonus)
            cumulative_impact = self._score_to_impact_level(cumulative_score)
            
            # Generate cluster description
            cluster_description = self._generate_cluster_description(cluster)
            
            # Create cross-references for the cluster
            cross_refs = []
            for match in cluster:
                cross_ref = CrossReference(
                    clause="Provision Cluster",
                    exact_quote=match.matched_text,
                    document="Primary Analysis"
                )
                cross_refs.append(cross_ref)
            
            # Create cluster object
            cluster_obj = OneroousProvisionCluster(
                cluster_id=f"CLUSTER_{i+1}_{len(cluster)}_PROVISIONS",
                cluster_name=f"Cluster {i+1}: {cluster_description['title']}",
                primary_terms=[match.provision_type for match in cluster],
                supporting_terms=[match.matched_text for match in cluster],
                cumulative_impact=cumulative_impact,
                interaction_description=cluster_description['description'],
                flags=[]  # Flags would be populated from other analyzers if needed
            )
            
            enhanced_clusters.append(cluster_obj)
        
        return enhanced_clusters
    
    def _calculate_interaction_bonus(self, cluster: List[ProvisionMatch]) -> float:
        """Calculate interaction bonus based on provision relationships.
        
        Args:
            cluster: List of provisions in the cluster
            
        Returns:
            Interaction bonus multiplier (0.0-1.0)
        """
        total_bonus = 0.0
        comparison_count = 0
        
        for i, match1 in enumerate(cluster):
            for j, match2 in enumerate(cluster[i+1:], i+1):
                pattern1 = next((p for p in self.patterns if p.provision_type == match1.provision_type), None)
                if pattern1:
                    multiplier = pattern1.interaction_multiplier.get(match2.provision_type, 0.0)
                    if multiplier > 1.0:
                        total_bonus += (multiplier - 1.0)
                        comparison_count += 1
        
        return total_bonus / max(1, comparison_count)
    
    def _score_to_impact_level(self, score: float) -> ImpactLevel:
        """Convert cumulative score to impact level.
        
        Args:
            score: Cumulative impact score
            
        Returns:
            ImpactLevel enum value
        """
        if score >= 1.5:
            return ImpactLevel.CRITICAL
        elif score >= 1.2:
            return ImpactLevel.HIGH
        elif score >= 0.8:
            return ImpactLevel.MEDIUM
        elif score >= 0.4:
            return ImpactLevel.LOW
        else:
            return ImpactLevel.MINIMAL
    
    def _generate_cluster_description(self, cluster: List[ProvisionMatch]) -> Dict[str, str]:
        """Generate descriptive information for a cluster.
        
        Args:
            cluster: List of provisions in the cluster
            
        Returns:
            Dictionary with title and description
        """
        provision_types = [match.provision_type for match in cluster]
        unique_types = list(set(provision_types))
        
        # Generate title based on provision types
        if len(unique_types) == 1:
            title = f"{unique_types[0].replace('_', ' ').title()} Concentration"
        else:
            title = f"Mixed {'/'.join([t.replace('_', ' ').title() for t in unique_types[:2]])} Cluster"
        
        # Generate description
        description = f"Cluster of {len(cluster)} interconnected provisions including "
        description += f"{', '.join([match.matched_text for match in cluster[:3]])}"
        if len(cluster) > 3:
            description += f" and {len(cluster) - 3} other provisions"
        description += ". These provisions interact to create cumulative operational and financial burdens."
        
        return {"title": title, "description": description}
    
    def _filter_clusters(self, clusters: List[OneroousProvisionCluster]) -> List[OneroousProvisionCluster]:
        """Filter clusters based on significance and configuration limits.
        
        Args:
            clusters: List of clusters to filter
            
        Returns:
            Filtered list of significant clusters
        """
        # Sort by cumulative impact (most severe first)
        sorted_clusters = sorted(clusters, key=lambda c: c.cumulative_impact.value, reverse=True)
        
        # Filter out minimal impact clusters unless specifically configured to include them
        significant_clusters = [c for c in sorted_clusters if c.cumulative_impact != ImpactLevel.MINIMAL]
        
        # Apply configuration limits (if any)
        max_clusters = getattr(self.configuration, 'max_clusters_per_analysis', 10)
        
        return significant_clusters[:max_clusters]
    
    def _calculate_clustering_confidence(self, clusters: List[OneroousProvisionCluster], 
                                       provisions: List[ProvisionMatch], 
                                       content: str) -> ConfidenceMetrics:
        """Calculate confidence metrics for the clustering analysis.
        
        Args:
            clusters: Generated clusters
            provisions: All identified provisions
            content: Original content
            
        Returns:
            ConfidenceMetrics object
        """
        base_confidence = 80.0
        
        # Adjust based on cluster quality
        if clusters:
            # Lower confidence if too many high-impact clusters (might be over-clustering)
            high_impact_count = len([c for c in clusters if c.cumulative_impact in [ImpactLevel.CRITICAL, ImpactLevel.HIGH]])
            if high_impact_count > 3:
                base_confidence -= 10.0
            
            # Adjust based on cluster sizes (prefer moderate-sized clusters)
            avg_cluster_size = sum(len(c.primary_terms) for c in clusters) / len(clusters)
            if avg_cluster_size < 2:
                base_confidence -= 15.0
            elif avg_cluster_size > 6:
                base_confidence -= 10.0
        
        # Adjust based on provision coverage
        clustered_provisions = sum(len(c.primary_terms) for c in clusters)
        coverage_ratio = clustered_provisions / max(1, len(provisions))
        if coverage_ratio < 0.3:
            base_confidence += 5.0  # Good - not over-clustering
        elif coverage_ratio > 0.8:
            base_confidence -= 10.0  # Might be clustering too aggressively
        
        final_confidence = max(20.0, min(90.0, base_confidence))
        
        # Calculate uncertainty range
        uncertainty_factor = self.configuration.uncertainty_range_factor
        lower_bound = max(15.0, final_confidence - (final_confidence * uncertainty_factor))
        upper_bound = min(95.0, final_confidence + (final_confidence * uncertainty_factor))
        
        # Generate factors and sources
        confidence_factors = [
            f"Analyzed {len(provisions)} lease provisions",
            f"Generated {len(clusters)} meaningful clusters",
            "Relationship mapping completed"
        ]
        
        uncertainty_sources = []
        if clusters:
            uncertainty_sources.extend([
                f"{len(clusters)} provision clusters identified",
                f"{high_impact_count} high-impact clusters detected"
            ])
        
        return ConfidenceMetrics(
            confidence_score=final_confidence,
            uncertainty_range={"lower": lower_bound, "upper": upper_bound},
            confidence_factors=confidence_factors,
            uncertainty_sources=uncertainty_sources
        )


def analyze_provision_clusters(analysis_content: str, configuration: Configuration) -> Tuple[List[OneroousProvisionCluster], ConfidenceMetrics]:
    """Main entry point for onerous provision clustering analysis.
    
    Args:
        analysis_content: The primary lease analysis text to review
        configuration: Reviewer configuration settings
        
    Returns:
        Tuple of (list of OneroousProvisionCluster objects, ConfidenceMetrics for the analysis)
    """
    analyzer = ClusteringAnalyzer(configuration)
    return analyzer.analyze_clusters(analysis_content) 