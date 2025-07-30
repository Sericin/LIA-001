#!/usr/bin/env python3
"""
LangSmith A/B Testing System for Lease Analysis Improvements

This system allows you to:
1. Test different configuration settings
2. Compare prompt variations
3. Evaluate model changes
4. Measure improvement impact
5. Run statistical significance tests
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from statistics import mean, stdev
import math
from dotenv import load_dotenv

from langsmith import Client, traceable
from react_agent.graph import graph
from react_agent.configuration import Configuration

load_dotenv()

@dataclass
class ABTestConfiguration:
    """Represents a configuration variant for A/B testing."""
    
    variant_name: str
    description: str
    configuration_changes: Dict[str, Any]
    expected_improvements: List[str]

@dataclass
class ABTestResult:
    """Results for a single A/B test variant."""
    
    variant_name: str
    test_cases_run: int
    avg_overall_score: float
    avg_flag_detection_accuracy: float
    avg_cluster_identification_accuracy: float
    avg_confidence_calibration_accuracy: float
    avg_processing_time: float
    error_rate: float
    success_rate: float

class LeaseAnalysisABTester:
    """A/B testing system for lease analysis improvements."""
    
    def __init__(self):
        self.client = Client()
        self.project_name = "lease-analysis-ab-testing"
    
    def create_test_configurations(self) -> List[ABTestConfiguration]:
        """Create different configuration variants for testing."""
        
        configurations = [
            # Control Group - Current Configuration
            ABTestConfiguration(
                variant_name="control",
                description="Current production configuration",
                configuration_changes={},
                expected_improvements=[]
            ),
            
            # Variant A - Higher Sensitivity
            ABTestConfiguration(
                variant_name="high_sensitivity",
                description="Increased flag sensitivity for more detection",
                configuration_changes={
                    "flag_sensitivity_legal": "high",
                    "flag_sensitivity_ambiguity": "high",
                    "max_flags_per_analysis": 25
                },
                expected_improvements=["Higher flag detection", "More comprehensive analysis"]
            ),
            
            # Variant B - Conservative Confidence
            ABTestConfiguration(
                variant_name="conservative_confidence",
                description="More conservative confidence calibration",
                configuration_changes={
                    "confidence_calibration_strategy": "conservative",
                    "confidence_threshold_high": 85.0,
                    "confidence_threshold_medium": 70.0
                },
                expected_improvements=["More realistic confidence scores", "Better uncertainty handling"]
            ),
            
            # Variant C - Comprehensive Review
            ABTestConfiguration(
                variant_name="comprehensive_review",
                description="Maximum review depth and analysis",
                configuration_changes={
                    "review_depth_level": "comprehensive",
                    "enable_legal_uncertainty_detection": True,
                    "enable_ambiguity_detection": True,
                    "enable_onerous_clustering": True,
                    "enable_confidence_calibration": True,
                    "max_terms_per_cluster": 10,
                    "cluster_interaction_depth": 3
                },
                expected_improvements=["Maximum analysis depth", "Best cluster detection"]
            ),
            
            # Variant D - Performance Optimized
            ABTestConfiguration(
                variant_name="performance_optimized",
                description="Optimized for speed while maintaining quality",
                configuration_changes={
                    "max_flags_per_analysis": 15,
                    "reviewer_timeout_seconds": 60,
                    "max_terms_per_cluster": 6,
                    "cluster_interaction_depth": 2
                },
                expected_improvements=["Faster processing", "Lower latency"]
            ),
            
            # Variant E - Balanced Approach
            ABTestConfiguration(
                variant_name="balanced_enhanced",
                description="Enhanced balanced configuration",
                configuration_changes={
                    "confidence_calibration_strategy": "balanced",
                    "flag_sensitivity_legal": "medium",
                    "flag_sensitivity_ambiguity": "medium", 
                    "max_flags_per_analysis": 20,
                    "confidence_threshold_high": 80.0,
                    "uncertainty_range_factor": 0.12
                },
                expected_improvements=["Balanced performance", "Optimal accuracy/speed trade-off"]
            )
        ]
        
        return configurations

    def create_quick_test_cases(self) -> List[str]:
        """Create a smaller set of test cases for quick A/B testing."""
        
        return [
            # Simple lease - should have minimal flags
            """
            OFFICE LEASE AGREEMENT
            Tenant: ABC Company, Landlord: XYZ Properties
            Term: 3 years, Rent: $25/sq ft annually
            Use: General office purposes
            """,
            
            # Moderate lease - some issues
            """
            COMMERCIAL LEASE
            Term: 5 years, Rent: $30/sq ft + annual increases at landlord's discretion
            Tenant pays operating expenses and taxes
            Maintenance: Tenant responsible for HVAC
            """,
            
            # Complex lease - many issues
            """
            LEASE AGREEMENT
            Term: 5 years, landlord sole discretion to delay start
            Rent: Base $28/sq ft + 4-8% increases at landlord's sole discretion
            Tenant pays ALL maintenance regardless of age or cause
            Default: 48 hours late = immediate termination + 25% penalty
            Assignment: Landlord's absolute discretion for any reason or no reason
            Tenant waives all claims regardless of cause
            """,
            
            # Retail lease - specialized terms
            """
            RETAIL LEASE
            Use: Restaurant only, 12 hours daily required
            Rent: $15/sq ft + 5% of sales over $500K
            Personal guarantee required, $200K minimum improvements
            """
        ]

    @traceable(name="ab_test_variant")
    async def test_single_variant(self, variant: ABTestConfiguration, test_cases: List[str]) -> ABTestResult:
        """Test a single configuration variant against test cases."""
        
        print(f"\n🧪 Testing Variant: {variant.variant_name}")
        print(f"   Description: {variant.description}")
        
        # Create configuration with changes
        base_config = Configuration()
        
        # Apply configuration changes
        config_dict = {}
        for key, value in variant.configuration_changes.items():
            config_dict[key] = value
            print(f"   Setting {key} = {value}")
        
        results = []
        processing_times = []
        errors = 0
        
        for i, test_case in enumerate(test_cases, 1):
            try:
                start_time = datetime.now()
                
                # Run analysis with variant configuration
                result = await graph.ainvoke(
                    {"messages": [{"role": "user", "content": f"Please analyze this commercial lease:\n\n{test_case}"}]},
                    {"configurable": config_dict}
                )
                
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                processing_times.append(processing_time)
                
                # Extract metrics
                reviewer_findings = result.get("reviewer_findings", {})
                flags = reviewer_findings.flags if reviewer_findings else []
                clusters = reviewer_findings.onerous_clusters if reviewer_findings else []
                confidence_metrics = reviewer_findings.confidence_metrics if reviewer_findings else {}
                
                test_result = {
                    "flags_count": len(flags),
                    "clusters_count": len(clusters),
                    "confidence_score": confidence_metrics.confidence_score if hasattr(confidence_metrics, 'confidence_score') else 0.0,
                    "processing_time": processing_time,
                    "success": True
                }
                
                results.append(test_result)
                print(f"   Test {i}: ✅ {len(flags)} flags, {len(clusters)} clusters, {processing_time:.2f}s")
                
            except Exception as e:
                errors += 1
                print(f"   Test {i}: ❌ Error: {str(e)[:50]}...")
                results.append({
                    "flags_count": 0,
                    "clusters_count": 0,
                    "confidence_score": 0.0,
                    "processing_time": 0.0,
                    "success": False
                })
        
        # Calculate aggregate metrics
        successful_results = [r for r in results if r["success"]]
        
        if successful_results:
            avg_flags = mean([r["flags_count"] for r in successful_results])
            avg_clusters = mean([r["clusters_count"] for r in successful_results])
            avg_confidence = mean([r["confidence_score"] for r in successful_results])
            avg_processing_time = mean([r["processing_time"] for r in successful_results])
        else:
            avg_flags = avg_clusters = avg_confidence = avg_processing_time = 0.0
        
        success_rate = len(successful_results) / len(results) if results else 0.0
        error_rate = errors / len(results) if results else 1.0
        
        # Calculate synthetic overall score (simplified)
        overall_score = success_rate * 0.4 + (avg_confidence / 100.0) * 0.3 + min(1.0, avg_flags / 5.0) * 0.3
        
        return ABTestResult(
            variant_name=variant.variant_name,
            test_cases_run=len(test_cases),
            avg_overall_score=overall_score,
            avg_flag_detection_accuracy=min(1.0, avg_flags / 5.0),  # Normalized
            avg_cluster_identification_accuracy=min(1.0, avg_clusters / 3.0),  # Normalized
            avg_confidence_calibration_accuracy=avg_confidence / 100.0,
            avg_processing_time=avg_processing_time,
            error_rate=error_rate,
            success_rate=success_rate
        )

    def calculate_statistical_significance(self, control_results: List[float], variant_results: List[float]) -> Dict[str, Any]:
        """Calculate statistical significance between control and variant."""
        
        if len(control_results) < 2 or len(variant_results) < 2:
            return {
                "significant": False,
                "reason": "Insufficient data for significance testing",
                "p_value": None,
                "effect_size": None
            }
        
        # Calculate means and standard deviations
        control_mean = mean(control_results)
        variant_mean = mean(variant_results)
        control_std = stdev(control_results) if len(control_results) > 1 else 0
        variant_std = stdev(variant_results) if len(variant_results) > 1 else 0
        
        # Calculate effect size (Cohen's d)
        pooled_std = math.sqrt(((len(control_results) - 1) * control_std**2 + 
                               (len(variant_results) - 1) * variant_std**2) / 
                              (len(control_results) + len(variant_results) - 2))
        
        effect_size = (variant_mean - control_mean) / pooled_std if pooled_std > 0 else 0
        
        # Simplified significance test (in practice, you'd use proper statistical tests)
        # This is a rough approximation
        standard_error = math.sqrt(control_std**2 / len(control_results) + variant_std**2 / len(variant_results))
        t_statistic = abs(variant_mean - control_mean) / standard_error if standard_error > 0 else 0
        
        # Rough p-value estimation (use proper statistical libraries in production)
        significant = t_statistic > 2.0  # Simplified threshold
        
        return {
            "significant": significant,
            "control_mean": control_mean,
            "variant_mean": variant_mean,
            "improvement": ((variant_mean - control_mean) / control_mean * 100) if control_mean > 0 else 0,
            "effect_size": effect_size,
            "t_statistic": t_statistic,
            "recommendation": "Deploy variant" if significant and variant_mean > control_mean else "Keep control"
        }

    async def run_ab_test_suite(self) -> Dict[str, Any]:
        """Run comprehensive A/B testing suite."""
        
        print("🚀 Starting Lease Analysis A/B Testing Suite")
        print("=" * 50)
        
        test_configurations = self.create_test_configurations()
        test_cases = self.create_quick_test_cases()
        
        print(f"📊 Testing {len(test_configurations)} variants with {len(test_cases)} test cases each")
        
        variant_results = {}
        
        # Test each variant
        for variant in test_configurations:
            result = await self.test_single_variant(variant, test_cases)
            variant_results[variant.variant_name] = {
                "configuration": variant,
                "result": result
            }
        
        # Compare variants to control
        control_result = variant_results.get("control", {}).get("result")
        if not control_result:
            print("❌ No control group found for comparison")
            return variant_results
        
        print(f"\n📈 A/B Test Results Summary")
        print("=" * 50)
        
        comparisons = {}
        
        for variant_name, variant_data in variant_results.items():
            if variant_name == "control":
                continue
                
            variant_result = variant_data["result"]
            configuration = variant_data["configuration"]
            
            # Compare key metrics
            metrics_comparison = {
                "overall_score": {
                    "control": control_result.avg_overall_score,
                    "variant": variant_result.avg_overall_score,
                    "improvement": ((variant_result.avg_overall_score - control_result.avg_overall_score) / control_result.avg_overall_score * 100) if control_result.avg_overall_score > 0 else 0
                },
                "flag_detection": {
                    "control": control_result.avg_flag_detection_accuracy,
                    "variant": variant_result.avg_flag_detection_accuracy,
                    "improvement": ((variant_result.avg_flag_detection_accuracy - control_result.avg_flag_detection_accuracy) / control_result.avg_flag_detection_accuracy * 100) if control_result.avg_flag_detection_accuracy > 0 else 0
                },
                "processing_time": {
                    "control": control_result.avg_processing_time,
                    "variant": variant_result.avg_processing_time,
                    "improvement": ((control_result.avg_processing_time - variant_result.avg_processing_time) / control_result.avg_processing_time * 100) if control_result.avg_processing_time > 0 else 0
                }
            }
            
            comparisons[variant_name] = {
                "configuration": configuration,
                "result": variant_result,
                "metrics_comparison": metrics_comparison
            }
            
            print(f"\n🔬 {variant_name.upper()}: {configuration.description}")
            print(f"   Overall Score: {variant_result.avg_overall_score:.2%} ({metrics_comparison['overall_score']['improvement']:+.1f}%)")
            print(f"   Flag Detection: {variant_result.avg_flag_detection_accuracy:.2%} ({metrics_comparison['flag_detection']['improvement']:+.1f}%)")
            print(f"   Processing Time: {variant_result.avg_processing_time:.2f}s ({metrics_comparison['processing_time']['improvement']:+.1f}%)")
            print(f"   Success Rate: {variant_result.success_rate:.2%}")
            
            # Determine recommendation
            if (metrics_comparison['overall_score']['improvement'] > 5 and 
                variant_result.success_rate >= control_result.success_rate):
                print(f"   🎯 RECOMMENDATION: ✅ Deploy - Significant improvement")
            elif metrics_comparison['overall_score']['improvement'] > 0:
                print(f"   🎯 RECOMMENDATION: 🔄 Consider - Marginal improvement")
            else:
                print(f"   🎯 RECOMMENDATION: ❌ Keep control - No improvement")
        
        # Find best performing variant
        best_variant = max(
            [v for k, v in variant_results.items() if k != "control"],
            key=lambda x: x["result"].avg_overall_score,
            default=None
        )
        
        if best_variant:
            best_name = best_variant["configuration"].variant_name
            best_score = best_variant["result"].avg_overall_score
            control_score = control_result.avg_overall_score
            improvement = ((best_score - control_score) / control_score * 100) if control_score > 0 else 0
            
            print(f"\n🏆 BEST PERFORMER: {best_name.upper()}")
            print(f"   Score: {best_score:.2%} (+{improvement:.1f}% vs control)")
            print(f"   Description: {best_variant['configuration'].description}")
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "control_result": asdict(control_result),
            "variant_comparisons": {},
            "best_variant": best_variant["configuration"].variant_name if best_variant else "control",
            "recommendations": self.generate_ab_test_recommendations(comparisons)
        }
        
        # Add comparisons to summary (convert dataclasses to dicts)
        for variant_name, comparison in comparisons.items():
            summary["variant_comparisons"][variant_name] = {
                "configuration": asdict(comparison["configuration"]),
                "result": asdict(comparison["result"]),
                "metrics_comparison": comparison["metrics_comparison"]
            }
        
        return summary

    def generate_ab_test_recommendations(self, comparisons: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on A/B test results."""
        
        recommendations = []
        
        # Find variants with significant improvements
        significant_improvements = []
        for variant_name, comparison in comparisons.items():
            overall_improvement = comparison["metrics_comparison"]["overall_score"]["improvement"]
            if overall_improvement > 5:
                significant_improvements.append((variant_name, overall_improvement))
        
        if significant_improvements:
            best_variant, best_improvement = max(significant_improvements, key=lambda x: x[1])
            recommendations.append(f"Deploy '{best_variant}' configuration for {best_improvement:.1f}% improvement")
        else:
            recommendations.append("Current configuration is optimal - no significant improvements found")
        
        # Performance-specific recommendations
        performance_variants = [
            name for name, comp in comparisons.items()
            if comp["metrics_comparison"]["processing_time"]["improvement"] > 20
        ]
        
        if performance_variants:
            recommendations.append(f"Consider '{performance_variants[0]}' for performance-critical scenarios")
        
        # Accuracy-specific recommendations
        accuracy_variants = [
            name for name, comp in comparisons.items()
            if comp["metrics_comparison"]["flag_detection"]["improvement"] > 10
        ]
        
        if accuracy_variants:
            recommendations.append(f"Consider '{accuracy_variants[0]}' for maximum accuracy requirements")
        
        return recommendations

async def main():
    """Main function to run A/B testing suite."""
    
    tester = LeaseAnalysisABTester()
    
    # Check LangSmith connection
    try:
        projects = list(tester.client.list_projects(limit=1))
        print("✅ LangSmith connection successful")
    except Exception as e:
        print(f"❌ LangSmith connection failed: {e}")
        return
    
    # Run A/B testing suite
    results = await tester.run_ab_test_suite()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"ab_test_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 A/B test results saved to: {results_file}")
    
    # Print final recommendations
    print(f"\n💡 FINAL RECOMMENDATIONS:")
    for i, rec in enumerate(results.get("recommendations", []), 1):
        print(f"   {i}. {rec}")

if __name__ == "__main__":
    asyncio.run(main()) 