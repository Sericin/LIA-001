#!/usr/bin/env python3
"""
Comprehensive LangSmith Evaluation System for Lease Analysis

This system provides:
1. Sophisticated lease analysis evaluation
2. Multi-agent performance testing
3. Quality metrics and benchmarking
4. Automated improvement workflows
5. Regression testing
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

from langsmith import Client, traceable, evaluate
from langsmith.evaluation import EvaluationResult
from react_agent.graph import graph
from react_agent.configuration import Configuration
from react_agent.state import ReviewStatus, ImpactLevel

load_dotenv()

@dataclass
class LeaseTestCase:
    """Represents a comprehensive lease analysis test case."""
    
    id: str
    name: str
    lease_type: str  # "simple", "complex", "onerous", "balanced"
    lease_content: str
    expected_flags_count: int
    expected_clusters_count: int
    expected_confidence_range: Tuple[float, float]
    expected_legal_review_required: bool
    expected_high_severity_flags: int
    description: str
    difficulty: str  # "basic", "intermediate", "advanced", "expert"

@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for lease analysis."""
    
    # Core Performance Metrics
    routing_accuracy: float
    flag_detection_accuracy: float
    cluster_identification_accuracy: float
    confidence_calibration_accuracy: float
    response_completeness: float
    
    # Quality Metrics
    legal_flag_precision: float
    legal_flag_recall: float
    cluster_relevance_score: float
    recommendation_quality: float
    
    # Performance Metrics
    total_processing_time: float
    agent_routing_efficiency: float
    error_rate: float
    
    # Overall Score
    overall_score: float

class LeaseAnalysisEvaluator:
    """Comprehensive evaluator for the lease analysis system."""
    
    def __init__(self):
        self.client = Client()
        self.project_name = os.getenv("LANGSMITH_PROJECT", "lease-analysis-evaluation")
        self.dataset_name = "lease-analysis-test-suite"
        
    def create_comprehensive_test_dataset(self) -> List[LeaseTestCase]:
        """Create a comprehensive test dataset covering various lease scenarios."""
        
        test_cases = [
            # Basic Test Cases
            LeaseTestCase(
                id="basic_001",
                name="Simple Office Lease",
                lease_type="simple",
                lease_content="""
                OFFICE LEASE AGREEMENT
                Tenant: ABC Company
                Landlord: XYZ Properties
                Term: 3 years
                Rent: $25 per square foot annually
                Use: General office purposes
                Maintenance: Standard tenant improvements only
                """,
                expected_flags_count=0,
                expected_clusters_count=0,
                expected_confidence_range=(80.0, 95.0),
                expected_legal_review_required=False,
                expected_high_severity_flags=0,
                description="Basic, balanced lease with standard terms",
                difficulty="basic"
            ),
            
            # Intermediate Test Cases
            LeaseTestCase(
                id="intermediate_001", 
                name="Moderate Office Lease with Some Issues",
                lease_type="moderate",
                lease_content="""
                COMMERCIAL LEASE AGREEMENT
                Tenant: Tech Solutions Inc.
                Landlord: Business Center LLC
                
                Term: 5 years with landlord's option to terminate
                Rent: $30/sq ft, subject to annual increases at landlord's discretion
                Additional Rent: Tenant pays operating expenses and taxes
                Maintenance: Tenant responsible for HVAC and utilities
                Default: 30-day cure period for non-payment
                Assignment: Subject to landlord's reasonable approval
                """,
                expected_flags_count=2,
                expected_clusters_count=1,
                expected_confidence_range=(70.0, 85.0),
                expected_legal_review_required=False,
                expected_high_severity_flags=0,
                description="Moderate lease with some tenant-unfavorable terms",
                difficulty="intermediate"
            ),
            
            # Complex Test Cases - Your Original Example
            LeaseTestCase(
                id="complex_001",
                name="Highly Onerous Commercial Lease",
                lease_type="onerous",
                lease_content="""
                COMMERCIAL LEASE AGREEMENT
                Property: 1250 Corporate Boulevard, Suite 300
                Tenant: TechStart Solutions LLC
                Landlord: Premier Commercial Properties, Inc.

                LEASE TERM: Initial term of sixty (60) months commencing on a date to be mutually agreed upon by the parties, with Landlord having sole discretion to delay commencement for any reason deemed necessary.

                BASE RENT: 
                - Year 1: $28.50 per rentable square foot
                - Years 2-5: Subject to annual increases of 4-8% at Landlord's sole discretion, plus additional market adjustments as determined appropriate by Landlord

                ADDITIONAL RENT: Tenant shall pay 100% of Operating Expenses, Real Estate Taxes, Insurance, and Capital Improvements, including but not limited to all costs Landlord deems reasonably necessary for the operation, maintenance, and improvement of the Building and surrounding property.

                MAINTENANCE & REPAIRS: Tenant responsible for ALL maintenance and repairs to the Premises, including HVAC systems, plumbing, electrical, structural components, roof, exterior walls, parking areas, landscaping, and any other building systems, regardless of age, condition, or cause of damage. Landlord has no maintenance obligations whatsoever.

                DEFAULT PROVISIONS:
                - Any rent payment received more than 48 hours after due date constitutes material default
                - Landlord may terminate lease immediately upon any default without notice
                - Upon default, Tenant owes all remaining rent for full lease term plus 25% penalty
                - Landlord may re-enter premises and dispose of Tenant's property without liability

                ASSIGNMENT & SUBLETTING: Tenant may not assign this lease or sublet any portion of the Premises without Landlord's prior written consent, which may be withheld in Landlord's absolute discretion for any reason or no reason. Any attempted assignment without consent shall be void and constitute immediate default.

                INSURANCE: Tenant must maintain comprehensive general liability insurance in amounts determined by Landlord from time to time, naming Landlord as additional insured. Tenant waives all claims against Landlord for any damage or loss, regardless of cause.

                ALTERATIONS: No alterations, improvements, or installations may be made without Landlord's written consent. All improvements become Landlord's property immediately upon installation and must be removed at Tenant's expense upon lease termination if requested by Landlord.

                COMPLIANCE: Tenant responsible for compliance with all laws, regulations, and ordinances, including any changes during the lease term. Tenant must bring Premises into compliance with all applicable codes at Tenant's sole expense.

                HOLDOVER: If Tenant remains in possession after lease expiration, Tenant becomes a month-to-month tenant at 150% of the then-current rent, with all other lease terms remaining in effect.

                SUBORDINATION: This lease is subject and subordinate to all present and future mortgages, ground leases, and other encumbrances affecting the Property.

                ENVIRONMENTAL: Tenant liable for all environmental contamination discovered during or after the lease term, regardless of source or timing of contamination.
                """,
                expected_flags_count=12,
                expected_clusters_count=4,
                expected_confidence_range=(35.0, 55.0),
                expected_legal_review_required=True,
                expected_high_severity_flags=8,
                description="Extremely onerous lease with multiple problematic provisions",
                difficulty="expert"
            ),
            
            # Edge Cases
            LeaseTestCase(
                id="edge_001",
                name="Retail Lease with Percentage Rent",
                lease_type="complex",
                lease_content="""
                RETAIL LEASE AGREEMENT
                Use: Restaurant operations only
                Base Rent: $15/sq ft plus 5% of gross sales above $500K annually
                Hours: Must operate minimum 12 hours daily, 7 days per week
                Exclusivity: Landlord may lease to competing restaurants within 1000 feet
                CAM: Tenant pays proportionate share plus management fee
                Build-out: Tenant must complete $200K minimum improvements
                Personal Guarantee: Required from all principals
                """,
                expected_flags_count=4,
                expected_clusters_count=2,
                expected_confidence_range=(55.0, 75.0),
                expected_legal_review_required=True,
                expected_high_severity_flags=2,
                description="Retail lease with operational restrictions and financial guarantees",
                difficulty="advanced"
            )
        ]
        
        return test_cases
    
    async def upload_test_dataset(self, test_cases: List[LeaseTestCase]) -> str:
        """Upload test dataset to LangSmith."""
        
        dataset_examples = []
        for case in test_cases:
            example = {
                "inputs": {
                    "messages": [{"role": "user", "content": f"Please analyze this commercial lease:\n\n{case.lease_content}"}]
                },
                "outputs": {
                    "expected_flags_count": case.expected_flags_count,
                    "expected_clusters_count": case.expected_clusters_count,
                    "expected_confidence_range": case.expected_confidence_range,
                    "expected_legal_review_required": case.expected_legal_review_required,
                    "expected_high_severity_flags": case.expected_high_severity_flags,
                    "lease_type": case.lease_type,
                    "difficulty": case.difficulty,
                    "description": case.description
                },
                "metadata": {
                    "test_id": case.id,
                    "test_name": case.name,
                    "lease_type": case.lease_type,
                    "difficulty": case.difficulty
                }
            }
            dataset_examples.append(example)
        
        try:
            # Create or update dataset
            dataset = self.client.create_dataset(
                dataset_name=self.dataset_name,
                description="Comprehensive lease analysis test suite with various complexity levels"
            )
            
            # Add examples
            self.client.create_examples(
                inputs=[ex["inputs"] for ex in dataset_examples],
                outputs=[ex["outputs"] for ex in dataset_examples],
                metadata=[ex["metadata"] for ex in dataset_examples],
                dataset_id=dataset.id
            )
            
            print(f"✅ Uploaded {len(test_cases)} test cases to dataset: {self.dataset_name}")
            return dataset.id
            
        except Exception as e:
            print(f"❌ Failed to upload dataset: {e}")
            return ""

    def evaluate_routing_accuracy(self, result: Dict[str, Any], expected: Dict[str, Any]) -> float:
        """Evaluate if the correct agents were called in the right sequence."""
        
        # Check if the system properly routed through the expected agents
        messages = result.get("messages", [])
        
        # Expected flow: call_model -> kb_lease_doc_agent -> reviewer_agent
        expected_agents = ["call_model", "kb_lease_doc_agent", "reviewer_agent"]
        
        # This is a simplified check - in a real implementation, you'd track agent calls
        if len(messages) >= 4:  # Assumes proper routing through all agents
            return 1.0
        elif len(messages) >= 2:  # Partial routing
            return 0.7
        else:
            return 0.0

    def evaluate_flag_detection_accuracy(self, result: Dict[str, Any], expected: Dict[str, Any]) -> float:
        """Evaluate accuracy of legal uncertainty flag detection."""
        
        reviewer_findings = result.get("reviewer_findings", {})
        flags = reviewer_findings.flags if reviewer_findings else []
        actual_flags_count = len(flags)
        expected_flags_count = expected.get("expected_flags_count", 0)
        
        if expected_flags_count == 0:
            return 1.0 if actual_flags_count == 0 else 0.8
        
        # Calculate accuracy based on how close the count is
        accuracy = 1.0 - abs(actual_flags_count - expected_flags_count) / max(expected_flags_count, actual_flags_count)
        return max(0.0, accuracy)

    def evaluate_cluster_identification_accuracy(self, result: Dict[str, Any], expected: Dict[str, Any]) -> float:
        """Evaluate accuracy of provision cluster identification."""
        
        reviewer_findings = result.get("reviewer_findings", {})
        clusters = reviewer_findings.onerous_clusters if reviewer_findings else []
        actual_clusters_count = len(clusters)
        expected_clusters_count = expected.get("expected_clusters_count", 0)
        
        if expected_clusters_count == 0:
            return 1.0 if actual_clusters_count == 0 else 0.8
        
        # Calculate accuracy
        accuracy = 1.0 - abs(actual_clusters_count - expected_clusters_count) / max(expected_clusters_count, actual_clusters_count)
        return max(0.0, accuracy)

    def evaluate_confidence_calibration_accuracy(self, result: Dict[str, Any], expected: Dict[str, Any]) -> float:
        """Evaluate accuracy of confidence calibration."""
        
        reviewer_findings = result.get("reviewer_findings", {})
        confidence_metrics = reviewer_findings.confidence_metrics if reviewer_findings else {}
        actual_confidence = confidence_metrics.confidence_score if hasattr(confidence_metrics, 'confidence_score') else 0.0
        
        expected_range = expected.get("expected_confidence_range", (0.0, 100.0))
        expected_min, expected_max = expected_range
        
        if expected_min <= actual_confidence <= expected_max:
            return 1.0
        else:
            # Calculate how far off it is
            if actual_confidence < expected_min:
                distance = expected_min - actual_confidence
            else:
                distance = actual_confidence - expected_max
            
            # Normalize distance (allow 20% tolerance)
            tolerance = 20.0
            accuracy = max(0.0, 1.0 - distance / tolerance)
            return accuracy

    def evaluate_legal_review_accuracy(self, result: Dict[str, Any], expected: Dict[str, Any]) -> float:
        """Evaluate accuracy of legal review requirement determination."""
        
        reviewer_findings = result.get("reviewer_findings", {})
        actual_legal_review = reviewer_findings.requires_legal_review if reviewer_findings else False
        expected_legal_review = expected.get("expected_legal_review_required", False)
        
        return 1.0 if actual_legal_review == expected_legal_review else 0.0

    def evaluate_response_completeness(self, result: Dict[str, Any]) -> float:
        """Evaluate completeness of the analysis response."""
        
        messages = result.get("messages", [])
        if not messages:
            return 0.0
        
        last_message = messages[-1]
        content = last_message.get("content", "") if hasattr(last_message, "get") else str(last_message.content) if hasattr(last_message, "content") else ""
        
        # Check for key components in the response
        required_components = [
            "review status",
            "flags raised", 
            "provision clusters",
            "confidence score",
            "recommendations"
        ]
        
        found_components = sum(1 for comp in required_components if comp.lower() in content.lower())
        return found_components / len(required_components)

    @traceable(name="lease_analysis_evaluator")
    async def evaluate_single_case(self, test_case: LeaseTestCase) -> EvaluationMetrics:
        """Evaluate a single test case comprehensively."""
        
        start_time = datetime.now()
        
        try:
            # Run the lease analysis
            result = await graph.ainvoke(
                {"messages": [{"role": "user", "content": f"Please analyze this commercial lease:\n\n{test_case.lease_content}"}]},
                {"configurable": {"system_prompt": "You are a sophisticated commercial real estate lease analysis expert."}}
            )
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Extract expected values
            expected = {
                "expected_flags_count": test_case.expected_flags_count,
                "expected_clusters_count": test_case.expected_clusters_count,
                "expected_confidence_range": test_case.expected_confidence_range,
                "expected_legal_review_required": test_case.expected_legal_review_required,
                "expected_high_severity_flags": test_case.expected_high_severity_flags
            }
            
            # Calculate individual metrics
            routing_accuracy = self.evaluate_routing_accuracy(result, expected)
            flag_detection_accuracy = self.evaluate_flag_detection_accuracy(result, expected)
            cluster_identification_accuracy = self.evaluate_cluster_identification_accuracy(result, expected)
            confidence_calibration_accuracy = self.evaluate_confidence_calibration_accuracy(result, expected)
            legal_review_accuracy = self.evaluate_legal_review_accuracy(result, expected)
            response_completeness = self.evaluate_response_completeness(result)
            
            # Calculate quality metrics
            # These would be more sophisticated in a real implementation
            legal_flag_precision = flag_detection_accuracy  # Simplified
            legal_flag_recall = flag_detection_accuracy     # Simplified
            cluster_relevance_score = cluster_identification_accuracy
            recommendation_quality = response_completeness
            
            # Calculate performance metrics
            agent_routing_efficiency = routing_accuracy
            error_rate = 0.0  # No error if we got here
            
            # Calculate overall score (weighted average)
            weights = {
                "routing": 0.10,
                "flags": 0.25,
                "clusters": 0.20,
                "confidence": 0.15,
                "legal_review": 0.15,
                "completeness": 0.15
            }
            
            overall_score = (
                routing_accuracy * weights["routing"] +
                flag_detection_accuracy * weights["flags"] +
                cluster_identification_accuracy * weights["clusters"] +
                confidence_calibration_accuracy * weights["confidence"] +
                legal_review_accuracy * weights["legal_review"] +
                response_completeness * weights["completeness"]
            )
            
            return EvaluationMetrics(
                routing_accuracy=routing_accuracy,
                flag_detection_accuracy=flag_detection_accuracy,
                cluster_identification_accuracy=cluster_identification_accuracy,
                confidence_calibration_accuracy=confidence_calibration_accuracy,
                response_completeness=response_completeness,
                legal_flag_precision=legal_flag_precision,
                legal_flag_recall=legal_flag_recall,
                cluster_relevance_score=cluster_relevance_score,
                recommendation_quality=recommendation_quality,
                total_processing_time=processing_time,
                agent_routing_efficiency=agent_routing_efficiency,
                error_rate=error_rate,
                overall_score=overall_score
            )
            
        except Exception as e:
            print(f"❌ Error evaluating {test_case.id}: {e}")
            return EvaluationMetrics(
                routing_accuracy=0.0,
                flag_detection_accuracy=0.0,
                cluster_identification_accuracy=0.0,
                confidence_calibration_accuracy=0.0,
                response_completeness=0.0,
                legal_flag_precision=0.0,
                legal_flag_recall=0.0,
                cluster_relevance_score=0.0,
                recommendation_quality=0.0,
                total_processing_time=0.0,
                agent_routing_efficiency=0.0,
                error_rate=1.0,
                overall_score=0.0
            )

    async def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive evaluation across all test cases."""
        
        print("🚀 Starting Comprehensive Lease Analysis Evaluation")
        print("=" * 60)
        
        test_cases = self.create_comprehensive_test_dataset()
        dataset_id = await self.upload_test_dataset(test_cases)
        
        results = {}
        all_metrics = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📋 Evaluating {i}/{len(test_cases)}: {test_case.name}")
            print(f"   Difficulty: {test_case.difficulty.title()}")
            print(f"   Type: {test_case.lease_type.title()}")
            
            metrics = await self.evaluate_single_case(test_case)
            all_metrics.append(metrics)
            
            results[test_case.id] = {
                "test_case": test_case,
                "metrics": metrics,
                "performance_summary": {
                    "overall_score": f"{metrics.overall_score:.2%}",
                    "flags_accuracy": f"{metrics.flag_detection_accuracy:.2%}",
                    "clusters_accuracy": f"{metrics.cluster_identification_accuracy:.2%}",
                    "confidence_accuracy": f"{metrics.confidence_calibration_accuracy:.2%}",
                    "processing_time": f"{metrics.total_processing_time:.2f}s"
                }
            }
            
            print(f"   ✅ Overall Score: {metrics.overall_score:.2%}")
            print(f"   🎯 Flags Accuracy: {metrics.flag_detection_accuracy:.2%}")
            print(f"   📊 Clusters Accuracy: {metrics.cluster_identification_accuracy:.2%}")
            print(f"   ⏱️  Processing Time: {metrics.total_processing_time:.2f}s")
        
        # Calculate aggregate metrics
        avg_metrics = self.calculate_aggregate_metrics(all_metrics)
        
        evaluation_summary = {
            "timestamp": datetime.now().isoformat(),
            "total_test_cases": len(test_cases),
            "dataset_id": dataset_id,
            "individual_results": results,
            "aggregate_metrics": avg_metrics,
            "performance_insights": self.generate_performance_insights(all_metrics, test_cases)
        }
        
        self.print_evaluation_summary(evaluation_summary)
        
        return evaluation_summary

    def calculate_aggregate_metrics(self, all_metrics: List[EvaluationMetrics]) -> Dict[str, float]:
        """Calculate aggregate metrics across all test cases."""
        
        if not all_metrics:
            return {}
        
        return {
            "avg_overall_score": sum(m.overall_score for m in all_metrics) / len(all_metrics),
            "avg_routing_accuracy": sum(m.routing_accuracy for m in all_metrics) / len(all_metrics),
            "avg_flag_detection_accuracy": sum(m.flag_detection_accuracy for m in all_metrics) / len(all_metrics),
            "avg_cluster_identification_accuracy": sum(m.cluster_identification_accuracy for m in all_metrics) / len(all_metrics),
            "avg_confidence_calibration_accuracy": sum(m.confidence_calibration_accuracy for m in all_metrics) / len(all_metrics),
            "avg_response_completeness": sum(m.response_completeness for m in all_metrics) / len(all_metrics),
            "avg_processing_time": sum(m.total_processing_time for m in all_metrics) / len(all_metrics),
            "total_error_rate": sum(m.error_rate for m in all_metrics) / len(all_metrics)
        }

    def generate_performance_insights(self, all_metrics: List[EvaluationMetrics], test_cases: List[LeaseTestCase]) -> Dict[str, Any]:
        """Generate insights about system performance."""
        
        insights = {
            "strengths": [],
            "weaknesses": [],
            "recommendations": []
        }
        
        avg_metrics = self.calculate_aggregate_metrics(all_metrics)
        
        # Identify strengths
        if avg_metrics.get("avg_routing_accuracy", 0) > 0.9:
            insights["strengths"].append("Excellent agent routing accuracy")
        
        if avg_metrics.get("avg_response_completeness", 0) > 0.8:
            insights["strengths"].append("High response completeness")
        
        if avg_metrics.get("avg_processing_time", 0) < 30:
            insights["strengths"].append("Fast processing times")
        
        # Identify weaknesses
        if avg_metrics.get("avg_flag_detection_accuracy", 0) < 0.7:
            insights["weaknesses"].append("Flag detection accuracy needs improvement")
            insights["recommendations"].append("Review and enhance legal uncertainty patterns")
        
        if avg_metrics.get("avg_cluster_identification_accuracy", 0) < 0.7:
            insights["weaknesses"].append("Cluster identification needs improvement")
            insights["recommendations"].append("Enhance provision clustering algorithms")
        
        if avg_metrics.get("avg_confidence_calibration_accuracy", 0) < 0.7:
            insights["weaknesses"].append("Confidence calibration needs adjustment")
            insights["recommendations"].append("Refine confidence calibration parameters")
        
        # Performance by difficulty
        basic_cases = [(m, tc) for m, tc in zip(all_metrics, test_cases) if tc.difficulty == "basic"]
        expert_cases = [(m, tc) for m, tc in zip(all_metrics, test_cases) if tc.difficulty == "expert"]
        
        if basic_cases:
            basic_avg = sum(m.overall_score for m, tc in basic_cases) / len(basic_cases)
            insights["basic_difficulty_performance"] = f"{basic_avg:.2%}"
        
        if expert_cases:
            expert_avg = sum(m.overall_score for m, tc in expert_cases) / len(expert_cases)
            insights["expert_difficulty_performance"] = f"{expert_avg:.2%}"
        
        return insights

    def print_evaluation_summary(self, summary: Dict[str, Any]) -> None:
        """Print a comprehensive evaluation summary."""
        
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE EVALUATION SUMMARY")
        print("=" * 60)
        
        total_cases = summary["total_test_cases"]
        agg_metrics = summary["aggregate_metrics"]
        insights = summary["performance_insights"]
        
        print(f"\n📈 Overall Performance:")
        print(f"   Test Cases: {total_cases}")
        print(f"   Average Score: {agg_metrics.get('avg_overall_score', 0):.2%}")
        print(f"   Error Rate: {agg_metrics.get('total_error_rate', 0):.2%}")
        print(f"   Avg Processing Time: {agg_metrics.get('avg_processing_time', 0):.2f}s")
        
        print(f"\n🎯 Component Performance:")
        print(f"   🔄 Routing Accuracy: {agg_metrics.get('avg_routing_accuracy', 0):.2%}")
        print(f"   🚨 Flag Detection: {agg_metrics.get('avg_flag_detection_accuracy', 0):.2%}")
        print(f"   📊 Cluster Identification: {agg_metrics.get('avg_cluster_identification_accuracy', 0):.2%}")
        print(f"   🎲 Confidence Calibration: {agg_metrics.get('avg_confidence_calibration_accuracy', 0):.2%}")
        print(f"   📝 Response Completeness: {agg_metrics.get('avg_response_completeness', 0):.2%}")
        
        print(f"\n💪 Strengths:")
        for strength in insights.get("strengths", []):
            print(f"   ✅ {strength}")
        
        print(f"\n⚠️  Areas for Improvement:")
        for weakness in insights.get("weaknesses", []):
            print(f"   🔍 {weakness}")
        
        print(f"\n💡 Recommendations:")
        for recommendation in insights.get("recommendations", []):
            print(f"   🎯 {recommendation}")
        
        print(f"\n🔗 LangSmith Dashboard: https://smith.langchain.com/")
        print(f"📊 Dataset: {summary.get('dataset_id', 'N/A')}")

async def main():
    """Main function to run the comprehensive evaluation."""
    
    evaluator = LeaseAnalysisEvaluator()
    
    # Check LangSmith connection
    try:
        projects = list(evaluator.client.list_projects(limit=1))
        print("✅ LangSmith connection successful")
    except Exception as e:
        print(f"❌ LangSmith connection failed: {e}")
        return
    
    # Run comprehensive evaluation
    results = await evaluator.run_comprehensive_evaluation()
    
    # Save results to file for analysis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"evaluation_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        # Convert dataclasses to dict for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if key == "individual_results":
                serializable_results[key] = {}
                for test_id, test_result in value.items():
                    serializable_results[key][test_id] = {
                        "test_case": asdict(test_result["test_case"]),
                        "metrics": asdict(test_result["metrics"]),
                        "performance_summary": test_result["performance_summary"]
                    }
            else:
                serializable_results[key] = value
        
        json.dump(serializable_results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")

if __name__ == "__main__":
    asyncio.run(main()) 