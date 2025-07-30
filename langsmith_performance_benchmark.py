#!/usr/bin/env python3
"""
LangSmith Performance Benchmarking for Lease Analysis

This tool provides:
1. Continuous performance monitoring
2. Regression detection
3. Performance trends analysis
4. Alert system for degradation
5. Automated performance reporting
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import statistics
from dotenv import load_dotenv

from langsmith import Client, traceable
from react_agent.graph import graph

load_dotenv()

@dataclass
class PerformanceBenchmark:
    """Performance benchmark metrics."""
    
    timestamp: str
    test_suite_version: str
    total_processing_time: float
    avg_processing_time_per_case: float
    max_processing_time: float
    min_processing_time: float
    flags_detection_rate: float
    clusters_detection_rate: float
    avg_confidence_score: float
    success_rate: float
    error_rate: float
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None

@dataclass
class PerformanceAlert:
    """Performance alert information."""
    
    alert_type: str  # "degradation", "improvement", "anomaly"
    severity: str    # "low", "medium", "high", "critical"
    metric_name: str
    current_value: float
    baseline_value: float
    threshold_exceeded: float
    message: str
    timestamp: str

class LeaseAnalysisPerformanceBenchmark:
    """Performance benchmarking system for lease analysis."""
    
    def __init__(self):
        self.client = Client()
        self.project_name = os.getenv("LANGSMITH_PROJECT", "lease-analysis-performance")
        self.benchmark_file = "performance_benchmarks.json"
        
        # Performance thresholds
        self.thresholds = {
            "max_processing_time": 60.0,      # seconds
            "avg_processing_time": 30.0,      # seconds
            "min_success_rate": 0.95,          # 95%
            "max_error_rate": 0.05,            # 5%
            "min_flags_detection": 0.70,       # 70%
            "min_clusters_detection": 0.60,    # 60%
            "min_confidence_score": 0.50       # 50%
        }
    
    def create_performance_test_cases(self) -> List[Dict[str, Any]]:
        """Create standardized test cases for performance benchmarking."""
        
        return [
            {
                "id": "perf_simple_001",
                "complexity": "simple",
                "expected_processing_time": 15.0,  # seconds
                "lease_content": """
                OFFICE LEASE AGREEMENT
                Tenant: ABC Company
                Landlord: XYZ Properties  
                Term: 3 years
                Rent: $25 per square foot annually
                Use: General office purposes
                Maintenance: Standard tenant improvements only
                """
            },
            {
                "id": "perf_moderate_001", 
                "complexity": "moderate",
                "expected_processing_time": 25.0,
                "lease_content": """
                COMMERCIAL LEASE AGREEMENT
                Tenant: Tech Solutions Inc.
                Landlord: Business Center LLC
                
                Term: 5 years with landlord's option to terminate
                Rent: $30/sq ft, subject to annual increases at landlord's discretion
                Additional Rent: Tenant pays operating expenses and taxes
                Maintenance: Tenant responsible for HVAC and utilities
                Default: 30-day cure period for non-payment
                Assignment: Subject to landlord's reasonable approval
                Insurance: Comprehensive coverage required
                """
            },
            {
                "id": "perf_complex_001",
                "complexity": "complex", 
                "expected_processing_time": 45.0,
                "lease_content": """
                COMMERCIAL LEASE AGREEMENT
                
                LEASE TERM: Initial term of sixty (60) months commencing on a date to be mutually agreed upon by the parties, with Landlord having sole discretion to delay commencement for any reason deemed necessary.

                BASE RENT: Subject to annual increases of 4-8% at Landlord's sole discretion, plus additional market adjustments as determined appropriate by Landlord

                MAINTENANCE & REPAIRS: Tenant responsible for ALL maintenance and repairs to the Premises, including HVAC systems, plumbing, electrical, structural components, roof, exterior walls, parking areas, landscaping, and any other building systems, regardless of age, condition, or cause of damage.

                DEFAULT PROVISIONS:
                - Any rent payment received more than 48 hours after due date constitutes material default
                - Landlord may terminate lease immediately upon any default without notice
                - Upon default, Tenant owes all remaining rent for full lease term plus 25% penalty

                ASSIGNMENT & SUBLETTING: Tenant may not assign this lease or sublet any portion of the Premises without Landlord's prior written consent, which may be withheld in Landlord's absolute discretion for any reason or no reason.

                INSURANCE: Tenant waives all claims against Landlord for any damage or loss, regardless of cause.

                ENVIRONMENTAL: Tenant liable for all environmental contamination discovered during or after the lease term, regardless of source or timing of contamination.
                """
            }
        ]

    @traceable(name="performance_benchmark_test")
    async def run_single_benchmark_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single performance benchmark test."""
        
        start_time = datetime.now()
        
        try:
            # Run the lease analysis
            result = await graph.ainvoke(
                {"messages": [{"role": "user", "content": f"Please analyze this commercial lease:\n\n{test_case['lease_content']}"}]},
                {"configurable": {"system_prompt": "You are a sophisticated commercial real estate lease analysis expert."}}
            )
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Extract performance metrics
            reviewer_findings = result.get("reviewer_findings", {})
            flags = reviewer_findings.flags if reviewer_findings else []
            clusters = reviewer_findings.onerous_clusters if reviewer_findings else []
            confidence_metrics = reviewer_findings.confidence_metrics if reviewer_findings else {}
            
            return {
                "test_id": test_case["id"],
                "complexity": test_case["complexity"],
                "processing_time": processing_time,
                "expected_processing_time": test_case["expected_processing_time"],
                "flags_count": len(flags),
                "clusters_count": len(clusters),
                "confidence_score": confidence_metrics.confidence_score if hasattr(confidence_metrics, 'confidence_score') else 0.0,
                "success": True,
                "error": None,
                "timestamp": start_time.isoformat()
            }
            
        except Exception as e:
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            return {
                "test_id": test_case["id"],
                "complexity": test_case["complexity"],
                "processing_time": processing_time,
                "expected_processing_time": test_case["expected_processing_time"],
                "flags_count": 0,
                "clusters_count": 0,
                "confidence_score": 0.0,
                "success": False,
                "error": str(e),
                "timestamp": start_time.isoformat()
            }

    async def run_performance_benchmark_suite(self) -> PerformanceBenchmark:
        """Run complete performance benchmark suite."""
        
        print("🚀 Starting Performance Benchmark Suite")
        print("=" * 45)
        
        test_cases = self.create_performance_test_cases()
        test_results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📊 Running benchmark {i}/{len(test_cases)}: {test_case['complexity']}")
            
            result = await self.run_single_benchmark_test(test_case)
            test_results.append(result)
            
            status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
            print(f"   {status}: {result['processing_time']:.2f}s (expected: {result['expected_processing_time']:.2f}s)")
            
            if result["success"]:
                print(f"   📈 Flags: {result['flags_count']}, Clusters: {result['clusters_count']}")
                print(f"   🎯 Confidence: {result['confidence_score']:.1f}%")
        
        # Calculate aggregate metrics
        successful_tests = [r for r in test_results if r["success"]]
        total_tests = len(test_results)
        
        if successful_tests:
            processing_times = [r["processing_time"] for r in successful_tests]
            confidence_scores = [r["confidence_score"] for r in successful_tests]
            flags_counts = [r["flags_count"] for r in successful_tests]
            clusters_counts = [r["clusters_count"] for r in successful_tests]
            
            total_processing_time = sum(processing_times)
            avg_processing_time = statistics.mean(processing_times)
            max_processing_time = max(processing_times)
            min_processing_time = min(processing_times)
            avg_confidence_score = statistics.mean(confidence_scores)
            
            # Calculate detection rates (simplified)
            flags_detection_rate = statistics.mean([min(1.0, f/5.0) for f in flags_counts])  # Normalize to 0-1
            clusters_detection_rate = statistics.mean([min(1.0, c/3.0) for c in clusters_counts])  # Normalize to 0-1
        else:
            total_processing_time = avg_processing_time = max_processing_time = min_processing_time = 0.0
            avg_confidence_score = flags_detection_rate = clusters_detection_rate = 0.0
        
        success_rate = len(successful_tests) / total_tests if total_tests > 0 else 0.0
        error_rate = 1.0 - success_rate
        
        benchmark = PerformanceBenchmark(
            timestamp=datetime.now().isoformat(),
            test_suite_version="1.0",
            total_processing_time=total_processing_time,
            avg_processing_time_per_case=avg_processing_time,
            max_processing_time=max_processing_time,
            min_processing_time=min_processing_time,
            flags_detection_rate=flags_detection_rate,
            clusters_detection_rate=clusters_detection_rate,
            avg_confidence_score=avg_confidence_score,
            success_rate=success_rate,
            error_rate=error_rate
        )
        
        print(f"\n📊 PERFORMANCE BENCHMARK RESULTS")
        print("=" * 45)
        print(f"   Total Processing Time: {total_processing_time:.2f}s")
        print(f"   Average Time per Case: {avg_processing_time:.2f}s")
        print(f"   Max Processing Time: {max_processing_time:.2f}s")
        print(f"   Success Rate: {success_rate:.2%}")
        print(f"   Flags Detection Rate: {flags_detection_rate:.2%}")
        print(f"   Clusters Detection Rate: {clusters_detection_rate:.2%}")
        print(f"   Average Confidence: {avg_confidence_score:.1f}%")
        
        return benchmark

    def load_historical_benchmarks(self) -> List[PerformanceBenchmark]:
        """Load historical benchmark data."""
        
        try:
            if os.path.exists(self.benchmark_file):
                with open(self.benchmark_file, 'r') as f:
                    data = json.load(f)
                    return [PerformanceBenchmark(**item) for item in data]
            return []
        except Exception as e:
            print(f"⚠️  Could not load historical benchmarks: {e}")
            return []

    def save_benchmark(self, benchmark: PerformanceBenchmark) -> None:
        """Save benchmark to historical data."""
        
        historical_benchmarks = self.load_historical_benchmarks()
        historical_benchmarks.append(benchmark)
        
        # Keep only last 50 benchmarks
        if len(historical_benchmarks) > 50:
            historical_benchmarks = historical_benchmarks[-50:]
        
        try:
            with open(self.benchmark_file, 'w') as f:
                json.dump([asdict(b) for b in historical_benchmarks], f, indent=2)
            print(f"💾 Benchmark saved to {self.benchmark_file}")
        except Exception as e:
            print(f"❌ Failed to save benchmark: {e}")

    def detect_performance_issues(self, current: PerformanceBenchmark, historical: List[PerformanceBenchmark]) -> List[PerformanceAlert]:
        """Detect performance issues by comparing with historical data and thresholds."""
        
        alerts = []
        
        # Check against absolute thresholds
        threshold_checks = [
            ("max_processing_time", current.max_processing_time, self.thresholds["max_processing_time"], "high"),
            ("avg_processing_time", current.avg_processing_time_per_case, self.thresholds["avg_processing_time"], "medium"),
            ("success_rate", current.success_rate, self.thresholds["min_success_rate"], "critical"),
            ("error_rate", current.error_rate, self.thresholds["max_error_rate"], "high"),
            ("flags_detection_rate", current.flags_detection_rate, self.thresholds["min_flags_detection"], "medium"),
            ("clusters_detection_rate", current.clusters_detection_rate, self.thresholds["min_clusters_detection"], "medium"),
            ("avg_confidence_score", current.avg_confidence_score, self.thresholds["min_confidence_score"], "low")
        ]
        
        for metric_name, current_value, threshold, severity in threshold_checks:
            if metric_name in ["success_rate", "flags_detection_rate", "clusters_detection_rate", "avg_confidence_score"]:
                # These should be above threshold
                if current_value < threshold:
                    alerts.append(PerformanceAlert(
                        alert_type="degradation",
                        severity=severity,
                        metric_name=metric_name,
                        current_value=current_value,
                        baseline_value=threshold,
                        threshold_exceeded=threshold - current_value,
                        message=f"{metric_name} ({current_value:.2f}) is below threshold ({threshold:.2f})",
                        timestamp=current.timestamp
                    ))
            else:
                # These should be below threshold  
                if current_value > threshold:
                    alerts.append(PerformanceAlert(
                        alert_type="degradation",
                        severity=severity,
                        metric_name=metric_name,
                        current_value=current_value,
                        baseline_value=threshold,
                        threshold_exceeded=current_value - threshold,
                        message=f"{metric_name} ({current_value:.2f}) exceeds threshold ({threshold:.2f})",
                        timestamp=current.timestamp
                    ))
        
        # Check against historical trends (if we have enough data)
        if len(historical) >= 5:
            recent_benchmarks = historical[-5:]  # Last 5 benchmarks
            
            # Calculate historical averages
            historical_avg_processing = statistics.mean([b.avg_processing_time_per_case for b in recent_benchmarks])
            historical_success_rate = statistics.mean([b.success_rate for b in recent_benchmarks])
            historical_confidence = statistics.mean([b.avg_confidence_score for b in recent_benchmarks])
            
            # Check for significant degradation (>20% worse than recent average)
            degradation_checks = [
                ("avg_processing_time", current.avg_processing_time_per_case, historical_avg_processing, "increase"),
                ("success_rate", current.success_rate, historical_success_rate, "decrease"),
                ("avg_confidence_score", current.avg_confidence_score, historical_confidence, "decrease")
            ]
            
            for metric_name, current_value, historical_avg, direction in degradation_checks:
                if direction == "increase" and current_value > historical_avg * 1.2:
                    alerts.append(PerformanceAlert(
                        alert_type="degradation",
                        severity="medium",
                        metric_name=metric_name,
                        current_value=current_value,
                        baseline_value=historical_avg,
                        threshold_exceeded=(current_value - historical_avg) / historical_avg,
                        message=f"{metric_name} increased by {((current_value - historical_avg) / historical_avg * 100):.1f}% vs recent average",
                        timestamp=current.timestamp
                    ))
                elif direction == "decrease" and current_value < historical_avg * 0.8:
                    alerts.append(PerformanceAlert(
                        alert_type="degradation", 
                        severity="medium",
                        metric_name=metric_name,
                        current_value=current_value,
                        baseline_value=historical_avg,
                        threshold_exceeded=(historical_avg - current_value) / historical_avg,
                        message=f"{metric_name} decreased by {((historical_avg - current_value) / historical_avg * 100):.1f}% vs recent average",
                        timestamp=current.timestamp
                    ))
        
        return alerts

    def generate_performance_report(self, current: PerformanceBenchmark, historical: List[PerformanceBenchmark], alerts: List[PerformanceAlert]) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        
        report = {
            "timestamp": current.timestamp,
            "current_benchmark": asdict(current),
            "alerts": [asdict(alert) for alert in alerts],
            "trends": {},
            "recommendations": []
        }
        
        # Calculate trends if we have historical data
        if len(historical) >= 2:
            last_benchmark = historical[-1]
            
            trends = {
                "processing_time_trend": ((current.avg_processing_time_per_case - last_benchmark.avg_processing_time_per_case) / last_benchmark.avg_processing_time_per_case * 100),
                "success_rate_trend": ((current.success_rate - last_benchmark.success_rate) / last_benchmark.success_rate * 100) if last_benchmark.success_rate > 0 else 0,
                "confidence_trend": ((current.avg_confidence_score - last_benchmark.avg_confidence_score) / last_benchmark.avg_confidence_score * 100) if last_benchmark.avg_confidence_score > 0 else 0
            }
            
            report["trends"] = trends
        
        # Generate recommendations based on alerts
        if alerts:
            critical_alerts = [a for a in alerts if a.severity == "critical"]
            high_alerts = [a for a in alerts if a.severity == "high"]
            
            if critical_alerts:
                report["recommendations"].append("URGENT: Critical performance issues detected - immediate investigation required")
            
            if high_alerts:
                report["recommendations"].append("HIGH PRIORITY: Performance degradation detected - review recent changes")
            
            if any(a.metric_name == "avg_processing_time" for a in alerts):
                report["recommendations"].append("Consider optimizing processing pipeline or reviewing configuration")
            
            if any(a.metric_name == "success_rate" for a in alerts):
                report["recommendations"].append("Investigate error patterns and improve error handling")
        else:
            report["recommendations"].append("Performance is within acceptable ranges")
        
        return report

    async def run_continuous_monitoring(self) -> Dict[str, Any]:
        """Run continuous performance monitoring with alerts."""
        
        print("🔍 Running Continuous Performance Monitoring")
        print("=" * 50)
        
        # Run benchmark
        current_benchmark = await self.run_performance_benchmark_suite()
        
        # Load historical data
        historical_benchmarks = self.load_historical_benchmarks()
        
        # Detect issues
        alerts = self.detect_performance_issues(current_benchmark, historical_benchmarks)
        
        # Generate report
        report = self.generate_performance_report(current_benchmark, historical_benchmarks, alerts)
        
        # Save current benchmark
        self.save_benchmark(current_benchmark)
        
        # Print alerts
        if alerts:
            print(f"\n🚨 PERFORMANCE ALERTS ({len(alerts)})")
            print("=" * 50)
            
            for alert in alerts:
                severity_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}
                print(f"{severity_icon.get(alert.severity, '⚪')} {alert.severity.upper()}: {alert.message}")
        else:
            print(f"\n✅ NO PERFORMANCE ISSUES DETECTED")
        
        # Print trends
        if report.get("trends"):
            print(f"\n📈 PERFORMANCE TRENDS")
            print("=" * 50)
            trends = report["trends"]
            
            trend_icons = {
                "processing_time_trend": "⏱️ ",
                "success_rate_trend": "✅",
                "confidence_trend": "🎯"
            }
            
            for trend_name, trend_value in trends.items():
                icon = trend_icons.get(trend_name, "📊")
                direction = "📈" if trend_value > 0 else "📉" if trend_value < 0 else "➡️"
                print(f"   {icon} {trend_name.replace('_', ' ').title()}: {direction} {trend_value:+.1f}%")
        
        # Print recommendations
        print(f"\n💡 RECOMMENDATIONS")
        print("=" * 50)
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"   {i}. {rec}")
        
        return report

async def main():
    """Main function to run performance benchmarking."""
    
    benchmark_tool = LeaseAnalysisPerformanceBenchmark()
    
    # Check LangSmith connection
    try:
        projects = list(benchmark_tool.client.list_projects(limit=1))
        print("✅ LangSmith connection successful")
    except Exception as e:
        print(f"❌ LangSmith connection failed: {e}")
        return
    
    # Run continuous monitoring
    report = await benchmark_tool.run_continuous_monitoring()
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"performance_report_{timestamp}.json"
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n💾 Performance report saved to: {report_file}")

if __name__ == "__main__":
    asyncio.run(main()) 