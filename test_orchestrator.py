#!/usr/bin/env python3
"""
Comprehensive Testing Orchestrator for Lease Analysis System

This orchestrator manages:
1. Unit and integration tests
2. LangSmith evaluations
3. A/B testing workflows
4. Performance benchmarking
5. Regression testing
6. Continuous improvement pipelines
"""

import asyncio
import subprocess
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class TestResult:
    """Test result information."""
    
    test_type: str
    success: bool
    duration: float
    details: Dict[str, Any]
    timestamp: str
    
@dataclass 
class TestingPipeline:
    """Testing pipeline configuration."""
    
    name: str
    description: str
    tests: List[str]
    required_env_vars: List[str]
    success_criteria: Dict[str, float]

class LeaseAnalysisTestOrchestrator:
    """Comprehensive testing orchestrator for the lease analysis system."""
    
    def __init__(self):
        self.results_dir = "test_results"
        self.ensure_results_directory()
        
    def ensure_results_directory(self):
        """Ensure test results directory exists."""
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
    
    def check_environment_setup(self) -> Dict[str, bool]:
        """Check if the environment is properly set up for testing."""
        
        required_vars = [
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY", 
            "TAVILY_API_KEY",
            "LANGCHAIN_API_KEY"
        ]
        
        optional_vars = [
            "LANGCHAIN_TRACING_V2",
            "LANGSMITH_PROJECT"
        ]
        
        env_status = {}
        
        print("🔍 Checking Environment Setup")
        print("=" * 40)
        
        # Check required variables (at least one AI provider)
        has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
        has_openai = bool(os.getenv("OPENAI_API_KEY"))
        has_tavily = bool(os.getenv("TAVILY_API_KEY"))
        has_langsmith = bool(os.getenv("LANGCHAIN_API_KEY"))
        
        env_status["ai_provider"] = has_anthropic or has_openai
        env_status["tavily"] = has_tavily
        env_status["langsmith"] = has_langsmith
        
        print(f"   🤖 AI Provider: {'✅' if env_status['ai_provider'] else '❌'} ({'Anthropic' if has_anthropic else ''}{'OpenAI' if has_openai else ''})")
        print(f"   🔍 Tavily Search: {'✅' if env_status['tavily'] else '❌'}")
        print(f"   📊 LangSmith: {'✅' if env_status['langsmith'] else '⚠️  (Optional)'}")
        
        # Check tracing
        tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
        env_status["tracing"] = tracing_enabled
        print(f"   📈 Tracing: {'✅' if tracing_enabled else '⚠️  (Optional)'}")
        
        return env_status
    
    async def run_subprocess_test(self, command: str, test_name: str) -> TestResult:
        """Run a subprocess test and capture results."""
        
        print(f"\n🧪 Running {test_name}")
        print(f"   Command: {command}")
        
        start_time = datetime.now()
        
        try:
            # Run the command
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.getcwd()
            )
            
            stdout, stderr = await process.communicate()
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            success = process.returncode == 0
            
            result = TestResult(
                test_type=test_name,
                success=success,
                duration=duration,
                details={
                    "command": command,
                    "return_code": process.returncode,
                    "stdout": stdout.decode('utf-8') if stdout else "",
                    "stderr": stderr.decode('utf-8') if stderr else ""
                },
                timestamp=start_time.isoformat()
            )
            
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"   {status} ({duration:.2f}s)")
            
            if not success and stderr:
                print(f"   Error: {stderr.decode('utf-8')[:200]}...")
            
            return result
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print(f"   ❌ FAILED ({duration:.2f}s): {str(e)}")
            
            return TestResult(
                test_type=test_name,
                success=False,
                duration=duration,
                details={
                    "command": command,
                    "error": str(e)
                },
                timestamp=start_time.isoformat()
            )
    
    def define_testing_pipelines(self) -> List[TestingPipeline]:
        """Define different testing pipelines for various scenarios."""
        
        return [
            TestingPipeline(
                name="unit_tests",
                description="Core unit tests for basic functionality",
                tests=["python -m pytest tests/unit_tests/ -v"],
                required_env_vars=[],
                success_criteria={"pass_rate": 1.0}
            ),
            
            TestingPipeline(
                name="integration_tests", 
                description="Integration tests with API dependencies",
                tests=["python -m pytest tests/integration_tests/ -v"],
                required_env_vars=["ANTHROPIC_API_KEY", "TAVILY_API_KEY"],
                success_criteria={"pass_rate": 0.95}
            ),
            
            TestingPipeline(
                name="comprehensive_evaluation",
                description="Comprehensive lease analysis evaluation with LangSmith",
                tests=["python langsmith_lease_evaluator.py"],
                required_env_vars=["ANTHROPIC_API_KEY", "TAVILY_API_KEY", "LANGCHAIN_API_KEY"],
                success_criteria={"avg_overall_score": 0.70}
            ),
            
            TestingPipeline(
                name="performance_benchmark",
                description="Performance benchmarking and monitoring",
                tests=["python langsmith_performance_benchmark.py"],
                required_env_vars=["ANTHROPIC_API_KEY", "TAVILY_API_KEY"],
                success_criteria={"avg_processing_time": 30.0, "success_rate": 0.95}
            ),
            
            TestingPipeline(
                name="ab_testing",
                description="A/B testing for configuration optimization",
                tests=["python langsmith_ab_testing.py"],
                required_env_vars=["ANTHROPIC_API_KEY", "TAVILY_API_KEY"],
                success_criteria={"completion_rate": 1.0}
            ),
            
            TestingPipeline(
                name="monitoring_check",
                description="LangSmith monitoring and dashboard checks",
                tests=[
                    "python monitor_with_langsmith.py",
                    "echo 'y' | python check_langsmith_dashboard.py"
                ],
                required_env_vars=["LANGCHAIN_API_KEY"],
                success_criteria={"completion_rate": 1.0}
            )
        ]
    
    async def run_pipeline(self, pipeline: TestingPipeline, env_status: Dict[str, bool]) -> List[TestResult]:
        """Run a specific testing pipeline."""
        
        print(f"\n🚀 PIPELINE: {pipeline.name.upper()}")
        print(f"   Description: {pipeline.description}")
        print("=" * 60)
        
        # Check if required environment variables are available
        missing_env = []
        for env_var in pipeline.required_env_vars:
            if env_var == "ANTHROPIC_API_KEY" or env_var == "OPENAI_API_KEY":
                if not env_status.get("ai_provider", False):
                    missing_env.append("AI Provider (Anthropic or OpenAI)")
            elif env_var == "TAVILY_API_KEY" and not env_status.get("tavily", False):
                missing_env.append("Tavily API Key")
            elif env_var == "LANGCHAIN_API_KEY" and not env_status.get("langsmith", False):
                missing_env.append("LangSmith API Key")
        
        if missing_env:
            print(f"   ⏭️  SKIPPED: Missing required environment: {', '.join(missing_env)}")
            return []
        
        results = []
        
        for test_command in pipeline.tests:
            result = await self.run_subprocess_test(test_command, f"{pipeline.name}_test")
            results.append(result)
        
        return results
    
    def evaluate_pipeline_results(self, pipeline: TestingPipeline, results: List[TestResult]) -> Dict[str, Any]:
        """Evaluate if pipeline results meet success criteria."""
        
        if not results:
            return {
                "pipeline": pipeline.name,
                "success": False,
                "reason": "No tests run (missing environment)",
                "metrics": {}
            }
        
        successful_tests = [r for r in results if r.success]
        total_tests = len(results)
        
        metrics = {
            "total_tests": total_tests,
            "successful_tests": len(successful_tests),
            "pass_rate": len(successful_tests) / total_tests if total_tests > 0 else 0.0,
            "avg_duration": sum(r.duration for r in results) / total_tests if total_tests > 0 else 0.0,
            "completion_rate": 1.0 if all(r.duration > 0 for r in results) else 0.0
        }
        
        # Check against success criteria
        meets_criteria = True
        criteria_details = {}
        
        for criterion, threshold in pipeline.success_criteria.items():
            actual_value = metrics.get(criterion, 0.0)
            
            if criterion in ["avg_processing_time"]:
                # Lower is better
                criterion_met = actual_value <= threshold
            else:
                # Higher is better
                criterion_met = actual_value >= threshold
                
            meets_criteria = meets_criteria and criterion_met
            criteria_details[criterion] = {
                "threshold": threshold,
                "actual": actual_value,
                "met": criterion_met
            }
        
        return {
            "pipeline": pipeline.name,
            "success": meets_criteria,
            "metrics": metrics,
            "criteria_evaluation": criteria_details
        }
    
    async def run_comprehensive_testing(self, selected_pipelines: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run comprehensive testing across all or selected pipelines."""
        
        print("🚀 COMPREHENSIVE LEASE ANALYSIS TESTING ORCHESTRATOR")
        print("=" * 65)
        
        # Check environment
        env_status = self.check_environment_setup()
        
        # Get pipelines
        all_pipelines = self.define_testing_pipelines()
        
        if selected_pipelines:
            pipelines_to_run = [p for p in all_pipelines if p.name in selected_pipelines]
        else:
            pipelines_to_run = all_pipelines
        
        print(f"\n📋 Running {len(pipelines_to_run)} testing pipelines...")
        
        # Run each pipeline
        all_results = {}
        pipeline_evaluations = {}
        
        for pipeline in pipelines_to_run:
            pipeline_results = await self.run_pipeline(pipeline, env_status)
            all_results[pipeline.name] = pipeline_results
            
            evaluation = self.evaluate_pipeline_results(pipeline, pipeline_results)
            pipeline_evaluations[pipeline.name] = evaluation
        
        # Generate summary
        summary = self.generate_testing_summary(pipeline_evaluations, env_status)
        
        # Save results
        await self.save_test_results(all_results, pipeline_evaluations, summary)
        
        # Print summary
        self.print_testing_summary(summary)
        
        return {
            "summary": summary,
            "pipeline_evaluations": pipeline_evaluations,
            "detailed_results": all_results
        }
    
    def generate_testing_summary(self, evaluations: Dict[str, Any], env_status: Dict[str, bool]) -> Dict[str, Any]:
        """Generate comprehensive testing summary."""
        
        total_pipelines = len(evaluations)
        successful_pipelines = sum(1 for e in evaluations.values() if e["success"])
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "environment_status": env_status,
            "total_pipelines": total_pipelines,
            "successful_pipelines": successful_pipelines,
            "pipeline_success_rate": successful_pipelines / total_pipelines if total_pipelines > 0 else 0.0,
            "individual_results": evaluations,
            "recommendations": []
        }
        
        # Generate recommendations
        if summary["pipeline_success_rate"] >= 0.8:
            summary["recommendations"].append("✅ System is performing well across most test pipelines")
        else:
            summary["recommendations"].append("⚠️  System has significant issues that need attention")
        
        # Environment-specific recommendations
        if not env_status.get("langsmith", False):
            summary["recommendations"].append("💡 Configure LangSmith for enhanced monitoring and evaluation")
        
        if not env_status.get("tracing", False):
            summary["recommendations"].append("💡 Enable tracing for better debugging and optimization")
        
        # Pipeline-specific recommendations
        failed_pipelines = [name for name, eval_data in evaluations.items() if not eval_data["success"]]
        if failed_pipelines:
            summary["recommendations"].append(f"🔧 Review and fix issues in: {', '.join(failed_pipelines)}")
        
        return summary
    
    async def save_test_results(self, results: Dict[str, List[TestResult]], evaluations: Dict[str, Any], summary: Dict[str, Any]):
        """Save test results to files."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = f"{self.results_dir}/test_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            # Convert TestResult objects to dicts for JSON serialization
            serializable_results = {}
            for pipeline_name, pipeline_results in results.items():
                serializable_results[pipeline_name] = [
                    {
                        "test_type": r.test_type,
                        "success": r.success,
                        "duration": r.duration,
                        "details": r.details,
                        "timestamp": r.timestamp
                    }
                    for r in pipeline_results
                ]
            
            json.dump({
                "results": serializable_results,
                "evaluations": evaluations,
                "summary": summary
            }, f, indent=2, default=str)
        
        print(f"\n💾 Test results saved to: {results_file}")
    
    def print_testing_summary(self, summary: Dict[str, Any]):
        """Print comprehensive testing summary."""
        
        print(f"\n📊 TESTING SUMMARY")
        print("=" * 65)
        
        print(f"\n📈 Overall Results:")
        print(f"   Total Pipelines: {summary['total_pipelines']}")
        print(f"   Successful: {summary['successful_pipelines']}")
        print(f"   Success Rate: {summary['pipeline_success_rate']:.2%}")
        
        print(f"\n🔧 Environment Status:")
        env_status = summary["environment_status"]
        print(f"   AI Provider: {'✅' if env_status.get('ai_provider') else '❌'}")
        print(f"   Tavily Search: {'✅' if env_status.get('tavily') else '❌'}")
        print(f"   LangSmith: {'✅' if env_status.get('langsmith') else '⚠️ '}")
        print(f"   Tracing: {'✅' if env_status.get('tracing') else '⚠️ '}")
        
        print(f"\n📋 Pipeline Results:")
        for pipeline_name, evaluation in summary["individual_results"].items():
            status = "✅" if evaluation["success"] else "❌"
            metrics = evaluation.get("metrics", {})
            pass_rate = metrics.get("pass_rate", 0.0)
            duration = metrics.get("avg_duration", 0.0)
            print(f"   {status} {pipeline_name}: {pass_rate:.1%} pass rate ({duration:.2f}s avg)")
        
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(summary["recommendations"], 1):
            print(f"   {i}. {rec}")

async def main():
    """Main function for testing orchestrator."""
    
    import sys
    
    orchestrator = LeaseAnalysisTestOrchestrator()
    
    # Parse command line arguments for selective testing
    selected_pipelines = None
    if len(sys.argv) > 1:
        selected_pipelines = sys.argv[1].split(',')
        print(f"🎯 Running selected pipelines: {', '.join(selected_pipelines)}")
    
    # Run comprehensive testing
    results = await orchestrator.run_comprehensive_testing(selected_pipelines)
    
    # Exit with appropriate code
    success_rate = results["summary"]["pipeline_success_rate"]
    exit_code = 0 if success_rate >= 0.8 else 1
    
    if exit_code == 0:
        print(f"\n🎉 TESTING COMPLETED SUCCESSFULLY!")
    else:
        print(f"\n⚠️  TESTING COMPLETED WITH ISSUES")
    
    sys.exit(exit_code)

if __name__ == "__main__":
    asyncio.run(main()) 