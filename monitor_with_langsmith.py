"""
LangSmith Monitoring and Testing Script for ReAct Agent

This script helps you:
1. Test your agent with different scenarios
2. Monitor performance with LangSmith
3. Analyze traces and metrics
4. Validate prompt management
"""

import asyncio
import os
from datetime import datetime
from typing import Dict, List, Any
from dotenv import load_dotenv

from langsmith import Client, traceable
from react_agent.graph import graph
from react_agent.configuration import Configuration

# Load environment variables
load_dotenv()

class LangSmithMonitor:
    """Monitor and test the ReAct Agent with LangSmith integration."""
    
    def __init__(self):
        self.client = Client()
        self.project_name = os.getenv("LANGSMITH_PROJECT", "react-agent")
        
    def check_langsmith_connection(self) -> bool:
        """Verify LangSmith connection and configuration."""
        try:
            # Test connection
            projects = list(self.client.list_projects(limit=1))
            print("✅ LangSmith connection successful")
            print(f"📊 Project: {self.project_name}")
            return True
        except Exception as e:
            print(f"❌ LangSmith connection failed: {e}")
            return False
    
    def list_available_prompts(self) -> List[str]:
        """List available prompts in LangSmith."""
        try:
            prompts = list(self.client.list_prompts())
            prompt_names = [p.name for p in prompts]
            print(f"📝 Available prompts: {prompt_names}")
            return prompt_names
        except Exception as e:
            print(f"❌ Failed to list prompts: {e}")
            return []
    
    @traceable(name="test_agent_scenario", metadata={"test_type": "functionality"})
    async def test_agent_scenario(self, query: str, scenario_name: str) -> Dict[str, Any]:
        """Test the agent with a specific scenario."""
        print(f"\n🧪 Testing scenario: {scenario_name}")
        print(f"Query: {query}")
        
        try:
            # Run the agent
            result = await graph.ainvoke(
                {"messages": [("user", query)]},
                {"configurable": {"system_prompt": "You are a helpful AI assistant."}}
            )
            
            final_message = result["messages"][-1].content
            print(f"✅ Response: {final_message[:200]}...")
            
            return {
                "scenario": scenario_name,
                "query": query,
                "response": final_message,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return {
                "scenario": scenario_name,
                "query": query,
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
    
    async def run_comprehensive_tests(self) -> List[Dict[str, Any]]:
        """Run a comprehensive test suite."""
        test_scenarios = [
            {
                "name": "Simple Question",
                "query": "What is the capital of France?"
            },
            {
                "name": "Research Query",
                "query": "I need detailed information about renewable energy trends in 2024"
            },
            {
                "name": "Lease Document Query",
                "query": "Can you help me analyze this commercial lease agreement for valuation purposes?"
            },
            {
                "name": "Tool Usage",
                "query": "Search for the latest news about artificial intelligence developments"
            },
            {
                "name": "Complex Reasoning",
                "query": "Compare the pros and cons of different database systems for a web application"
            }
        ]
        
        print("🚀 Starting comprehensive test suite...")
        results = []
        
        for scenario in test_scenarios:
            result = await self.test_agent_scenario(
                scenario["query"], 
                scenario["name"]
            )
            results.append(result)
            
            # Add delay between tests
            await asyncio.sleep(1)
        
        return results
    
    def analyze_test_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze test results and provide summary."""
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r["success"])
        failure_rate = (total_tests - successful_tests) / total_tests if total_tests > 0 else 0
        
        analysis = {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": total_tests - successful_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            "failure_rate": failure_rate,
            "failed_scenarios": [r["scenario"] for r in results if not r["success"]]
        }
        
        print(f"\n📊 Test Results Summary:")
        print(f"   Total Tests: {analysis['total_tests']}")
        print(f"   ✅ Successful: {analysis['successful_tests']}")
        print(f"   ❌ Failed: {analysis['failed_tests']}")
        print(f"   📈 Success Rate: {analysis['success_rate']:.2%}")
        
        if analysis['failed_scenarios']:
            print(f"   🔍 Failed Scenarios: {', '.join(analysis['failed_scenarios'])}")
        
        return analysis
    
    def get_recent_traces(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent traces from LangSmith."""
        try:
            runs = list(self.client.list_runs(
                project_name=self.project_name,
                limit=limit
            ))
            
            trace_data = []
            for run in runs:
                trace_data.append({
                    "id": str(run.id),
                    "name": run.name,
                    "status": run.status,
                    "start_time": run.start_time,
                    "end_time": run.end_time,
                    "duration": (run.end_time - run.start_time).total_seconds() if run.end_time else None,
                    "inputs": run.inputs,
                    "outputs": run.outputs
                })
            
            print(f"\n🔍 Recent Traces ({len(trace_data)}):")
            for trace in trace_data[:5]:  # Show top 5
                duration = f"{trace['duration']:.2f}s" if trace['duration'] else "N/A"
                print(f"   📝 {trace['name']}: {trace['status']} ({duration})")
            
            return trace_data
        except Exception as e:
            print(f"❌ Failed to get traces: {e}")
            return []
    
    async def validate_prompt_management(self) -> bool:
        """Validate that prompt management is working correctly."""
        print("\n🔧 Validating prompt management...")
        
        try:
            # Try to pull the lease document prompt
            prompt = self.client.pull_prompt("v22-prompt1-establish-baseline-terms")
            print("✅ Successfully pulled lease document prompt")
            
            # Test the KB agent with lease query
            result = await self.test_agent_scenario(
                "Help me with lease document analysis",
                "Prompt Management Test"
            )
            
            return result["success"]
        except Exception as e:
            print(f"❌ Prompt management validation failed: {e}")
            return False

async def main():
    """Main function to run the monitoring suite."""
    print("🚀 LangSmith ReAct Agent Monitor")
    print("=" * 50)
    
    monitor = LangSmithMonitor()
    
    # 1. Check LangSmith connection
    if not monitor.check_langsmith_connection():
        print("❌ Please check your LangSmith configuration and try again.")
        return
    
    # 2. List available prompts
    monitor.list_available_prompts()
    
    # 3. Validate prompt management
    await monitor.validate_prompt_management()
    
    # 4. Run comprehensive tests
    results = await monitor.run_comprehensive_tests()
    
    # 5. Analyze results
    analysis = monitor.analyze_test_results(results)
    
    # 6. Get recent traces
    monitor.get_recent_traces()
    
    # 7. Provide recommendations
    print("\n💡 Recommendations:")
    if analysis["success_rate"] >= 0.8:
        print("   ✅ Your agent is performing well!")
    else:
        print("   ⚠️  Consider reviewing failed scenarios and improving prompts")
    
    print("   🔗 View detailed traces at: https://smith.langchain.com/")
    print(f"   📊 Project: {monitor.project_name}")

if __name__ == "__main__":
    asyncio.run(main()) 