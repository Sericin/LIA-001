"""
Quick LangSmith Dashboard Check

This script provides a quick overview of your LangSmith project status.
"""

import os
import webbrowser
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

def quick_dashboard_check():
    """Perform a quick check of LangSmith dashboard metrics."""
    
    client = Client()
    project_name = os.getenv("LANGSMITH_PROJECT", "react-agent")
    
    print("🔍 LangSmith Dashboard Quick Check")
    print("=" * 40)
    
    try:
        # Check recent runs (last 24 hours)
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)
        
        recent_runs = list(client.list_runs(
            project_name=project_name,
            start_time=start_time,
            limit=50
        ))
        
        print(f"📊 Project: {project_name}")
        print(f"📈 Runs in last 24h: {len(recent_runs)}")
        
        if recent_runs:
            # Calculate success rate
            successful_runs = sum(1 for run in recent_runs if run.status == "success")
            success_rate = successful_runs / len(recent_runs)
            
            # Calculate average duration
            durations = []
            for run in recent_runs:
                if run.end_time and run.start_time:
                    duration = (run.end_time - run.start_time).total_seconds()
                    durations.append(duration)
            
            avg_duration = sum(durations) / len(durations) if durations else 0
            
            print(f"✅ Success rate: {success_rate:.1%}")
            print(f"⏱️  Average duration: {avg_duration:.2f}s")
            
            # Show recent run types
            run_types = {}
            for run in recent_runs[:10]:  # Last 10 runs
                run_type = run.name or "unknown"
                run_types[run_type] = run_types.get(run_type, 0) + 1
            
            print("\n📋 Recent run types:")
            for run_type, count in run_types.items():
                print(f"   {run_type}: {count}")
                
        else:
            print("⚠️  No recent runs found")
        
        # Generate dashboard URL
        dashboard_url = f"https://smith.langchain.com/o/default/projects/p/{project_name}"
        print(f"\n🔗 Dashboard URL: {dashboard_url}")
        
        # Ask if user wants to open dashboard
        response = input("\n🌐 Open LangSmith dashboard in browser? (y/n): ").lower().strip()
        if response == 'y':
            webbrowser.open(dashboard_url)
            print("✅ Dashboard opened in browser")
        
    except Exception as e:
        print(f"❌ Error checking dashboard: {e}")
        print("💡 Make sure your LANGSMITH_API_KEY is set correctly")

if __name__ == "__main__":
    quick_dashboard_check() 