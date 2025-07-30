# Complete Testing Guide for LangGraph Studio

## 🎯 Overview
This multi-agent system includes:
- **Main ReAct Agent**: General conversation and routing
- **Research Agent**: Information gathering specialist  
- **KB Lease Doc Agent**: Commercial real estate lease analysis
- **Reviewer Agent**: Quality control with legal/ambiguity/clustering analysis
- **Advanced Analytics**: Legal uncertainty detection, provision clustering, confidence calibration

## 📋 Pre-Testing Setup

### 1. Environment Configuration
Create a `.env` file with:
```bash
# AI Model Provider (Choose one)
ANTHROPIC_API_KEY=your-anthropic-api-key-here
# OR  
OPENAI_API_KEY=your-openai-api-key-here

# Search Functionality
TAVILY_API_KEY=your-tavily-api-key-here

# LangSmith Integration (for monitoring)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your-langsmith-api-key-here
LANGCHAIN_PROJECT=react-agent-testing
```

### 2. Install and Launch LangGraph Studio
```bash
# Ensure dependencies are installed
pip install -e .

# Open in LangGraph Studio
# Point studio to this project directory
```

## 🧪 Comprehensive Testing Scenarios

### **Phase 1: Basic Agent Flow Testing**

#### Test 1: Simple Conversation Flow
**Input**: "Hello, can you help me understand your capabilities?"
**Expected**: Main agent responds, no routing to specialized agents
**Validation**: Check response completeness and clarity

#### Test 2: Research Agent Routing  
**Input**: "I need detailed information about commercial real estate market trends"
**Expected**: Routes to research_agent → uses search tools → provides comprehensive info
**Validation**: Verify research quality and source citations

#### Test 3: Tool Usage Testing
**Input**: "Search for the latest news about AI in real estate"
**Expected**: Uses Tavily search tool → processes results → provides summary
**Validation**: Check tool call success and result integration

### **Phase 2: Lease Analysis Testing**

#### Test 4: Basic Lease Analysis
**Input**: "Please analyze this commercial lease document: [paste simple lease text]"
**Expected**: Routes to kb_lease_doc_agent → processes with v22-prompt1-establish-baseline-terms
**Validation**: Check for baseline terms identification

#### Test 5: Complex Lease with Review
**Input**: Complex lease with ambiguous terms and onerous provisions
**Expected**: 
1. kb_lease_doc_agent completes primary analysis
2. Automatically routes to reviewer_agent
3. Reviewer performs multi-phase analysis
4. Returns comprehensive findings with flags/clusters/confidence metrics

**Sample Complex Lease Input**:
```
COMMERCIAL LEASE AGREEMENT

Property: 123 Business Plaza, Suite 500
Tenant: ABC Corp
Landlord: XYZ Properties

TERM: Five (5) years commencing on a date to be determined by Landlord

RENT: Base rent of $25 per square foot annually, subject to annual increases at Landlord's discretion up to 15%, plus additional charges as deemed necessary by Landlord

MAINTENANCE: Tenant responsible for all repairs, maintenance, and improvements to the premises, building systems, parking areas, and surrounding grounds, regardless of cause or necessity

DEFAULT: Any delay in rent payment beyond 24 hours constitutes default, entitling Landlord to immediate termination and damages equal to remaining lease term payments

ASSIGNMENT: Tenant may not assign or sublet without Landlord's consent, which may be withheld for any reason or no reason

INSURANCE: Tenant must maintain comprehensive insurance acceptable to Landlord, with limits as determined by Landlord from time to time
```

**Validation Checklist**:
- [ ] Legal uncertainties flagged (vague dates, discretionary increases)
- [ ] Ambiguities identified (undefined terms, discretionary clauses) 
- [ ] Onerous provisions clustered (maintenance burden + default terms)
- [ ] Confidence calibration reflects uncertainty levels
- [ ] Recommendations include legal counsel suggestion

### **Phase 3: Configuration Testing**

#### Test 6: Review Depth Levels
Test different configuration settings:

**Basic Level**:
```json
{"review_depth_level": "basic", "enable_legal_uncertainty_detection": false}
```

**Comprehensive Level**:
```json
{
  "review_depth_level": "comprehensive",
  "enable_legal_uncertainty_detection": true,
  "enable_ambiguity_detection": true, 
  "enable_onerous_clustering": true,
  "enable_confidence_calibration": true
}
```

#### Test 7: Confidence Calibration Strategies
Test each strategy with the same input:
- `"confidence_calibration_strategy": "conservative"`  
- `"confidence_calibration_strategy": "balanced"`
- `"confidence_calibration_strategy": "aggressive"`

**Validation**: Different confidence scores and uncertainty ranges

#### Test 8: Flag Sensitivity Testing
```json
{
  "flag_sensitivity_legal": "high",
  "flag_sensitivity_ambiguity": "high", 
  "max_flags_per_analysis": 30
}
```
**Expected**: More flags detected with higher sensitivity

### **Phase 4: Error Handling & Edge Cases**

#### Test 9: Empty/Invalid Inputs
- Empty message
- Non-lease document text
- Extremely long input (>10,000 characters)
- Special characters and formatting

#### Test 10: API Failures
- Test with invalid API keys
- Test with network interruptions
- Test timeout scenarios

#### Test 11: Missing Dependencies
- Test without LangSmith connection
- Test without Tavily API key
- Test prompt pulling failures

### **Phase 5: Performance & Monitoring**

#### Test 12: LangSmith Integration
**Setup**: Ensure LANGCHAIN_TRACING_V2=true
**Action**: Run any test scenario
**Validation**: 
- Check LangSmith dashboard for trace visibility
- Verify all agents show up as separate traces
- Confirm metadata and tags are properly set

#### Test 13: Processing Time Monitoring
Use the built-in monitoring:
```bash
make monitor  # Run comprehensive monitoring
make dashboard  # Quick dashboard check
```

#### Test 14: Concurrent User Testing
- Open multiple LangGraph Studio sessions
- Submit different queries simultaneously  
- Monitor performance degradation

### **Phase 6: Advanced Testing Scenarios**

#### Test 15: Multi-Turn Conversations
**Conversation Flow**:
1. "Analyze this lease: [simple lease]"
2. "What are the biggest risks in this lease?"
3. "Should I negotiate different terms?"
4. "What would be a fair market rent adjustment?"

**Validation**: Context maintenance across turns

#### Test 16: Complex Document Types
Test with various document formats:
- Standard office lease
- Retail lease with percentage rent
- Industrial lease with CAM charges
- Ground lease with development rights
- License agreement vs. lease distinction

#### Test 17: Interruption and Resume Testing
1. Start complex lease analysis
2. Interrupt mid-processing  
3. Try to resume or start new analysis
4. Verify state management

### **Phase 7: Integration Testing**

#### Test 18: End-to-End Workflow
**Complete Business Scenario**:
1. User uploads lease document
2. System performs initial analysis
3. Reviewer identifies issues
4. User asks follow-up questions
5. Research agent gathers market data
6. Final recommendations provided

#### Test 19: Regression Testing
Run the existing test suite:
```bash
make test                    # Unit tests
make test_langsmith         # Integration tests  
make extended_tests         # Extended test suite
```

## 🔍 Validation Checklist

### **Agent Routing Validation**
- [ ] Correct agent selection based on query content
- [ ] Proper fallback to main agent when specialized agents complete
- [ ] No infinite loops or incorrect routing

### **Analysis Quality Validation**  
- [ ] Lease terms properly identified and categorized
- [ ] Legal uncertainties accurately flagged
- [ ] Ambiguities appropriately detected
- [ ] Provision clusters logically grouped
- [ ] Confidence scores reflect actual uncertainty

### **Configuration Validation**
- [ ] All configuration options work as intended
- [ ] Changes in config produce expected behavioral changes
- [ ] Invalid configurations handled gracefully

### **Performance Validation**
- [ ] Response times under 2 minutes for complex analysis
- [ ] Memory usage remains stable during long sessions
- [ ] No resource leaks or performance degradation

### **Error Handling Validation**
- [ ] Graceful degradation when APIs unavailable
- [ ] Clear error messages for user issues
- [ ] System remains stable after errors

## 📊 LangSmith Studio Testing Features

### Built-in Testing Capabilities
1. **Thread Management**: Create new threads for different test scenarios
2. **State Inspection**: View internal state at each step
3. **Agent Debugging**: Step through agent decisions
4. **Configuration Hot-Reload**: Test config changes without restart
5. **Performance Profiling**: Monitor token usage and latency

### Advanced Studio Features
1. **Interrupt Points**: Add interrupts before tool calls for debugging
2. **State Editing**: Modify state mid-conversation for testing edge cases  
3. **Time Travel**: Go back to previous states and retry from there
4. **Trace Analysis**: Deep dive into LangSmith traces directly in studio

## 🎯 Test Success Criteria

### **Functional Success**
- All agents route correctly based on input
- Lease analysis produces comprehensive, accurate results
- Reviewer agent identifies real issues without false positives
- Configuration changes produce expected behavioral differences

### **Quality Success** 
- Analysis results are professionally formatted and actionable
- Legal flags are relevant and properly categorized
- Confidence scores align with actual uncertainty levels
- Recommendations are specific and valuable

### **Performance Success**
- Complex analysis completes within reasonable time
- System handles multiple concurrent users
- Memory and CPU usage remain stable
- LangSmith integration provides useful debugging data

### **Robustness Success**
- System handles invalid inputs gracefully  
- API failures don't crash the system
- Edge cases are managed appropriately
- User experience remains smooth during errors

## 🔧 Debugging Tips

### **Common Issues and Solutions**
1. **Agent Not Routing**: ✅ **FIXED** - The routing now checks both AI responses AND conversation history for keywords
   - Keywords for lease analysis: "lease", "rental", "valuation", "commercial real estate", "tenant", "landlord", "rent"
   - Keywords for research: "research", "information", "details", "market trends", "search for"
2. **Limited Flag Detection**: ✅ **FIXED** - Enhanced legal uncertainty patterns now detect:
   - Sole/absolute discretion language
   - "For any reason or no reason" clauses  
   - "Regardless of" risk shifting
   - Immediate action without notice/cure
   - Rights waivers and tenant expense burdens
   - Penalty provisions and environmental liability
3. **Missing Analysis Results**: Verify API keys and prompt pulling
4. **Low Confidence Scores**: Adjust calibration strategy or thresholds  
5. **Performance Issues**: Check LangSmith traces for bottlenecks
6. **Tool Failures**: Verify Tavily API key and connectivity

### **Routing Troubleshooting**
If agents aren't routing properly:
- Check that your input contains relevant keywords (see list above)
- Verify the conversation context is being captured
- Use `test_routing_fix.py` to test routing logic offline

### **LangGraph Studio Debugging**
- Use the graph visualization to understand flow
- Check individual node outputs in the state panel
- Monitor token usage in the metrics panel
- Use interrupts to pause execution at critical points

## 📈 Continuous Testing Strategy

### **Daily Testing**
- Run basic conversation flow tests
- Check API connectivity and response quality
- Monitor LangSmith dashboard for issues

### **Weekly Testing**  
- Execute full test suite with `make test`
- Test complex lease analysis scenarios
- Validate configuration changes

### **Monthly Testing**
- Performance benchmarking with large documents
- Comprehensive regression testing
- User acceptance testing with real lease documents 