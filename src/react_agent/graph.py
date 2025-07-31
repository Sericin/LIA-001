"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from react_agent.confidence_calibrator import calibrate_confidence
from react_agent.clustering_analyzer import analyze_provision_clusters
from react_agent.ambiguity_analyzer import analyze_ambiguities
from react_agent.legal_analyzer import analyze_legal_uncertainties
from react_agent import prompts
from react_agent.utils import load_chat_model
from react_agent.tools import TOOLS
from react_agent.state import InputState, State, ReviewerFindings, ReviewStatus, ConfidenceMetrics, ReviewMetrics
from react_agent.configuration import Configuration
from langsmith import Client, traceable
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph
from langchain_core.messages import AIMessage, HumanMessage
from datetime import UTC, datetime
from typing import Dict, List, Literal, cast, Any
import os
import time
import asyncio
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Configure logging for better debugging and monitoring
logger = logging.getLogger(__name__)

# Performance monitoring utility


def track_performance(func_name: str, start_time: float, additional_metrics: dict = None):
    """Track performance metrics for agents."""
    duration = time.time() - start_time
    logger.info(f"{func_name} completed in {duration:.2f}s")
    if additional_metrics:
        for key, value in additional_metrics.items():
            logger.info(f"{func_name} - {key}: {value}")
    return duration

# Define the function that calls the model


def create_error_response(error: Exception, agent_name: str, context: str = "") -> AIMessage:
    """Create a consistent error response across all agents."""
    error_msg = str(error)[:200]  # Limit error message length
    content = f"{agent_name} encountered an issue"
    if context:
        content += f" during {context}"
    content += f": {error_msg}..."
    if len(str(error)) > 200:
        content += " (truncated)"
    content += " Please try rephrasing your request or contact support if the issue persists."

    logger.error(f"{agent_name} error in {context}: {error}")
    return AIMessage(content=content)


@traceable(name="call_model", metadata={"node_type": "reasoning"})
async def call_model(state: State) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    configuration = Configuration.from_context()

    # Initialize the model with tool binding. Change the model or add more tools here.
    model = load_chat_model(configuration.model).bind_tools(TOOLS)

    # Format the system prompt. Customize this to change the agent's behavior.
    system_message = configuration.system_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    # Get the model's response
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages]
        ),
    )

    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    # Return the model's response as a list to be added to existing messages
    return {"messages": [response]}

# Add this new research agent function


@traceable(name="research_agent", metadata={"node_type": "research", "specialization": "information_gathering"})
async def research_agent(state: State) -> Dict[str, List[AIMessage]]:
    """Enhanced research agent that specializes in gathering comprehensive information with tool access."""
    configuration = Configuration.from_context()

    # Bind tools to research agent for actual research capabilities
    model = load_chat_model(configuration.model).bind_tools(TOOLS)

    # Enhanced research prompt with specific instructions
    research_prompt = f"""You are a specialized research agent with access to web search tools. Your expertise includes:

- Market research and analysis
- Legal precedent and regulatory information
- Commercial real estate market trends
- Industry standards and benchmarks
- Technical documentation and expert resources

**Research Guidelines:**
- Always use search tools when specific information is requested
- Provide comprehensive, well-sourced information with citations
- Cross-reference multiple sources when possible
- Distinguish between verified facts and general information
- Include relevant context and background information

**Search Strategy:**
- Break complex queries into focused searches
- Use specific terminology and industry keywords
- Search for recent developments and current market conditions
- Look for authoritative sources and expert opinions

System time: {datetime.now(tz=UTC).isoformat()}"""

    try:
        response = cast(
            AIMessage,
            await model.ainvoke([
                {"role": "system", "content": research_prompt},
                *state.messages
            ])
        )

        # If research agent was called but no tools were used, suggest what could be researched
        if not response.tool_calls and not any("search" in msg.content.lower() for msg in state.messages[-3:]):
            # Enhance response with research suggestions
            enhanced_content = response.content
            if isinstance(enhanced_content, str):
                enhanced_content += "\n\n💡 **Research Capabilities Available:** I can search for current market data, legal precedents, industry standards, or any specific information that would help with your analysis. Just let me know what you'd like me to research!"

            response = AIMessage(
                id=response.id,
                content=enhanced_content,
                tool_calls=response.tool_calls
            )

        return {"messages": [response]}

    except Exception as e:
        # Graceful error handling with informative response
        error_response = create_error_response(
            e, "Research Agent", "research_agent")
        return {"messages": [error_response]}

# Add this new kb and lease document processing agent function


@traceable(name="kb_lease_doc_agent", metadata={"node_type": "domain_expert", "specialization": "real_estate_leases"})
async def kb_lease_doc_agent(state: State) -> Dict[str, List[AIMessage]]:
    """KB and lease document processing agent using LangSmith prompt."""

    configuration = Configuration.from_context()
    model = load_chat_model(configuration.model)
    client = Client()

    # Extract lease content for analysis (outside try/catch for scope)
    lease_content = ""
    for msg in state.messages:
        if hasattr(msg, 'content') and isinstance(msg.content, str):
            content_lower = msg.content.lower()
            if any(term in content_lower for term in ["lease", "rental", "tenant", "landlord"]):
                lease_content = msg.content
                break

    # Use comprehensive direct prompt (LangSmith integration optional)
    try:
        # Try LangSmith first, but fallback is robust
        prompt_template = client.pull_prompt(
            "v22-prompt1-establish-baseline-terms")

        formatted_prompt = prompt_template.invoke({
            "document_file": lease_content or "Please provide a lease document to analyze.",
            "kb_materials": "Commercial real estate lease analysis knowledge base. Focus on identifying key terms, potential issues, rent structures, maintenance obligations, default provisions, and overall fairness. Provide a comprehensive analysis that identifies any problematic clauses."
        })

        response = cast(
            AIMessage,
            await model.ainvoke(formatted_prompt.messages)
        )
        print("✅ Using LangSmith prompt")

    except Exception as e:
        # Robust fallback - actually works better than LangSmith attempt
        print(f"⚠️ LangSmith unavailable: {str(e)[:100]}...")
        print("🔄 Using comprehensive fallback prompt")
        fallback_system_prompt = """You are a commercial real estate expert specializing in lease document analysis and valuation. 
        
Analyze the provided commercial lease document thoroughly. Focus on:
1. Key lease terms and conditions
2. Rent structure and escalation clauses  
3. Maintenance and repair obligations
4. Default provisions and penalties
5. Assignment and subletting restrictions
6. Insurance and liability requirements
7. Potential problematic or onerous clauses
8. Overall fairness and risks for the tenant
9. Recommendations for negotiation or legal review

Provide a comprehensive, detailed analysis of all significant lease provisions."""

        # Create focused analysis request
        analysis_messages = [
            {"role": "system", "content": fallback_system_prompt},
            {"role": "user", "content": f"Please provide a comprehensive analysis of this commercial lease:\n\n{lease_content}"}
        ]

        response = cast(
            AIMessage,
            await model.ainvoke(analysis_messages)
        )

    # lease_content already extracted above for prompt parameters

    # Store primary analysis results in state for reviewer
    primary_results = {
        "analysis_completed": True,
        "analysis_timestamp": datetime.now(tz=UTC).isoformat(),
        "analysis_content": response.content,  # AI's analysis
        # Original lease text for sophisticated analysis
        "original_lease_content": lease_content,
        "analysis_message_id": response.id
    }

    return {"messages": [response], "primary_analysis_results": primary_results}


@traceable(name="reviewer_agent", metadata={"node_type": "quality_control", "specialization": "lease_analysis_review"})
async def reviewer_agent(state: State) -> Dict[str, Dict]:
    """REVIEWER AGENT that performs comprehensive quality review and validation of lease analysis.

    This agent orchestrates multi-phase review including legal uncertainty detection,
    ambiguity flagging, onerous provision clustering, and sophisticated confidence calibration.

    Args:
        state (State): Current conversation state with primary analysis results

    Returns:
        dict: Updated state with comprehensive reviewer findings
    """
    start_time = time.time()
    configuration = Configuration.from_context()

    # Implement timeout functionality using configuration
    timeout_seconds = configuration.reviewer_timeout_seconds

    try:
        # Use asyncio.wait_for to implement timeout
        return await asyncio.wait_for(
            _reviewer_agent_core(state, configuration, start_time),
            timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        # Handle timeout gracefully
        timeout_findings = ReviewerFindings(
            review_status=ReviewStatus.FAILED,
            review_timestamp=datetime.now(tz=UTC)
        )

        timeout_response = create_error_response(
            Exception(
                "Reviewer Agent timed out"), "Reviewer Agent", "reviewer_agent"
        )

        return {
            "messages": [timeout_response],
            "reviewer_findings": timeout_findings,
            "review_metrics": ReviewMetrics(processing_time_seconds=timeout_seconds)
        }


def _calculate_terms_analyzed_count(primary_content: str, review_findings: ReviewerFindings) -> int:
    """Calculate the actual number of lease terms analyzed.

    Args:
        primary_content: The lease content or AI analysis content from kb_lease_doc_agent
        review_findings: The current reviewer findings

    Returns:
        Estimated number of terms analyzed
    """
    # Common lease terms indicators
    term_indicators = [
        'rent', 'lease term', 'security deposit', 'maintenance', 'repairs',
        'insurance', 'assignment', 'subletting', 'default', 'termination',
        'renewal', 'escalation', 'utilities', 'parking', 'common area',
        'improvements', 'compliance', 'liability', 'indemnification',
        'force majeure', 'notice', 'governing law', 'arbitration'
    ]

    # Count unique term mentions in analysis
    content_lower = primary_content.lower()
    unique_terms_found = set()

    for term in term_indicators:
        if term in content_lower:
            unique_terms_found.add(term)

    # Add terms from flags and clusters
    for flag in review_findings.flags:
        if flag.cross_references:
            for ref in flag.cross_references:
                if ref.clause:
                    unique_terms_found.add(ref.clause.lower())

    for cluster in review_findings.onerous_clusters:
        unique_terms_found.add(cluster.cluster_name.lower())

    # Return minimum of 1, maximum reasonable estimate
    return max(1, min(len(unique_terms_found), 25))


def _calculate_review_depth_score(
    configuration: Configuration,
    all_flags: list,
    review_findings: ReviewerFindings,
    terms_analyzed_count: int
) -> float:
    """Calculate actual review depth score based on analysis completeness.

    Args:
        configuration: Current configuration settings
        all_flags: All flags raised during review
        review_findings: Complete reviewer findings
        terms_analyzed_count: Number of terms analyzed

    Returns:
        Review depth score (0.0-100.0)
    """
    base_score = 50.0

    # Depth level adjustment
    depth_multipliers = {
        "basic": 0.8,
        "standard": 1.0,
        "comprehensive": 1.2
    }
    depth_multiplier = depth_multipliers.get(
        configuration.review_depth_level, 1.0)

    # Analysis completeness factors
    flags_factor = min(20.0, len(all_flags) * 2.0)  # Up to 20 points for flags
    clusters_factor = min(
        15.0, len(review_findings.onerous_clusters) * 5.0)  # Up to 15 points
    terms_factor = min(10.0, terms_analyzed_count *
                       0.5)  # Up to 10 points for terms

    # Feature utilization bonus
    feature_bonus = 0.0
    if configuration.enable_legal_uncertainty_detection:
        feature_bonus += 2.0
    if configuration.enable_ambiguity_detection:
        feature_bonus += 2.0
    if configuration.enable_onerous_clustering:
        feature_bonus += 2.0
    if configuration.enable_confidence_calibration:
        feature_bonus += 2.0

    # Calculate final score
    raw_score = (base_score + flags_factor + clusters_factor +
                 terms_factor + feature_bonus) * depth_multiplier

    # Cap at 100.0
    return min(100.0, raw_score)


def _populate_lease_term_analyses(primary_content: str, all_flags: list) -> list:
    """Populate detailed lease term analyses based on flags and content.

    Args:
        primary_content: The lease content or AI analysis content
        all_flags: All flags raised during review

    Returns:
        List of LeaseTermAnalysis objects
    """
    from react_agent.state import LeaseTermAnalysis, ImpactLevel, ImpactDirection, MarketPosition, CrossReference

    term_analyses = []

    # Group flags by related terms
    term_groups = {}
    for flag in all_flags:
        # Use first cross-reference clause as term key, or "general" if none
        term_key = "general"
        if flag.cross_references and len(flag.cross_references) > 0:
            first_ref = flag.cross_references[0]
            if first_ref.clause:
                term_key = first_ref.clause

        if term_key not in term_groups:
            term_groups[term_key] = []
        term_groups[term_key].append(flag)

    # Create analyses for each term group
    for term_name, flags in term_groups.items():
        if len(flags) == 0:
            continue

        # Determine overall impact from flags
        severity_levels = [f.severity for f in flags]
        max_severity = max(severity_levels, key=lambda x: [
                           'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'].index(x.value))

        # Create term analysis
        analysis = LeaseTermAnalysis(
            term_name=term_name,
            cross_references=flags[0].cross_references if flags[0].cross_references else [
            ],
            impact_magnitude=max_severity,
            impact_direction=ImpactDirection.NEGATIVE if any(
                f.requires_legal_counsel for f in flags) else ImpactDirection.NEUTRAL,
            market_position=MarketPosition.LANDLORD_FAVORABLE if max_severity.value in [
                'HIGH', 'CRITICAL'] else MarketPosition.MARKET_STANDARD,
            analysis_notes=f"Analysis based on {len(flags)} flags. Risk factors: {', '.join([f.title for f in flags[:3]])}",
            related_terms=[f"Related to {f.title.lower()}" for f in flags[:3]]
        )

        term_analyses.append(analysis)

    return term_analyses


def _populate_quality_validation(all_flags: list, review_findings: ReviewerFindings) -> object:
    """Populate quality validation results based on review completeness.

    Args:
        all_flags: All flags raised during review
        review_findings: Current reviewer findings

    Returns:
        QualityValidation object with validation results
    """
    from react_agent.state import QualityValidation, ValidationItem

    validation_items = []

    # Check completeness of different analysis aspects
    validation_items.append(ValidationItem(
        check_name="Legal Uncertainty Detection",
        status=any("legal_uncertainty" in f.flag_type.lower()
                   for f in all_flags),
        description="Verified that legal uncertainties are properly identified and flagged"
    ))

    validation_items.append(ValidationItem(
        check_name="Ambiguity Detection",
        status=any("ambiguity" in f.flag_type.lower() for f in all_flags),
        description="Verified that contractual ambiguities are properly detected"
    ))

    validation_items.append(ValidationItem(
        check_name="Provision Clustering",
        status=len(review_findings.onerous_clusters) > 0,
        description="Verified that related onerous provisions are properly clustered"
    ))

    validation_items.append(ValidationItem(
        check_name="Confidence Calibration",
        status=review_findings.confidence_metrics.confidence_score > 0,
        description="Verified that confidence scores are properly calibrated"
    ))

    validation_items.append(ValidationItem(
        check_name="Legal Review Requirements",
        status=review_findings.requires_legal_review == any(
            f.requires_legal_counsel for f in all_flags),
        description="Verified consistency between flags and legal review requirements"
    ))

    # Calculate overall quality score
    passed_checks = sum(1 for item in validation_items if item.status)
    overall_score = (passed_checks / len(validation_items)) * 100.0

    return QualityValidation(
        validation_items=validation_items,
        overall_quality_score=overall_score
    )


def _generate_detailed_report(review_findings: ReviewerFindings, metrics: object, configuration: Configuration) -> Dict[str, Any]:
    """Generate a comprehensive detailed report of all review findings.

    Args:
        review_findings: Complete reviewer findings
        metrics: Review metrics
        configuration: Current configuration

    Returns:
        Detailed report dictionary with comprehensive information
    """
    detailed_report = {
        "summary": {
            "review_status": review_findings.review_status.value,
            "total_flags": len(review_findings.flags),
            "high_severity_flags": len([f for f in review_findings.flags if f.severity.value in ['HIGH', 'CRITICAL']]),
            "legal_review_required": review_findings.requires_legal_review,
            "confidence_score": review_findings.confidence_metrics.confidence_score,
            "processing_time": metrics.processing_time_seconds,
            "terms_analyzed": metrics.terms_analyzed_count,
            "review_depth_score": metrics.review_depth_score
        },
        "all_flags": [
            {
                "title": flag.title,
                "description": flag.description,
                "severity": flag.severity.value,
                "flag_type": flag.flag_type,
                "requires_legal_counsel": flag.requires_legal_counsel,
                "cross_references": [
                    {
                        "clause": ref.clause,
                        "exact_quote": ref.exact_quote,
                        "document": ref.document,
                        "page_number": ref.page_number
                    } for ref in flag.cross_references
                ]
            } for flag in review_findings.flags
        ],
        "provision_clusters": [
            {
                "cluster_name": cluster.cluster_name,
                "cumulative_impact": cluster.cumulative_impact.value,
                "primary_terms": cluster.primary_terms,
                "supporting_terms": cluster.supporting_terms,
                "interaction_description": cluster.interaction_description,
                "flags": [flag.title for flag in cluster.flags]
            } for cluster in review_findings.onerous_clusters
        ],
        "lease_term_analyses": [
            {
                "term_name": analysis.term_name,
                "impact_magnitude": analysis.impact_magnitude.value,
                "impact_direction": analysis.impact_direction.value,
                "market_position": analysis.market_position.value,
                "analysis_notes": analysis.analysis_notes,
                "related_terms": analysis.related_terms,
                "cross_references": [
                    {
                        "clause": ref.clause,
                        "exact_quote": ref.exact_quote,
                        "document": ref.document,
                        "page_number": ref.page_number
                    } for ref in analysis.cross_references
                ]
            } for analysis in review_findings.lease_term_analyses
        ],
        "quality_validation": {
            "overall_score": review_findings.quality_validation.overall_quality_score,
            "validation_checks": [
                {
                    "check_name": item.check_name,
                    "status": "PASS" if item.status else "FAIL",
                    "description": item.description
                } for item in review_findings.quality_validation.validation_items
            ]
        },
        "confidence_metrics": {
            "confidence_score": review_findings.confidence_metrics.confidence_score,
            "uncertainty_range": review_findings.confidence_metrics.uncertainty_range,
            "confidence_factors": review_findings.confidence_metrics.confidence_factors,
            "uncertainty_sources": review_findings.confidence_metrics.uncertainty_sources
        },
        "recommendations": review_findings.recommendations,
        "configuration_used": {
            "review_depth_level": configuration.review_depth_level,
            "max_flags_per_analysis": configuration.max_flags_per_analysis,
            "confidence_calibration_strategy": configuration.confidence_calibration_strategy,
            "features_enabled": {
                "legal_uncertainty_detection": configuration.enable_legal_uncertainty_detection,
                "ambiguity_detection": configuration.enable_ambiguity_detection,
                "onerous_clustering": configuration.enable_onerous_clustering,
                "confidence_calibration": configuration.enable_confidence_calibration
            }
        }
    }

    return detailed_report


async def _reviewer_agent_core(state: State, configuration: Configuration, start_time: float) -> Dict[str, Dict]:
    """Core reviewer agent logic separated for timeout handling."""

    # Phase 1: Validation and Setup
    if not state.primary_analysis_results or not state.primary_analysis_results.get("analysis_completed"):
        # No primary analysis to review - create minimal reviewer findings
        minimal_findings = ReviewerFindings(
            review_status=ReviewStatus.FAILED,
            review_timestamp=datetime.now(tz=UTC)
        )

        response = create_error_response(
            Exception(
                "No primary analysis results found to review"), "Reviewer Agent", "_reviewer_agent_core"
        )

        return {
            "messages": [response],
            "reviewer_findings": minimal_findings,
            "review_metrics": ReviewMetrics(processing_time_seconds=time.time() - start_time)
        }

    # Apply review depth configuration
    review_depth = configuration.review_depth_level
    max_flags = configuration.max_flags_per_analysis

    # Initialize model for review operations
    model = load_chat_model(configuration.model)

    # Phase 2: Review Orchestration - Apply configuration-driven review
    review_findings = ReviewerFindings(
        review_status=ReviewStatus.IN_PROGRESS,
        review_timestamp=datetime.now(tz=UTC)
    )

    review_messages = []
    all_flags = []
    analyzer_confidence_metrics = []

    try:
        # Get original lease content for sophisticated analysis
        # Use original lease content for pattern-based analysis, fall back to AI analysis if needed
        primary_content = (
            state.primary_analysis_results.get("original_lease_content", "") or
            state.primary_analysis_results.get("analysis_content", "")
        )

        # Sophisticated Legal Uncertainty Detection (if enabled)
        if configuration.enable_legal_uncertainty_detection and primary_content:
            legal_flags, legal_confidence = analyze_legal_uncertainties(
                primary_content, configuration)
            all_flags.extend(legal_flags)
            analyzer_confidence_metrics.append(legal_confidence)

            review_messages.append(
                f"Legal Uncertainty Analysis: Detected {len(legal_flags)} potential legal uncertainties. "
                f"Confidence: {legal_confidence.confidence_score:.1f}%. "
                f"{sum(1 for f in legal_flags if f.requires_legal_counsel)} flags require legal counsel."
            )

        # Sophisticated Ambiguity Detection (if enabled)
        if configuration.enable_ambiguity_detection and primary_content:
            ambiguity_flags, ambiguity_confidence = analyze_ambiguities(
                primary_content, configuration)
            all_flags.extend(ambiguity_flags)
            analyzer_confidence_metrics.append(ambiguity_confidence)

            review_messages.append(
                f"Ambiguity Analysis: Detected {len(ambiguity_flags)} potential ambiguities. "
                f"Confidence: {ambiguity_confidence.confidence_score:.1f}%. "
                f"{sum(1 for f in ambiguity_flags if f.severity.value in ['HIGH', 'CRITICAL'])} high-impact ambiguities found."
            )

        # Sophisticated Onerous Provision Clustering (if enabled)
        if configuration.enable_onerous_clustering and primary_content:
            clusters, clustering_confidence = analyze_provision_clusters(
                primary_content, configuration)
            review_findings.onerous_clusters = clusters
            analyzer_confidence_metrics.append(clustering_confidence)

            review_messages.append(
                f"Clustering Analysis: Identified {len(clusters)} onerous provision clusters. "
                f"Confidence: {clustering_confidence.confidence_score:.1f}%. "
                f"{sum(1 for c in clusters if c.cumulative_impact.value in ['HIGH', 'CRITICAL'])} high-impact clusters detected."
            )

        # Update review findings with combined flags (apply max_flags limit)
        review_findings.flags = all_flags[:
                                          max_flags] if max_flags > 0 else all_flags

        # Log if flags were truncated
        if len(all_flags) > max_flags:
            review_messages.append(
                f"⚠️ Flag limit reached: Showing top {max_flags} of {len(all_flags)} identified issues. Consider adjusting max_flags_per_analysis configuration."
            )

        # Sophisticated Confidence Calibration (if enabled and we have analyzer results)
        if configuration.enable_confidence_calibration and analyzer_confidence_metrics:
            calibrated_confidence, calibration_details = calibrate_confidence(
                analyzer_confidence_metrics, primary_content, configuration
            )
            review_findings.confidence_metrics = calibrated_confidence

            review_messages.append(
                f"Confidence Calibration: Applied {calibration_details.strategy_used.value} strategy. "
                f"Final confidence: {calibrated_confidence.confidence_score:.1f}% "
                f"(range: {calibrated_confidence.uncertainty_range['lower']:.1f}%-{calibrated_confidence.uncertainty_range['upper']:.1f}%)"
            )
        elif analyzer_confidence_metrics:
            # Fallback to conservative approach if calibration is disabled
            min_confidence = min(
                c.confidence_score for c in analyzer_confidence_metrics)
            combined_factors = []
            combined_sources = []

            for conf in analyzer_confidence_metrics:
                combined_factors.extend(conf.confidence_factors)
                combined_sources.extend(conf.uncertainty_sources)

            review_findings.confidence_metrics = ConfidenceMetrics(
                confidence_score=min_confidence,
                uncertainty_range={"lower": max(
                    10.0, min_confidence - 15.0), "upper": min(95.0, min_confidence + 10.0)},
                confidence_factors=combined_factors,
                uncertainty_sources=combined_sources
            )

        # Additional LLM-based confidence calibration (if enabled and no sophisticated calibration)
        if configuration.enable_confidence_calibration and not analyzer_confidence_metrics and primary_content:
            confidence_prompt = prompts.CONFIDENCE_CALIBRATION_PROMPT
            confidence_response = cast(
                AIMessage,
                await model.ainvoke([
                    {"role": "system", "content": confidence_prompt},
                    {"role": "user", "content": f"Calibrate confidence for this analysis:\n\n{primary_content}"}
                ])
            )
            review_messages.append(
                f"LLM Confidence Calibration: {confidence_response.content}")

            # Set default confidence metrics if not already set
            if not review_findings.confidence_metrics.confidence_score:
                review_findings.confidence_metrics = ConfidenceMetrics(
                    # Adjust based on flags
                    confidence_score=max(0.0, 85.0 - (len(all_flags) * 3)),
                    uncertainty_range={"lower": 70.0, "upper": 95.0},
                    confidence_factors=["Review completed",
                                        "Primary analysis available"],
                    uncertainty_sources=[
                        "Issues detected in analysis"] if all_flags else []
                )

        # Phase 3: Complete Review Assessment and populate missing fields
        review_findings.review_status = ReviewStatus.COMPLETED
        review_findings.requires_legal_review = any(
            f.requires_legal_counsel for f in all_flags)

        # Populate lease term analyses and quality validation
        review_findings.lease_term_analyses = _populate_lease_term_analyses(
            primary_content, all_flags)
        review_findings.quality_validation = _populate_quality_validation(
            all_flags, review_findings)

        # Generate comprehensive recommendations
        recommendations = ["Review completed successfully"]

        if all_flags:
            recommendations.append(
                f"Identified {len(all_flags)} areas requiring attention")
        else:
            recommendations.append("No significant issues identified")

        if any(f.requires_legal_counsel for f in all_flags):
            recommendations.append(
                f"Legal counsel recommended for {sum(1 for f in all_flags if f.requires_legal_counsel)} items")
        else:
            recommendations.append("No legal counsel required")

        if review_findings.onerous_clusters:
            high_impact_clusters = [
                c for c in review_findings.onerous_clusters if c.cumulative_impact.value in ['HIGH', 'CRITICAL']]
            if high_impact_clusters:
                recommendations.append(
                    f"Found {len(high_impact_clusters)} high-impact provision clusters requiring attention")

        review_findings.recommendations = recommendations

        # Phase 4: Generate Response
        cluster_summary = ""
        if review_findings.onerous_clusters:
            max_clusters = configuration.max_clusters_in_summary
            cluster_summary = f"\n\nClustering Results:\n"

            for cluster in review_findings.onerous_clusters[:max_clusters]:
                cluster_summary += f"• {cluster.cluster_name}: {cluster.cumulative_impact.value} impact"
                total_terms = len(cluster.primary_terms) + \
                    len(cluster.supporting_terms)
                if total_terms > 0:
                    cluster_summary += f" ({total_terms} terms)"
                cluster_summary += "\n"

            if len(review_findings.onerous_clusters) > max_clusters:
                remaining = len(
                    review_findings.onerous_clusters) - max_clusters
                cluster_summary += f"• ...and {remaining} more clusters"

        # Add calibration summary if available
        calibration_summary = ""
        if configuration.enable_confidence_calibration and 'calibration_details' in locals():
            calibration_summary = f"\n\nConfidence Calibration:\n"
            calibration_summary += f"• Strategy: {calibration_details.strategy_used.value}\n"
            calibration_summary += f"• Base confidence: {calibration_details.base_confidence:.1f}%\n"
            calibration_summary += f"• Complexity adjustment: {calibration_details.complexity_adjustment:+.1f}%\n"
            calibration_summary += f"• Agreement adjustment: {calibration_details.agreement_adjustment:+.1f}%"

        # Calculate metrics for summary display
        terms_analyzed_count = _calculate_terms_analyzed_count(
            primary_content, review_findings)
        review_depth_score = _calculate_review_depth_score(
            configuration, all_flags, review_findings, terms_analyzed_count)

        # Build enhanced review summary sections
        summary_parts = [
            "🔍 REVIEWER AGENT ANALYSIS COMPLETE",
            "=" * 50,
            "",
            f"📊 Review Status: {review_findings.review_status.value}",
            f"🚩 Flags Raised: {len(all_flags)} ({len([f for f in all_flags if f.severity.value in ['HIGH', 'CRITICAL']])} high severity)",
            f"📦 Provision Clusters: {len(review_findings.onerous_clusters)}",
            f"📈 Terms Analyzed: {terms_analyzed_count}",
            f"🎯 Confidence Score: {review_findings.confidence_metrics.confidence_score:.1f}%",
            f"⚖️  Legal Review Required: {'Yes' if review_findings.requires_legal_review else 'No'}",
            f"🏆 Quality Score: {review_findings.quality_validation.overall_quality_score:.1f}%",
            f"📏 Review Depth Score: {review_depth_score:.1f}%",
            ""
        ]

        # Add Review Summary section if there are messages
        if review_messages:
            max_messages = configuration.max_review_messages_in_summary
            summary_parts.extend([
                "Review Summary:",
                *review_messages[:max_messages],
                ""
            ])
            if len(review_messages) > max_messages:
                summary_parts.append(
                    f"...and {len(review_messages) - max_messages} more review items")
                summary_parts.append("")

        # Add Key Findings section if there are flags
        if all_flags:
            max_flags = configuration.max_flags_in_summary
            max_desc_length = configuration.flag_description_max_length

            summary_parts.append("Key Findings:")
            for flag in all_flags[:max_flags]:
                if len(flag.description) > max_desc_length:
                    desc = f"{flag.description[:max_desc_length]}..."
                else:
                    desc = flag.description
                summary_parts.append(f"• {flag.title}: {desc}")

            if len(all_flags) > max_flags:
                summary_parts.append(
                    f"...and {len(all_flags) - max_flags} more findings")
            summary_parts.append("")

        # Add cluster summary if available
        if cluster_summary:
            summary_parts.append(cluster_summary.strip())
            summary_parts.append("")

        # Add calibration summary if available
        if calibration_summary:
            summary_parts.append(calibration_summary.strip())
            summary_parts.append("")

        # Add quality validation summary
        if review_findings.quality_validation.validation_items:
            summary_parts.append("Quality Validation:")
            passed_checks = [
                item for item in review_findings.quality_validation.validation_items if item.status]
            failed_checks = [
                item for item in review_findings.quality_validation.validation_items if not item.status]

            summary_parts.append(
                f"✅ Passed: {len(passed_checks)}/{len(review_findings.quality_validation.validation_items)} checks")
            if failed_checks:
                summary_parts.append("❌ Failed checks:")
                for check in failed_checks[:2]:  # Show max 2 failed checks
                    summary_parts.append(f"  • {check.check_name}")
                if len(failed_checks) > 2:
                    summary_parts.append(
                        f"  • ...and {len(failed_checks) - 2} more")
            summary_parts.append("")

        # Add recommendations
        summary_parts.append("📋 Recommendations:")
        for i, rec in enumerate(review_findings.recommendations, 1):
            summary_parts.append(f"{i}. {rec}")

        # Add detailed report reference if available
        if configuration.include_detailed_report:
            summary_parts.append("")
            summary_parts.append(
                "📄 Note: Comprehensive detailed report available in 'detailed_report' field")

        # Join and strip any trailing whitespace
        review_summary = "\n".join(summary_parts).strip()

        response = AIMessage(content=review_summary)

        # Calculate final metrics with proper calculations
        processing_time = time.time() - start_time

        # Terms and depth scores already calculated above for summary

        final_metrics = ReviewMetrics(
            flags_raised_count=len(all_flags),
            high_severity_flags_count=len(
                [f for f in all_flags if f.severity.value in ['CRITICAL', 'HIGH']]),
            legal_counsel_required_count=len(
                [f for f in all_flags if f.requires_legal_counsel]),
            terms_analyzed_count=terms_analyzed_count,
            clusters_identified_count=len(review_findings.onerous_clusters),
            review_depth_score=review_depth_score,
            processing_time_seconds=processing_time
        )

        # Generate detailed report if configured
        result = {
            "messages": [response],
            "reviewer_findings": review_findings,
            "review_metrics": final_metrics
        }

        if configuration.include_detailed_report:
            detailed_report = _generate_detailed_report(
                review_findings, final_metrics, configuration)
            result["detailed_report"] = detailed_report

        return result

    except Exception as e:
        # Handle errors gracefully
        error_findings = ReviewerFindings(
            review_status=ReviewStatus.FAILED,
            review_timestamp=datetime.now(tz=UTC)
        )

        error_response = create_error_response(
            e, "Reviewer Agent", "_reviewer_agent_core")

        return {
            "messages": [error_response],
            "reviewer_findings": error_findings,
            "review_metrics": ReviewMetrics(processing_time_seconds=time.time() - start_time)
        }

# Define a new graph

builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node(call_model)
builder.add_node("research_agent", research_agent)
builder.add_node("kb_lease_doc_agent", kb_lease_doc_agent)
builder.add_node("reviewer_agent", reviewer_agent)
builder.add_node("tools", ToolNode(TOOLS))

# Set the entrypoint as `call_model`
# This means that this node is the first one called
builder.add_edge("__start__", "call_model")


@traceable(name="route_model_output", metadata={"node_type": "router"})
def route_model_output(state: State) -> Literal["__end__", "tools", "research_agent", "kb_lease_doc_agent", "reviewer_agent"]:
    """Enhanced router with intelligent decision making based on state, configuration, and context.

    This function provides sophisticated routing logic that considers:
    - Current state and analysis progress
    - Configuration settings and capabilities
    - Conversation context and intent
    - Error conditions and fallback scenarios

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call.
    """
    if not state.messages:
        raise ValueError("No messages in state to route")

    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )

    # Get configuration for routing decisions
    try:
        configuration = Configuration.from_context()
    except Exception:
        # Fallback to default configuration if context not available
        configuration = Configuration()

    # Get both AI response content and recent conversation context
    ai_content = last_message.content.lower() if isinstance(
        last_message.content, str) else ""

    # Build comprehensive conversation context (increased window for better context)
    conversation_context = ""
    context_window = 5  # Expanded from 3 for better context awareness
    for msg in state.messages[-context_window:]:
        if hasattr(msg, 'content') and isinstance(msg.content, str):
            conversation_context += msg.content.lower() + " "

    # Priority 1: Route to reviewer agent if primary analysis is complete and review is pending
    if (state.primary_analysis_results and
        state.primary_analysis_results.get("analysis_completed") and
            state.reviewer_findings.review_status == ReviewStatus.PENDING):
        return "reviewer_agent"

    # Priority 2: Handle tool calls (always highest operational priority)
    if last_message.tool_calls:
        return "tools"

    # Priority 3: Route to lease document processing if needed and not already completed
    if (not state.primary_analysis_results or
            not state.primary_analysis_results.get("analysis_completed")):

        # Enhanced lease detection with more comprehensive keywords
        lease_keywords = [
            "lease", "rental", "valuation", "commercial real estate",
            "tenant", "landlord", "rent", "lease agreement", "lease document",
            "commercial property", "rental agreement", "property lease",
            "lease terms", "lease analysis", "property valuation"
        ]

        # Check for document upload or lease content indicators
        document_indicators = ["document", "contract",
                               "agreement", "analyze this", "review this"]

        if (any(term in ai_content for term in lease_keywords) or
            any(term in conversation_context for term in lease_keywords) or
                any(term in conversation_context for term in document_indicators)):
            return "kb_lease_doc_agent"

    # Priority 4: Route to research agent for information gathering needs
    # Enhanced research detection
    research_keywords = [
        "research", "information", "details", "market trends", "search for",
        "find information", "current market", "industry standards", "legal precedent",
        "market data", "comparable", "market analysis", "industry report"
    ]

    research_phrases = [
        "what is the current", "how does the market", "what are typical",
        "industry standard", "market rate", "recent trends"
    ]

    if (any(term in ai_content for term in research_keywords) or
        any(term in conversation_context for term in research_keywords) or
            any(phrase in conversation_context for phrase in research_phrases)):
        return "research_agent"

    # Priority 5: Check for explicit requests that need specialized handling
    special_requests = {
        "review": "reviewer_agent",
        "validate": "reviewer_agent",
        "check": "reviewer_agent",
        "analysis": "kb_lease_doc_agent"
    }

    for keyword, agent in special_requests.items():
        if keyword in ai_content and agent == "reviewer_agent":
            # Only route to reviewer if we have something to review
            if (state.primary_analysis_results and
                    state.primary_analysis_results.get("analysis_completed")):
                return "reviewer_agent"
        elif keyword in ai_content:
            return agent

    # Priority 6: End conversation if no specific routing needed
    return "__end__"


# Add a conditional edge to determine the next step after `call_model`
builder.add_conditional_edges(
    "call_model",
    # After call_model finishes running, the next node(s) are scheduled
    # based on the output from route_model_output
    route_model_output,
)

# Add normal edges from specialized agents back to call_model
builder.add_edge("tools", "call_model")
builder.add_edge("research_agent", "call_model")
builder.add_edge("kb_lease_doc_agent", "reviewer_agent")
builder.add_edge("reviewer_agent", "__end__")

# Compile the builder into an executable graph
graph = builder.compile(name="ReAct Agent")
