"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from datetime import UTC, datetime
from typing import Dict, List, Literal, cast
import os
import time

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langsmith import Client, traceable

from react_agent.configuration import Configuration
from react_agent.state import InputState, State, ReviewerFindings, ReviewStatus, ConfidenceMetrics, ReviewMetrics
from react_agent.tools import TOOLS
from react_agent.utils import load_chat_model
from react_agent import prompts
from react_agent.legal_analyzer import analyze_legal_uncertainties
from react_agent.ambiguity_analyzer import analyze_ambiguities
from react_agent.clustering_analyzer import analyze_provision_clusters
from react_agent.confidence_calibrator import calibrate_confidence

# Define the function that calls the model


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
    """Research agent that specializes in gathering information."""
    configuration = Configuration.from_context()

    model = load_chat_model(configuration.model)

    research_prompt = "You are a research specialist. Provide detailed, well-sourced information about the user's question."

    response = cast(
        AIMessage,
        await model.ainvoke([
            {"role": "system", "content": research_prompt},
            *state.messages
        ])
    )

    return {"messages": [response]}

# Add this new kb and lease document processing agent function


@traceable(name="kb_lease_doc_agent", metadata={"node_type": "domain_expert", "specialization": "real_estate_leases"})
async def kb_lease_doc_agent(state: State) -> Dict[str, List[AIMessage]]:
    """KB and lease document processing agent using LangSmith prompt."""

    configuration = Configuration.from_context()
    model = load_chat_model(configuration.model)
    client = Client()

    try:
        # Pull the complete prompt template
        prompt_template = client.pull_prompt(
            "v22-prompt1-establish-baseline-terms")

        # The prompt is likely complete - just invoke without parameters
        # or with minimal context if needed
        formatted_prompt = prompt_template.invoke({})

        # Add the user's message to the existing system messages
        if not state.messages:
            raise ValueError("No messages in state to process")
        all_messages = formatted_prompt.messages + [state.messages[-1]]

        # Use all messages with model
        response = cast(
            AIMessage,
            await model.ainvoke(all_messages)
        )
    except Exception as e:
        # Fallback if prompt pulling fails
        fallback_prompt = "You are a commercial real estate expert specializing in lease document analysis and valuation."
        response = cast(
            AIMessage,
            await model.ainvoke([
                {"role": "system", "content": fallback_prompt},
                *state.messages
            ])
        )

    # Extract original lease content from user messages for reviewer analysis
    lease_content = ""
    for msg in state.messages:
        if hasattr(msg, 'content') and isinstance(msg.content, str):
            # Look for lease-related content in user messages
            content_lower = msg.content.lower()
            if any(term in content_lower for term in ["lease", "rental", "tenant", "landlord"]):
                lease_content = msg.content
                break

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

    # Phase 1: Validation and Setup
    if not state.primary_analysis_results or not state.primary_analysis_results.get("analysis_completed"):
        # No primary analysis to review - create minimal reviewer findings
        minimal_findings = ReviewerFindings(
            review_status=ReviewStatus.FAILED,
            review_timestamp=datetime.now(tz=UTC)
        )

        response = AIMessage(
            content="No primary analysis results found to review. Please ensure primary lease analysis is completed first."
        )

        return {
            "messages": [response],
            "reviewer_findings": minimal_findings,
            "review_metrics": ReviewMetrics(processing_time_seconds=time.time() - start_time)
        }

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

        # Update review findings with combined flags
        review_findings.flags = all_flags

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

        # Phase 3: Complete Review Assessment
        review_findings.review_status = ReviewStatus.COMPLETED
        review_findings.requires_legal_review = any(
            f.requires_legal_counsel for f in all_flags)

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
            cluster_summary = f"\n\nClustering Results:\n"
            cluster_summary += "\n".join([f"• {c.cluster_name}: {c.cumulative_impact.value} impact"
                                          for c in review_findings.onerous_clusters[:3]])
            if len(review_findings.onerous_clusters) > 3:
                cluster_summary += f"\n• ...and {len(review_findings.onerous_clusters) - 3} more clusters"

        # Add calibration summary if available
        calibration_summary = ""
        if configuration.enable_confidence_calibration and 'calibration_details' in locals():
            calibration_summary = f"\n\nConfidence Calibration:\n"
            calibration_summary += f"• Strategy: {calibration_details.strategy_used.value}\n"
            calibration_summary += f"• Base confidence: {calibration_details.base_confidence:.1f}%\n"
            calibration_summary += f"• Complexity adjustment: {calibration_details.complexity_adjustment:+.1f}%\n"
            calibration_summary += f"• Agreement adjustment: {calibration_details.agreement_adjustment:+.1f}%"

        # Build review summary sections carefully to avoid trailing whitespace
        summary_parts = [
            "REVIEWER AGENT ANALYSIS COMPLETE",
            "",
            f"Review Status: {review_findings.review_status.value}",
            f"Flags Raised: {len(all_flags)}",
            f"Provision Clusters: {len(review_findings.onerous_clusters)}",
            f"Confidence Score: {review_findings.confidence_metrics.confidence_score:.1f}%",
            f"Legal Review Required: {review_findings.requires_legal_review}",
            ""
        ]

        # Add Review Summary section if there are messages
        if review_messages:
            summary_parts.extend([
                "Review Summary:",
                *review_messages[:5],
                ""
            ])

        # Add Key Findings section if there are flags
        if all_flags:
            summary_parts.append("Key Findings:")
            summary_parts.extend(
                [f"• {f.title}: {f.description[:100]}..." for f in all_flags[:3]])
            summary_parts.append("")

        # Add cluster summary if available
        if cluster_summary:
            summary_parts.append(cluster_summary.strip())
            summary_parts.append("")

        # Add calibration summary if available
        if calibration_summary:
            summary_parts.append(calibration_summary.strip())
            summary_parts.append("")

        # Add recommendations
        summary_parts.append("Recommendations:")
        summary_parts.extend(review_findings.recommendations)

        # Join and strip any trailing whitespace
        review_summary = "\n".join(summary_parts).strip()

        response = AIMessage(content=review_summary)

        # Calculate final metrics
        processing_time = time.time() - start_time
        final_metrics = ReviewMetrics(
            flags_raised_count=len(all_flags),
            high_severity_flags_count=len(
                [f for f in all_flags if f.severity.value in ['CRITICAL', 'HIGH']]),
            legal_counsel_required_count=len(
                [f for f in all_flags if f.requires_legal_counsel]),
            terms_analyzed_count=1,  # Simplified - would count actual terms in real implementation
            clusters_identified_count=len(review_findings.onerous_clusters),
            review_depth_score=85.0,  # Based on configuration and completeness
            processing_time_seconds=processing_time
        )

        return {
            "messages": [response],
            "reviewer_findings": review_findings,
            "review_metrics": final_metrics
        }

    except Exception as e:
        # Handle errors gracefully
        error_findings = ReviewerFindings(
            review_status=ReviewStatus.FAILED,
            review_timestamp=datetime.now(tz=UTC)
        )

        error_response = AIMessage(
            content=f"REVIEWER AGENT encountered an error during analysis: {str(e)[:200]}... Review could not be completed."
        )

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
    """Determine the next node based on the model's output and conversation context.

    This function checks if the model's last message contains tool calls
    or if specialized processing is needed based on both the AI response
    and the conversation history.

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

    # Get both AI response content and recent conversation context
    ai_content = last_message.content.lower() if isinstance(
        last_message.content, str) else ""

    # Check recent conversation history (last few messages) for context
    conversation_context = ""
    for msg in state.messages[-3:]:  # Check last 3 messages for context
        if hasattr(msg, 'content') and isinstance(msg.content, str):
            conversation_context += msg.content.lower() + " "

    # Check if we have primary analysis results and should route to reviewer
    if (state.primary_analysis_results and
        state.primary_analysis_results.get("analysis_completed") and
            state.reviewer_findings.review_status == ReviewStatus.PENDING):
        return "reviewer_agent"

    # Check for lease document processing (check both AI response and conversation context)
    lease_keywords = ["lease", "rental", "valuation", "commercial real estate",
                      "tenant", "landlord", "rent", "lease agreement", "lease document"]
    if any(term in ai_content for term in lease_keywords) or any(term in conversation_context for term in lease_keywords):
        return "kb_lease_doc_agent"

    # Check if needing research (check both AI response and conversation context)
    research_keywords = ["research", "information", "details",
                         "market trends", "search for", "find information"]
    if any(term in ai_content for term in research_keywords) or any(term in conversation_context for term in research_keywords):
        return "research_agent"

    # If there are tool calls, go to tools
    if last_message.tool_calls:
        return "tools"

    # Otherwise we're done
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
builder.add_edge("kb_lease_doc_agent", "call_model")
builder.add_edge("reviewer_agent", "call_model")

# Compile the builder into an executable graph
graph = builder.compile(name="ReAct Agent")
