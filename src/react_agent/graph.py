"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from datetime import UTC, datetime
from typing import Dict, List, Literal, cast

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.state import InputState, State
from react_agent.tools import TOOLS
from react_agent.utils import load_chat_model

# Define the function that calls the model


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

## Add this new kb and lease document processing agent function
async def kb_lease_doc_agent(state: State) -> Dict[str, List[AIMessage]]:
    """KB and lease document processing agent using LangSmith prompt."""
    from langsmith import Client
    
    configuration = Configuration.from_context()
    model = load_chat_model(configuration.model)
    client = Client()
    
    # Pull the complete prompt template
    prompt_template = client.pull_prompt("v22-prompt1-establish-baseline-terms")
    
    # The prompt is likely complete - just invoke without parameters
    # or with minimal context if needed
    formatted_prompt = prompt_template.invoke({})
    
    # Add the user's message to the existing system messages
    all_messages = formatted_prompt.messages + [state.messages[-1]]
    
    # Use all messages with model
    response = cast(
        AIMessage,
        await model.ainvoke(all_messages)
    )
    
    return {"messages": [response]}

# Define a new graph

builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Define the two nodes we will cycle between
builder.add_node(call_model)
builder.add_node("research_agent", research_agent)
builder.add_node("kb_lease_doc_agent", kb_lease_doc_agent)
builder.add_node("tools", ToolNode(TOOLS))

# Set the entrypoint as `call_model`
# This means that this node is the first one called
builder.add_edge("__start__", "call_model")

def route_model_output(state: State) -> Literal["__end__", "tools", "research_agent", "kb_lease_doc_agent"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls
    or if specialized processing is needed.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call.
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    
    content = last_message.content.lower() if isinstance(last_message.content, str) else ""
    
    # Check for lease document processing
    if any(term in content for term in ["lease", "rental", "valuation", "commercial real estate"]):
        return "kb_lease_doc_agent"
    
    # Check if the response mentions needing research
    if "research" in content or "information" in content or "details" in content:
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

# Add a normal edge from `tools` to `call_model`
# This creates a cycle: after using tools, we always return to the model
builder.add_edge("tools", "call_model")
builder.add_edge("research_agent", "call_model")
builder.add_edge("kb_lease_doc_agent", "call_model")

# Compile the builder into an executable graph
graph = builder.compile(name="ReAct Agent")
