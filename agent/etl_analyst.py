import os
import sys
from typing import Union, List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END

from utils.llm_switch import switch_llm
from models.schema import ETLGenerationSchema
from config.settings import LLMTier
from utils.etl_tools import ETLTools
from utils.prompts import TRANSFORM_LOAD_PROMPT, LLM_TOOL_PROMPT


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@tool
def extract_load_tool(
    url: Union[str, List[str]],
    output_path: str,
    format: str = "csv",
    mode: str = "overwrite",
) -> str:
    """
    Extract data from one or more URLs and load it into a file.

    Args:
        url: A single URL or a list of URLs to extract data from.
        output_path: Destination path (file or directory) for the extracted data.
        format: Output format, e.g. "csv", "parquet", "json". Defaults to "csv".
        mode: Write mode, e.g. "overwrite" or "append". Defaults to "overwrite".

    Returns:
        The path to the saved extracted data.
    """
    etl = ETLTools()
    return etl.extract_load(url, output_path, format=format, mode=mode)


@tool
def transform_load_tool(
    input_file_path: str,
    output_path: str,
    format: str ,
    user_question: Optional[str] = None,
) -> str:
    """
    Transform data from a file using an LLM-generated pandas snippet and save it.

    The tool first gathers a lightweight context (top 5 rows) so the LLM
    can reason about column names/dtypes before emitting transformation code.

    Args:
        input_file_path: Path to the source data file.
        output_path: Folder where the transformed data should be written.
        format: Output format for the transformed data.
        user_question: Optional free-text description of the desired transform.

    Returns:
        A status message including the saved path, the execution result, and
        the pandas code that was executed.
    """
    etl = ETLTools()

    # 1. Sample context so the LLM knows what columns/types it's working with.
    top_5_rows = etl.transform_load_context(
        input_file_path, output_path, format=format
    )

    # 2. Use the high-tier LLM for code generation (more powerful and accurate).
    llm = switch_llm(LLMTier.HIGH)
    response = llm.invoke(
        TRANSFORM_LOAD_PROMPT.format(
            input_file_path=input_file_path,
            output_folder=output_path,
            format=format,
            top_5_rows=top_5_rows,
            user_question=user_question or "",
        )
    )
    
    code=etl.extract_python_code(response)
    # 3. Execute the generated code in the ETL sandbox.
    result = etl.execute_code(code)

    return (
        f"The data is transformed and saved to {output_path}. "
        f"Result: {result}\n pandas code executed:\n{code}"
    )



# Registry used by the tools node for dispatch by name.
TOOLS = [extract_load_tool, transform_load_tool]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def llm_node(state: ETLGenerationSchema) -> ETLGenerationSchema:
    """
    Ask the LLM what to do next given the current conversation state.

    The LLM is bound to the ETL tools, so it may either:
      - return plain content (final answer), or
      - emit `tool_calls` which the tools_node will execute next.
    """
    messages = state.messages

    # Bind tools here (not at import time) so the binding always matches the
    # current registry; also keeps module import side-effect free.
    llm_with_tools = switch_llm(LLMTier.HIGH).bind_tools(TOOLS)

    # Pass the entire message history so the LLM sees prior tool results.
    ai_message: AIMessage = llm_with_tools.invoke(LLM_TOOL_PROMPT.format(messages=messages))

    state.messages = messages + [ai_message]
    return state


def tools_node(state: ETLGenerationSchema) -> ETLGenerationSchema:
    """
    Execute every tool call emitted by the most recent AIMessage and append
    the resulting ToolMessages back into the state.
    """
    last_message = state.messages[-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    tool_messages: List[ToolMessage] = []
    for tool_call in tool_calls:
        tool_fn = TOOLS_BY_NAME.get(tool_call["name"])
        if tool_fn is None:
            # Unknown tool — surface as a ToolMessage so the LLM can recover.
            tool_messages.append(
                ToolMessage(
                    content=f"Unknown tool: {tool_call['name']}",
                    tool_call_id=tool_call["id"],
                )
            )
            continue

        try:
            observation = tool_fn.invoke(tool_call["args"])
        except Exception as exc:  # keep the loop alive even if one tool fails
            observation = f"Tool error: {exc!r}"

        tool_messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
        )

    state.messages = state.messages + tool_messages
    return state


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_llm(state: ETLGenerationSchema) -> str:
    """
    Route to `tools_node` if the LLM requested a tool, otherwise terminate.
    """
    last_message = state.messages[-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []
    return "tools_node" if tool_calls else END


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Build and compile the ETL analyst state graph."""
    graph = StateGraph(ETLGenerationSchema)

    graph.add_node("llm_node", llm_node)
    graph.add_node("tools_node", tools_node)

    graph.add_edge(START, "llm_node")
    graph.add_conditional_edges(
        "llm_node",
        route_after_llm,
        {"tools_node": "tools_node", END: END},
    )
    graph.add_edge("tools_node", "llm_node")  # loop back for follow-up reasoning

    return graph.compile()


etl_analyst_graph = build_graph()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from models.schema import ETLGenerationSchema as _State
    from langchain_core.messages import HumanMessage

    user_prompt = (
        "i want to extract data from "
        "'https://breedhealthscore.com/api/breeds.json', save it to "
        "'data/extracted' in csv format,Then use the file stored in 'data/extracted' as input to transform the data and save it to 'data/transformed' in json format.The transformation should filter the data to include only the columns 'breed', 'health_score', and 'size'."
    )

    # Kick off the agent with an initial human message.
    initial_state = _State(messages=[HumanMessage(content=user_prompt)])
    # final_state = etl_analyst_graph.invoke(initial_state)
    for update in etl_analyst_graph.stream(
    initial_state,
    stream_mode="updates",):
        
        for node_name, node_state in update.items():
            print(f"\n{'=' * 20} {node_name} {'=' * 20}")

            # Your nodes return the full state, including earlier messages.
            messages = (
                node_state.get("messages", [])
                if isinstance(node_state, dict)
                else node_state.messages
            )

            if node_name == "llm_node":
                # Print only the new AI message.
                messages[-1].pretty_print()

            elif node_name == "tools_node":
                # Print only the tool results appended by this node.
                new_tool_messages = []
                for message in reversed(messages):
                    if not isinstance(message, ToolMessage):
                        break
                    new_tool_messages.append(message)

                for message in reversed(new_tool_messages):
                    message.pretty_print()

    