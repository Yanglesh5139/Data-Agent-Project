import os
import sys
from typing import Union, List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain.tools import tool
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

# ---------------------------------------------------------------------------
# Centralized logging
# ---------------------------------------------------------------------------
from utils.logger import get_logger, setup_logging, _truncate

setup_logging()                      # idempotent; safe even if called elsewhere
logger = get_logger(__name__)


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
    logger.info(
        "extract_load_tool START | url=%s | output_path=%s | format=%s | mode=%s",
        _truncate(url, 200), output_path, format, mode,
    )
    try:
        etl = ETLTools()
        result = etl.extract_load(url, output_path, format=format, mode=mode)
        logger.info("extract_load_tool SUCCESS | saved_to=%s", result)
        logger.debug("extract_load_tool raw_result=%s", _truncate(result))
        return result
    except Exception:
        logger.exception(
            "extract_load_tool FAILED | url=%s | output_path=%s",
            _truncate(url, 200), output_path,
        )
        raise


@tool
def transform_load_tool(
    input_file_path: str,
    output_path: str,
    format: str,
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
    logger.info(
        "transform_load_tool START | input=%s | output=%s | format=%s | question=%s",
        input_file_path, output_path, format, _truncate(user_question or "<none>", 300),
    )
    etl = ETLTools()

    try:
        # 1. Sample context so the LLM knows what columns/types it's working with.
        logger.debug("transform_load_tool step 1/3: gathering context sample")
        top_5_rows = etl.transform_load_context(
            input_file_path, output_path, format=format
        )
        logger.debug("transform_load_tool context_sample=%s", _truncate(top_5_rows))

        # 2. Use the high-tier LLM for code generation (more powerful and accurate).
        logger.debug("transform_load_tool step 2/3: invoking LLM for code generation")
        llm = switch_llm(LLMTier.LOW)
        response = llm.invoke(
            TRANSFORM_LOAD_PROMPT.format(
                input_file_path=input_file_path,
                output_folder=output_path,
                format=format,
                top_5_rows=top_5_rows,
                user_question=user_question or "",
            )
        )
        logger.debug("transform_load_tool LLM raw_response=%s", _truncate(response))

        code = etl.extract_python_code(response)
        logger.info("transform_load_tool generated pandas code:\n%s", code)

        # 3. Execute the generated code in the ETL sandbox.
        logger.debug("transform_load_tool step 3/3: executing generated code")
        result = etl.execute_code(code)
        logger.info("transform_load_tool execution_result=%s", _truncate(result))

        logger.info(
            "transform_load_tool SUCCESS | output=%s | format=%s",
            output_path, format,
        )
        return (
            f"The data is transformed and saved to {output_path}. "
            f"Result: {result}\n pandas code executed:\n{code}"
        )
    except Exception:
        logger.exception(
            "transform_load_tool FAILED | input=%s | output=%s",
            input_file_path, output_path,
        )
        raise


# Registry used by the tools node for dispatch by name.
TOOLS = [extract_load_tool, transform_load_tool]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}
logger.debug("Tool registry initialized | tools=%s", list(TOOLS_BY_NAME.keys()))


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
    logger.info("llm_node START | history_length=%d", len(messages))
    logger.debug("llm_node last_message_type=%s", type(messages[-1]).__name__)

    # Bind tools here (not at import time) so the binding always matches the
    # current registry; also keeps module import side-effect free.
    llm_with_tools = switch_llm(LLMTier.LOW).bind_tools(TOOLS)

    # Pass the entire message history so the LLM sees prior tool results.
    try:
        ai_message: AIMessage = llm_with_tools.invoke(
            LLM_TOOL_PROMPT.format(messages=messages)
        )
    except Exception:
        logger.exception("llm_node LLM invocation FAILED")
        raise

    tool_calls = getattr(ai_message, "tool_calls", None) or []
    if tool_calls:
        logger.info(
            "llm_node requested tool_calls | count=%d | names=%s",
            len(tool_calls), [tc.get("name") for tc in tool_calls],
        )
        for tc in tool_calls:
            logger.debug(
                "llm_node tool_call | id=%s | name=%s | args=%s",
                tc.get("id"), tc.get("name"), _truncate(tc.get("args")),
            )
    else:
        logger.info(
            "llm_node produced final answer | content=%s",
            _truncate(getattr(ai_message, "content", "")),
        )

    state.messages = messages + [ai_message]
    logger.info("llm_node END | new_history_length=%d", len(state.messages))
    return state


def tools_node(state: ETLGenerationSchema) -> ETLGenerationSchema:
    """
    Execute every tool call emitted by the most recent AIMessage and append
    the resulting ToolMessages back into the state.
    """
    last_message = state.messages[-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []
    logger.info("tools_node START | tool_calls=%d", len(tool_calls))

    tool_messages: List[ToolMessage] = []
    for idx, tool_call in enumerate(tool_calls, start=1):
        tool_name = tool_call.get("name")
        tool_id = tool_call.get("id")
        tool_args = tool_call.get("args", {})
        logger.info(
            "tools_node executing %d/%d | name=%s | id=%s",
            idx, len(tool_calls), tool_name, tool_id,
        )

        tool_fn = TOOLS_BY_NAME.get(tool_name)
        if tool_fn is None:
            # Unknown tool — surface as a ToolMessage so the LLM can recover.
            logger.warning(
                "tools_node unknown tool requested | name=%s | id=%s",
                tool_name, tool_id,
            )
            tool_messages.append(
                ToolMessage(
                    content=f"Unknown tool: {tool_name}",
                    tool_call_id=tool_id,
                )
            )
            continue

        try:
            observation = tool_fn.invoke(tool_args)
            logger.info(
                "tools_node tool SUCCESS | name=%s | id=%s | result=%s",
                tool_name, tool_id, _truncate(observation),
            )
        except Exception as exc:  # keep the loop alive even if one tool fails
            logger.exception(
                "tools_node tool FAILED | name=%s | id=%s",
                tool_name, tool_id,
            )
            observation = f"Tool error: {exc!r}"

        tool_messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_id)
        )

    state.messages = state.messages + tool_messages
    logger.info(
        "tools_node END | appended=%d | new_history_length=%d",
        len(tool_messages), len(state.messages),
    )
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
    route = "tools_node" if tool_calls else END
    logger.debug(
        "route_after_llm | tool_calls=%d | routing_to=%s", len(tool_calls), route
    )
    return route


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Build and compile the ETL analyst state graph."""
    logger.info("build_graph START")
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

    compiled = graph.compile()
    logger.info("build_graph END | compiled")
    return compiled




