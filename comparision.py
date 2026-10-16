"""SQL analyst agent graph.

Flow:
    curate_question
        -> generate_prompt_query_context
        -> generate_sql
        -> is_safe  (deterministic precheck + LLM judge)
            ├── execute_sql -> represent_node -> END
            └── canceled_sql -> END
"""
import os
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END


from models.schema import AgentSchema, JudgeSchema, SQLGenerationSchema
from utils.db_connection import get_db_util
from utils.llm_switch import switch_llm
from utils.prompts import (
    CURATE_QUESTION_PROMPT,
    FINAL_ANSWER_PROMPT,
    GENERATE_SQL_PROMPT,
    SAFETY_JUDGE_PROMPT,
)
from utils.safety import precheck_sql
from config.settings import LLMTier,SQLConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SQL_FENCE_RE = re.compile(r"^```(?:sql)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
_HAS_LIMIT_RE = re.compile(r"\bLIMIT\b", re.IGNORECASE)


def _clean_sql(sql: str) -> str:
    """Strip markdown fences and trailing semicolons."""
    cleaned = _SQL_FENCE_RE.sub("", sql or "").strip()
    return cleaned.rstrip(";").strip()


def _enforce_row_limit(sql: str, limit: int = SQLConfig.DEFAULT_ROW_LIMIT) -> str:
    """Append LIMIT if the query lacks one (best-effort; safe on non-CTE queries)."""
    if _HAS_LIMIT_RE.search(sql):
        return sql
    return f"{sql}\nLIMIT {limit}"


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def curate_question(state: AgentSchema) -> AgentSchema:
    """Rewrite the user's question for clarity before SQL generation."""
    llm = switch_llm(LLMTier.LOW)
    response = llm.invoke(CURATE_QUESTION_PROMPT.format(user_input=state.user_input)).content
    state.curated_query = response.strip()
    state.messages = state.messages + [HumanMessage(content=state.curated_query)]
    return state


def generate_prompt_query_context(state: AgentSchema) -> AgentSchema:
    """Fetch DB schema and build the SQL-generation prompt."""
    schema_info = get_db_util().schema_details(SQLConfig.SCHEMA_NAME)
    state.prompt_query_context = GENERATE_SQL_PROMPT.format(
        curated_query=state.curated_query,
        schema_info=schema_info,
        row_limit=SQLConfig.DEFAULT_ROW_LIMIT,
    )
    return state


def generate_sql(state: AgentSchema) -> AgentSchema:
    """Generate SQL via structured output, then post-process it."""
    llm = switch_llm(LLMTier.MEDIUM).with_structured_output(SQLGenerationSchema)
    try:
        result = llm.invoke(state.prompt_query_context)
        raw_sql = result.sql if result else ""
    except Exception as exc:
        state.generated_sql_query = ""
        state.comments = f"SQL generation failed: {exc}"
        return state

    sql = _enforce_row_limit(_clean_sql(raw_sql))
    state.generated_sql_query = sql
    return state


def is_safe(state: AgentSchema) -> AgentSchema:
    """Two-layer safety: fast deterministic precheck, then LLM judge.

    Fails closed — any error or ambiguity results in is_safe = "No".
    """
    sql = state.generated_sql_query

    # Layer 1: deterministic precheck (cheap, catches obvious violations)
    verdict = precheck_sql(sql)
    if not verdict.is_safe:
        state.is_safe = "No"
        state.comments = f"Precheck failed: {verdict.reason}"
        return state

    # Layer 2: LLM judge (handles nuanced cases)
    judge = switch_llm(LLMTier.MEDIUM).with_structured_output(JudgeSchema)
    prompt = SAFETY_JUDGE_PROMPT.format(sql_query=sql)

    try:
        result = judge.invoke(prompt)
    except Exception as exc:
        state.is_safe = "No"
        state.comments = f"Judge invocation failed: {exc}"
        return state

    if result is None:
        state.is_safe = "No"
        state.comments = "Judge returned no response."
        return state

    state.is_safe = result.answer
    state.comments = result.comments
    return state


def canceled_sql(state: AgentSchema) -> AgentSchema:
    """Terminal node for queries that failed the safety check."""
    state.final_response = (
        "The generated SQL query was deemed unsafe to execute. "
        f"Reason: {state.comments}. Aborting."
    )
    state.messages = state.messages + [AIMessage(content=state.final_response)]
    return state


def execute_sql(state: AgentSchema) -> AgentSchema:
    """Run the approved SQL and store the raw result."""
    try:
        state.sql_query_execution_result = str(
            get_db_util().execute_sql(state.generated_sql_query)
        )
    except Exception as exc:
        state.sql_query_execution_result = f"Execution error: {exc}"
    return state


def represent_node(state: AgentSchema) -> AgentSchema:
    """Turn the SQL result into a user-friendly final answer."""
    llm = switch_llm(LLMTier.LOW)
    prompt = FINAL_ANSWER_PROMPT.format(
        execution_result=state.sql_query_execution_result,
        curated_query=state.curated_query,
    )
    response = llm.invoke(prompt).content.strip()
    state.final_response = response
    state.messages = state.messages + [AIMessage(content=response)]
    return state


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_safety(state: AgentSchema) -> str:
    """Conditional edge: route to execution or cancellation."""
    return "execute_sql" if state.is_safe == "Yes" else "canceled_sql"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(AgentSchema)

    graph.add_node("curate_question", curate_question)
    graph.add_node("generate_prompt_query_context", generate_prompt_query_context)
    graph.add_node("generate_sql", generate_sql)
    graph.add_node("is_safe", is_safe)
    graph.add_node("canceled_sql", canceled_sql)
    graph.add_node("execute_sql", execute_sql)
    graph.add_node("represent_node", represent_node)

    graph.add_edge(START, "curate_question")
    graph.add_edge("curate_question", "generate_prompt_query_context")
    graph.add_edge("generate_prompt_query_context", "generate_sql")
    graph.add_edge("generate_sql", "is_safe")

    graph.add_conditional_edges(
        "is_safe",
        route_after_safety,
        {"execute_sql": "execute_sql", "canceled_sql": "canceled_sql"},
    )

    graph.add_edge("canceled_sql", END)
    graph.add_edge("execute_sql", "represent_node")
    graph.add_edge("represent_node", END)

    return graph


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _save_graph_diagram(graph, path: str = "sql_analyst.png") -> None:
    """Best-effort diagram export (requires IPython + graphviz)."""
    try:
        from IPython.display import Image
        img = Image(graph.get_graph().draw_mermaid_png())
        with open(path, "wb") as f:
            f.write(img.data)
    except Exception:
        pass


def _print_result(result: dict) -> None:
    separator = "\n" + "*" * 60 + "\n"
    sections = [
        ("MESSAGES", result.get("messages")),
        ("CURATED QUERY", result.get("curated_query")),
        ("GENERATED SQL", result.get("generated_sql_query")),
        ("SAFETY VERDICT", f"{result.get('is_safe')} — {result.get('comments')}"),
        ("EXECUTION RESULT", result.get("sql_query_execution_result")),
        ("FINAL RESPONSE", result.get("final_response")),
    ]
    for title, body in sections:
        print(f"{title}:")
        print(body)
        print(separator, end="")


def main() -> None:
    graph = build_graph().compile()
    _save_graph_diagram(graph)

    initial_state = {
        "messages": [],
        "user_input": (
            "What is the average rating of the rider and also the "
            "driver name with highest rating?"
        ),
    }

    result = graph.invoke(initial_state)
    _print_result(result)


if __name__ == "__main__":
    main()