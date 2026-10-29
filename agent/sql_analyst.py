import os
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ---------------------------------------------------------------------------
# Centralized logging
# ---------------------------------------------------------------------------
from utils.logger import get_logger, setup_logging, _truncate

setup_logging()                       # idempotent; safe if called elsewhere too
logger = get_logger(__name__)


from utils.llm_switch import switch_llm
from models.schema import AgentSchema, JudgeSchema, SQLGenerationSchema
from langchain_core.messages import AIMessage, HumanMessage
from utils.database import DatabaseUtil
from config.settings import LLMTier, SQLConfig
from langgraph.graph import StateGraph, START, END
from utils.prompts import (
    CURATE_QUESTION_PROMPT,
    FINAL_ANSWER_PROMPT,
    GENERATE_SQL_PROMPT,
    SAFETY_JUDGE_PROMPT,
)
from utils.db_connection import get_db_util
from utils.safety import precheck_sql


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
        logger.debug("_enforce_row_limit | LIMIT already present, unchanged")
        return sql
    logger.debug("_enforce_row_limit | appending LIMIT %d", limit)
    return f"{sql}\nLIMIT {limit}"


# ---------------------------------------------------------------------------
# AI Agent Nodes
# ---------------------------------------------------------------------------

# --------------------------- Question Curation -----------------------------

def curate_question(state: AgentSchema) -> AgentSchema:
    logger.info(
        "curate_question START | user_input=%s",
        _truncate(state.user_input, 300),
    )
    llm = switch_llm(LLMTier.LOW)  # switching to low level LLM for question curation
    try:
        response = llm.invoke(
            CURATE_QUESTION_PROMPT.format(user_input=state.user_input)
        ).content
    except Exception:
        logger.exception("curate_question LLM invocation FAILED")
        raise

    state.curated_query = response.strip()
    state.messages = state.messages + [HumanMessage(content=state.curated_query)]
    logger.info(
        "curate_question SUCCESS | curated_query=%s",
        _truncate(state.curated_query, 300),
    )
    return state


# --------------------------- Prompt / Context ------------------------------

def generate_prompt_query_context(state: AgentSchema) -> AgentSchema:
    logger.info(
        "generate_prompt_query_context START | schema=%s", SQLConfig.SCHEMA_NAME
    )
    try:
        # To get the DB schema details
        schema_info = get_db_util().schema_details(SQLConfig.SCHEMA_NAME)
    except Exception:
        logger.exception(
            "generate_prompt_query_context schema fetch FAILED | schema=%s",
            SQLConfig.SCHEMA_NAME,
        )
        raise

    logger.debug(
        "generate_prompt_query_context | schema_info=%s", _truncate(schema_info, 1000)
    )

    state.prompt_query_context = GENERATE_SQL_PROMPT.format(
        curated_query=state.curated_query,
        schema_info=schema_info,
        row_limit=SQLConfig.DEFAULT_ROW_LIMIT,
    )
    logger.info(
        "generate_prompt_query_context SUCCESS | prompt_length=%d",
        len(state.prompt_query_context),
    )
    return state


# --------------------------- SQL Generation --------------------------------

def generate_sql(state: AgentSchema):
    logger.info("generate_sql START")
    llm = switch_llm(LLMTier.MEDIUM).with_structured_output(SQLGenerationSchema)
    try:
        result = llm.invoke(state.prompt_query_context)
        raw_sql = result.sql if result else ""
        logger.debug("generate_sql | raw_sql=%s", _truncate(raw_sql))
    except Exception as e:
        logger.exception("generate_sql LLM invocation FAILED")
        state.generated_sql_query = ""
        state.comments = f"SQL generation failed: {e}"
        return state

    sql = _enforce_row_limit(_clean_sql(raw_sql))
    state.generated_sql_query = sql
    logger.info("generate_sql SUCCESS | sql=%s", _truncate(sql, 800))
    return state


# --------------------------- Safety Check ----------------------------------

def is_safe(state: AgentSchema):
    sql = state.generated_sql_query
    logger.info("is_safe START | sql=%s", _truncate(sql, 800))

    # Layer 1: deterministic precheck (cheap, catches obvious violations)
    try:
        verdict = precheck_sql(sql)
    except Exception:
        logger.exception("is_safe precheck FAILED")
        state.is_safe = "No"
        state.comments = "Precheck raised an exception."
        return state

    if not verdict.is_safe:
        logger.warning(
            "is_safe PRECHECK REJECTED | reason=%s", verdict.reason
        )
        state.is_safe = "No"
        state.comments = f"Precheck failed: {verdict.reason}"
        return state

    logger.debug("is_safe precheck PASSED")

    # Layer 2: LLM judge (handles nuanced cases)
    llm_judge = switch_llm(LLMTier.MEDIUM).with_structured_output(JudgeSchema)
    prompt = SAFETY_JUDGE_PROMPT.format(sql_query=sql)

    try:
        result = llm_judge.invoke(prompt)
    except Exception as exc:
        logger.exception("is_safe judge invocation FAILED")
        state.is_safe = "No"
        state.comments = f"Judge invocation failed: {exc}"
        return state

    if result is None:
        logger.error("is_safe judge returned no response")
        state.is_safe = "No"
        state.comments = "Judge failed to return a response."
        return state

    data = result.model_dump()
    state.is_safe = data["answer"]
    state.comments = data["comments"]

    if state.is_safe.lower() == "yes":
        logger.info(
            "is_safe SUCCESS | verdict=YES | comments=%s",
            _truncate(state.comments, 300),
        )
    else:
        logger.warning(
            "is_safe JUDGE REJECTED | comments=%s",
            _truncate(state.comments, 300),
        )
    return state


def canceled_sql(state: AgentSchema):
    state.final_response = (
        f"The generated SQL query was deemed unsafe to execute. "
        f"Reason: {state.comments}. Aborting."
    )
    state.messages = state.messages + [AIMessage(content=state.final_response)]
    logger.warning(
        "canceled_sql | final_response=%s",
        _truncate(state.final_response, 300),
    )
    return state


# --------------------------- Execution -------------------------------------

def execute_sql(state: AgentSchema):
    logger.info("execute_sql START | sql=%s", _truncate(state.generated_sql_query, 800))
    try:
        state.sql_query_execution_result = str(
            get_db_util().execute_sql(state.generated_sql_query)
        )
        logger.info(
            "execute_sql SUCCESS | result=%s",
            _truncate(state.sql_query_execution_result, 1000),
        )
    except Exception as e:
        logger.exception("execute_sql FAILED")
        state.sql_query_execution_result = f"Execution error: {e}"
    return state


# --------------------------- Representation --------------------------------

def represent_node(state: AgentSchema):
    logger.info("represent_node START")
    llm = switch_llm(LLMTier.LOW)
    prompt = FINAL_ANSWER_PROMPT.format(
        execution_result=state.sql_query_execution_result,
        curated_query=state.curated_query,
    )
    try:
        response = llm.invoke(prompt).content.strip()
    except Exception:
        logger.exception("represent_node LLM invocation FAILED")
        raise

    state.final_response = response
    state.messages = state.messages + [AIMessage(content=response)]
    logger.info(
        "represent_node SUCCESS | final_response=%s",
        _truncate(response, 500),
    )
    return state


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_safety(state: AgentSchema) -> str:
    route = "execute_sql" if state.is_safe.lower() == "yes" else "canceled_sql"
    logger.debug(
        "route_after_safety | verdict=%s | routing_to=%s", state.is_safe, route
    )
    return route


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    logger.info("build_graph START")
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

    logger.info("build_graph END")
    return graph


