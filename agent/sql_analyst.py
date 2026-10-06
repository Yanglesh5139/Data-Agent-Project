import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.llm_switch import switch_llm
from models.schema import AgentSchema,JudgeSchema
from langchain_core.messages import AIMessage, HumanMessage
from utils.database import DatabaseUtil
from langgraph.graph import StateGraph,START,END
from utils.prompts import (
    CURATE_QUESTION_PROMPT,
    FINAL_ANSWER_PROMPT,
    GENERATE_SQL_PROMPT,
    SAFETY_JUDGE_PROMPT,
)
from utils.db_connection import get_db_util

# AI Agent Nodes

##SECTION - Question Curation Node

def curate_question(state: AgentSchema) -> AgentSchema:
   
    llm=switch_llm("low") #*switching to low level LLM for question curation
    response = llm.invoke(CURATE_QUESTION_PROMPT.format(user_input=state.user_input)).content
    state.curated_query = response
    state.messages=state.messages + [HumanMessage(content=response)] #append the curated question to the message list
    return state

def generate_prompt_query_context(state: AgentSchema) -> AgentSchema:
    
    schema_info=get_db_util().schema_details("public")

    
    state.prompt_query_context=GENERATE_SQL_PROMPT.format(
        curated_query=state.curated_query,
        schema_info=schema_info,
    )
    
    return state

def generate_sql(state: AgentSchema):
    llm=switch_llm("medium")
    state.generated_sql_query = llm.invoke(state.prompt_query_context).content.strip()
    return state


def is_safe(state: AgentSchema):
    llm_judge = switch_llm("medium").with_structured_output(JudgeSchema)
    prompt = SAFETY_JUDGE_PROMPT.format(sql_query=state.generated_sql_query)

    try:
        result = llm_judge.invoke(prompt)
    except Exception as exc:
        state.is_safe = "No"
        state.comments = f"Judge invocation failed: {exc}"
        return state

    if result is None:
        state.is_safe = "No"
        state.comments = "Judge failed to return a response."
        return state

    data = result.model_dump()
    state.is_safe = data["answer"]
    state.comments = data["comments"]
    return state

def canceled_sql(state: AgentSchema):
    state.final_response = (
        f"The generated SQL query was deemed unsafe to execute. "
        f"Reason: {state.comments}. Aborting."
    )
    state.messages = state.messages + [AIMessage(content=state.final_response)]
    return state

def execute_sql(state: AgentSchema):

    state.sql_query_execution_result = get_db_util().execute_sql(state.generated_sql_query)
    return state

def represent_node(state: AgentSchema):
    llm = switch_llm("low")
    prompt = FINAL_ANSWER_PROMPT.format(
        execution_result=state.sql_query_execution_result,
        curated_query=state.curated_query,
    )
    response = llm.invoke(prompt).content
    state.final_response = response
    state.messages = state.messages + [AIMessage(content=response)]
    return state

# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_safety(state: AgentSchema) -> str:
    return "execute_sql" if state.is_safe.lower() == "yes" else "canceled_sql"


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

def main() -> None:
    sql_analyst = build_graph().compile()

    # Optional: save the graph diagram (skip in non-dev environments)
    try:
        from IPython.display import Image
        img = Image(sql_analyst.get_graph().draw_mermaid_png())
        with open("sql_analyst.png", "wb") as f:
            f.write(img.data)
    except Exception:
        pass  # Diagram is a nice-to-have, not essential

    initial_state = {
        "messages": [],
        "user_input": (
            "What is the average rating of the rider and also the "
            "driver name with highest rating?"
        ),
    }

    result = sql_analyst.invoke(initial_state)

    separator = "\n" + "*" * 60 + "\n"
    print("MESSAGES:", result["messages"], sep="\n", end=separator)
    print("GENERATED SQL:", result["generated_sql_query"], sep="\n", end=separator)
    print("EXECUTION RESULT:", result["sql_query_execution_result"], sep="\n", end=separator)
    print("PROMPT CONTEXT:", result["prompt_query_context"], sep="\n")


if __name__ == "__main__":
    main()