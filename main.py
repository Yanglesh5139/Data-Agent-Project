import logging
from time import perf_counter

from langchain_core.messages import HumanMessage, ToolMessage

from utils.logger import get_logger, setup_logging, _truncate

logger = get_logger(__name__)

SQL_QUESTION = (
    "What is the average rating of the rider and also the "
    "driver name with highest rating?"
)

ETL_PROMPT = (
    "Extract data from 'https://breedhealthscore.com/api/breeds.json' "
    "and save it to 'data/extracted' in csv format. "
    "Then use the extracted file as input, keep only the columns "
    "'breed', 'health_score', and 'size', and save the transformed "
    "data to 'data/transformed' in json format."
)


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

    # Mirror the structured outcome into the log so it's auditable later.
    logger.info("=" * 60)
    logger.info("Run summary")
    logger.info("Curated query: %s", _truncate(result.get("curated_query"), 300))
    logger.info("Generated SQL: %s", _truncate(result.get("generated_sql_query"), 800))
    logger.info(
        "Safety verdict: %s | comments=%s",
        result.get("is_safe"),
        _truncate(result.get("comments"), 300),
    )
    logger.info(
        "Execution result: %s",
        _truncate(result.get("sql_query_execution_result"), 1000),
    )
    logger.info("Final response: %s", _truncate(result.get("final_response"), 500))
    logger.info("=" * 60)



def run_sql_analyst(question: str) -> dict:
    """Build the SQL graph, execute the question, and display its result."""
    from agent.sql_analyst import build_graph

    logger.info("[SQL] Building graph")
    graph = build_graph().compile()
    result = graph.invoke({"messages": [], "user_input": question})
    _print_result(result)
    return result


def run_etl_analyst(prompt: str) -> None:
    """Stream ETL updates, displaying only newly added messages."""
    from agent.etl_analyst import build_graph
    from models.schema import ETLGenerationSchema

    # ETL build_graph() already returns a compiled graph.
    graph = build_graph()
    initial_state = ETLGenerationSchema(messages=[HumanMessage(content=prompt)])

    for update in graph.stream(initial_state, stream_mode="updates"):
        for node_name, node_state in update.items():
            logger.info("[ETL] Stream update | node=%s", node_name)
            print(f"\n{'=' * 20} {node_name} {'=' * 20}")
            messages = (
                node_state.get("messages", [])
                if isinstance(node_state, dict)
                else node_state.messages
            )
            if not messages:
                continue

            if node_name == "llm_node":
                messages[-1].pretty_print()
            elif node_name == "tools_node":
                new_tool_messages = []
                for message in reversed(messages):
                    if not isinstance(message, ToolMessage):
                        break
                    new_tool_messages.append(message)
                for message in reversed(new_tool_messages):
                    message.pretty_print()


def main() -> int:
    """Run both workflows once, recording errors independently."""
    setup_logging()
    started_at = perf_counter()
    failed = []
    workflows = (
        ("SQL", run_sql_analyst, SQL_QUESTION),
        ("ETL", run_etl_analyst, ETL_PROMPT),
    )

    try:
        for name, run_workflow, prompt in workflows:
            workflow_started = perf_counter()
            logger.info("[%s] Starting | prompt=%s", name, _truncate(prompt, 400))
            try:
                run_workflow(prompt)
            except Exception:
                failed.append(name)
                logger.exception("[%s] Workflow raised an exception", name)
            else:
                logger.info("[%s] Workflow returned normally", name)
            finally:
                logger.info("[%s] Finished | elapsed=%.2fs", name, perf_counter() - workflow_started)
    except KeyboardInterrupt:
        logger.warning("Driver interrupted by user")
        return 130
    finally:
        logger.info("Driver finished | elapsed=%.2fs", perf_counter() - started_at)
        logging.shutdown()

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
