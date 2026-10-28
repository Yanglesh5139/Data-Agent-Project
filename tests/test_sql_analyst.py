import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import builtins
import importlib
import unittest
from unittest.mock import Mock, mock_open, patch

from models.schema import AgentSchema, JudgeSchema, SQLGenerationSchema


def _load_module():
    """Import the module without connecting to or writing to the database."""
    cursor = Mock()
    cursor.fetchall.return_value = []
    connection = Mock()
    connection.cursor.return_value = cursor

    with (
        patch("psycopg2.connect", return_value=connection),
        patch.object(builtins, "open", mock_open()),
    ):
        return importlib.import_module("agent.sql_analyst")


sql_analyst = _load_module()


class SqlAnalystHelpersTests(unittest.TestCase):
    def test_clean_sql_removes_markdown_fence_and_trailing_semicolons(self):
        self.assertEqual(
            sql_analyst._clean_sql("```sql\nSELECT 1;\n```") ,
            "SELECT 1",
        )

    def test_enforce_row_limit_appends_limit_only_when_missing(self):
        self.assertEqual(
            sql_analyst._enforce_row_limit("SELECT * FROM users"),
            "SELECT * FROM users\nLIMIT 10",
        )
        self.assertEqual(
            sql_analyst._enforce_row_limit("SELECT * FROM users LIMIT 3"),
            "SELECT * FROM users LIMIT 3",
        )

    def test_route_after_safety_is_case_insensitive(self):
        self.assertEqual(
            sql_analyst.route_after_safety(AgentSchema(user_input="q", is_safe="Yes")),
            "execute_sql",
        )
        self.assertEqual(
            sql_analyst.route_after_safety(AgentSchema(user_input="q", is_safe="No")),
            "canceled_sql",
        )


class SqlAnalystNodeTests(unittest.TestCase):
    def test_curate_question_updates_query_and_messages(self):
        state = AgentSchema(user_input="  How many users?  ")
        llm = Mock()
        llm.invoke.return_value.content = "  Count the users  "

        with patch.object(sql_analyst, "switch_llm", return_value=llm) as switch:
            result = sql_analyst.curate_question(state)

        switch.assert_called_once_with(sql_analyst.LLMTier.LOW)
        self.assertEqual(result.curated_query, "Count the users")
        self.assertEqual(result.messages[-1].content, "Count the users")

    def test_generate_prompt_query_context_uses_schema_and_row_limit(self):
        state = AgentSchema(user_input="q", curated_query="Count the users")
        db = Mock()
        db.schema_details.return_value = "TABLE users (id integer)"

        with patch.object(sql_analyst, "get_db_util", return_value=db):
            result = sql_analyst.generate_prompt_query_context(state)

        db.schema_details.assert_called_once_with(sql_analyst.SQLConfig.SCHEMA_NAME)
        self.assertIn("Count the users", result.prompt_query_context)
        self.assertIn("TABLE users (id integer)", result.prompt_query_context)
        self.assertIn(str(sql_analyst.SQLConfig.DEFAULT_ROW_LIMIT), result.prompt_query_context)

    def test_generate_sql_cleans_generated_sql_and_adds_limit(self):
        state = AgentSchema(user_input="q", prompt_query_context="prompt")
        llm = Mock()
        llm.with_structured_output.return_value = llm
        llm.invoke.return_value = SQLGenerationSchema(sql="```sql\nSELECT * FROM users;\n```")

        with patch.object(sql_analyst, "switch_llm", return_value=llm):
            result = sql_analyst.generate_sql(state)

        self.assertEqual(result.generated_sql_query, "SELECT * FROM users\nLIMIT 10")
        self.assertEqual(result.comments, "")

    def test_generate_sql_records_llm_errors_and_clears_sql(self):
        state = AgentSchema(user_input="q", prompt_query_context="prompt")
        llm = Mock()
        llm.with_structured_output.return_value = llm
        llm.invoke.side_effect = RuntimeError("provider unavailable")

        with patch.object(sql_analyst, "switch_llm", return_value=llm):
            result = sql_analyst.generate_sql(state)

        self.assertEqual(result.generated_sql_query, "")
        self.assertIn("SQL generation failed: provider unavailable", result.comments)

    def test_is_safe_rejects_precheck_failure_without_calling_judge(self):
        state = AgentSchema(user_input="q", generated_sql_query="DELETE FROM users")
        switch = Mock()

        with patch.object(sql_analyst, "switch_llm", switch):
            result = sql_analyst.is_safe(state)

        switch.assert_not_called()
        self.assertEqual(result.is_safe, "No")
        self.assertIn("Precheck failed", result.comments)

    def test_is_safe_uses_judge_result_for_safe_sql(self):
        state = AgentSchema(user_input="q", generated_sql_query="SELECT * FROM users")
        llm = Mock()
        llm.with_structured_output.return_value = llm
        llm.invoke.return_value = JudgeSchema(answer="Yes", comments="Read-only query")

        with patch.object(sql_analyst, "switch_llm", return_value=llm):
            result = sql_analyst.is_safe(state)

        self.assertEqual(result.is_safe, "Yes")
        self.assertEqual(result.comments, "Read-only query")

    def test_is_safe_fails_closed_when_judge_errors_or_returns_none(self):
        for judge_result in (RuntimeError("judge unavailable"), None):
            state = AgentSchema(user_input="q", generated_sql_query="SELECT 1")
            llm = Mock()
            llm.with_structured_output.return_value = llm
            if isinstance(judge_result, Exception):
                llm.invoke.side_effect = judge_result
            else:
                llm.invoke.return_value = judge_result

            with patch.object(sql_analyst, "switch_llm", return_value=llm):
                result = sql_analyst.is_safe(state)

            self.assertEqual(result.is_safe, "No")
            self.assertTrue(result.comments)

    def test_canceled_sql_sets_final_response_and_appends_message(self):
        state = AgentSchema(user_input="q", comments="DELETE is forbidden")

        result = sql_analyst.canceled_sql(state)

        self.assertIn("DELETE is forbidden", result.final_response)
        self.assertEqual(result.messages[-1].content, result.final_response)

    def test_execute_sql_stores_result_and_handles_errors(self):
        state = AgentSchema(user_input="q", generated_sql_query="SELECT 1")
        db = Mock()
        db.execute_sql.return_value = [(1,)]

        with patch.object(sql_analyst, "get_db_util", return_value=db):
            result = sql_analyst.execute_sql(state)

        db.execute_sql.assert_called_once_with("SELECT 1")
        self.assertEqual(result.sql_query_execution_result, "[(1,)]")

        db.execute_sql.side_effect = RuntimeError("connection lost")
        with patch.object(sql_analyst, "get_db_util", return_value=db):
            result = sql_analyst.execute_sql(state)
        self.assertEqual(result.sql_query_execution_result, "Execution error: connection lost")

    def test_represent_node_formats_response_and_appends_message(self):
        state = AgentSchema(
            user_input="q",
            curated_query="Count the users",
            sql_query_execution_result="[(3,)]",
        )
        llm = Mock()
        llm.invoke.return_value.content = "  There are 3 users.  "

        with patch.object(sql_analyst, "switch_llm", return_value=llm):
            result = sql_analyst.represent_node(state)

        self.assertEqual(result.final_response, "There are 3 users.")
        self.assertEqual(result.messages[-1].content, "There are 3 users.")


class SqlAnalystGraphTests(unittest.TestCase):
    def test_build_graph_compiles(self):
        graph = sql_analyst.build_graph().compile()

        self.assertIsNotNone(graph)
        self.assertIn("generate_sql", graph.get_graph().nodes)
        self.assertIn("is_safe", graph.get_graph().nodes)


if __name__ == "__main__":
    unittest.main()
