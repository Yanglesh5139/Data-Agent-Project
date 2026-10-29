from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent import etl_analyst
from config.settings import LLMTier
from models.schema import ETLGenerationSchema


def test_extract_load_tool_delegates_to_etl_tools(monkeypatch):
    etl = Mock()
    etl.extract_load.return_value = "saved.csv"
    constructor = Mock(return_value=etl)
    monkeypatch.setattr(etl_analyst, "ETLTools", constructor)

    result = etl_analyst.extract_load_tool.invoke(
        {
            "url": ["https://example.test/data.json"],
            "output_path": "data/extracted",
            "format": "json",
            "mode": "append",
        }
    )

    assert result == "saved.csv"
    constructor.assert_called_once_with()
    etl.extract_load.assert_called_once_with(
        ["https://example.test/data.json"],
        "data/extracted",
        format="json",
        mode="append",
    )


def test_transform_load_tool_uses_context_and_executes_generated_code(monkeypatch):
    etl = Mock()
    etl.transform_load_context.return_value = "column\nvalue"
    etl.extract_python_code.return_value = "df = df.head(1)"
    etl.execute_code.return_value = "Code executed successfully."
    monkeypatch.setattr(etl_analyst, "ETLTools", Mock(return_value=etl))

    llm = Mock()
    llm.invoke.return_value = AIMessage(content="```python\ndf = df.head(1)\n```")
    switch_llm = Mock(return_value=llm)
    monkeypatch.setattr(etl_analyst, "switch_llm", switch_llm)

    result = etl_analyst.transform_load_tool.invoke(
        {
            "input_file_path": "data/input.csv",
            "output_path": "data/transformed",
            "format": "json",
            "user_question": "Keep only the id column",
        }
    )

    etl.transform_load_context.assert_called_once_with(
        "data/input.csv", "data/transformed", format="json"
    )
    switch_llm.assert_called_once_with(LLMTier.HIGH)
    llm.invoke.assert_called_once()
    etl.extract_python_code.assert_called_once_with(llm.invoke.return_value)
    etl.execute_code.assert_called_once_with("df = df.head(1)")
    assert "data/transformed" in result
    assert "df = df.head(1)" in result


def test_llm_node_appends_ai_message_to_state(monkeypatch):
    state = ETLGenerationSchema(messages=[HumanMessage(content="extract data")])
    ai_message = AIMessage(content="Done")
    bound_llm = Mock()
    bound_llm.invoke.return_value = ai_message
    llm = Mock()
    llm.bind_tools.return_value = bound_llm
    monkeypatch.setattr(etl_analyst, "switch_llm", Mock(return_value=llm))

    result = etl_analyst.llm_node(state)

    assert result is state
    assert result.messages[-1] is ai_message
    llm.bind_tools.assert_called_once_with(etl_analyst.TOOLS)
    bound_llm.invoke.assert_called_once()


def test_tools_node_dispatches_known_tool_and_preserves_tool_call_id(monkeypatch):
    tool = Mock()
    tool.invoke.return_value = {"status": "success"}
    monkeypatch.setattr(etl_analyst, "TOOLS_BY_NAME", {"extract_load_tool": tool})
    state = ETLGenerationSchema(
        messages=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "extract_load_tool",
                        "args": {"url": "https://example.test/data.json"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    )

    result = etl_analyst.tools_node(state)

    tool.invoke.assert_called_once_with({"url": "https://example.test/data.json"})
    message = result.messages[-1]
    assert isinstance(message, ToolMessage)
    assert message.tool_call_id == "call-1"
    assert message.content == "{'status': 'success'}"


def test_tools_node_reports_unknown_tool():
    state = ETLGenerationSchema(
        messages=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "missing_tool",
                        "args": {},
                        "id": "call-unknown",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    )

    result = etl_analyst.tools_node(state)

    message = result.messages[-1]
    assert isinstance(message, ToolMessage)
    assert message.tool_call_id == "call-unknown"
    assert message.content == "Unknown tool: missing_tool"


def test_tools_node_converts_tool_exception_to_tool_message(monkeypatch):
    tool = Mock()
    tool.invoke.side_effect = RuntimeError("boom")
    monkeypatch.setattr(etl_analyst, "TOOLS_BY_NAME", {"extract_load_tool": tool})
    state = ETLGenerationSchema(
        messages=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "extract_load_tool",
                        "args": {},
                        "id": "call-error",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    )

    result = etl_analyst.tools_node(state)

    assert result.messages[-1].content == "Tool error: RuntimeError('boom')"


@pytest.mark.parametrize(
    ("tool_calls", "expected"),
    [
        ([], etl_analyst.END),
        (
            [
                {
                    "name": "extract_load_tool",
                    "args": {},
                    "id": "call-route",
                    "type": "tool_call",
                }
            ],
            "tools_node",
        ),
    ],
)
def test_route_after_llm(tool_calls, expected):
    message = AIMessage(content="", tool_calls=tool_calls)
    state = ETLGenerationSchema(messages=[message])

    assert etl_analyst.route_after_llm(state) == expected


def test_build_graph_compiles():
    graph = etl_analyst.build_graph()

    assert graph is not None
    assert set(graph.nodes) >= {"llm_node", "tools_node"}
