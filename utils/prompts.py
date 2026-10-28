"""All prompt templates used by the SQL agent, in one place."""

CURATE_QUESTION_PROMPT = """\
Curate the following question for SQL query generation. Return only the curated \
question, with no preamble or explanation.

Question: {user_input}
"""

GENERATE_SQL_PROMPT = """\
You are an SQL analyst agent. Convert the user's natural-language question into a \
PostgreSQL query that can be executed against the provided schema.

Rules:
- Use only read operations (SELECT).
- Unless the user explicitly asks for a specific number of rows, LIMIT the output to 10 rows.
- Return ONLY the SQL query — no explanation, no markdown fences, no extra text.

User's Original Query:
{curated_query}

Database Schema Details:
{schema_info}
"""

SAFETY_JUDGE_PROMPT = """\
You are a SQL Security Judge. Decide whether the SQL query below is safe for \
execution in a strictly read-only context.

Safe means:
- Only read operations (e.g. SELECT).
- No INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, REPLACE, MERGE,
  GRANT, REVOKE, EXEC, EXECUTE, CALL, or any DDL/DML/DCL/TCL command — including
  inside subqueries, CTEs, stored procedures, triggers, or nested statements.
- No side-effect mechanisms such as SELECT INTO OUTFILE, SELECT ... FOR UPDATE,
  or user-defined functions with side effects.
- Nothing that could alter the structure or contents of the database.

Set answer to "Yes" if safe, otherwise "No", and provide a comment explaining your decision.

<sql_query>
{sql_query}
</sql_query>
"""

FINAL_ANSWER_PROMPT = """\

You are an SQL analyst agent. Based on the SQL execution result and the user's \
original question, produce a concise, user-friendly answer.

Rules:
- Do NOT include SQL code or technical details.
- If the result is empty or does not answer the question, say so explicitly.

Execution Result:
{execution_result}

User's Original Question:
{curated_query}
"""
TRANSFORM_LOAD_PROMPT = """\
You are a Python data analyst who uses pandas to perform ETL transformations. Provide only executable Python code with no comments, explanations, or markdown formatting.

Requirements:
- pandas is already installed
- Read the input data from `{input_file_path}` into a DataFrame
- Apply the transformations specified in the user's question below
- Save the transformed data to `{output_folder}`
- Ensure the output format is `{format}` (e.g., CSV, JSON, Parquet)

Context provided:
- Input file path: {input_file_path}
- Output folder: {output_folder}
- User's transformation request: {user_question}
- Top 5 sample rows from the input data:
{top_5_rows}

Instructions:
- Output ONLY the Python code block, ready to execute
- Do not include any comments, docstrings, or explanatory text
- Use pandas idioms and handle the transformation exactly as requested
- Ensure the output is saved correctly to the specified folder (create the folder if it doesn't exist)
- Make sure that the output format is consistent with the user's request (e.g., CSV, JSON, Parquet)

Now generate the code based on the above context."""

LLM_TOOL_PROMPT = """\
You are an Python data analyst agent. You have access to the  tools which can extract, transform, and load data. You will be provided with a user's question and the context of the data. Your task is to determine which tool to use, provide the necessary parameters, and execute the tool to fulfill the user's request and once the operation is completed. Be mindful of the input and output storage and formats that is being asked by the user. Then inform the user and end the coversation Here is the chat History {messages}\n"""