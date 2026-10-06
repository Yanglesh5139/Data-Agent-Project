import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.llm_switch import switch_llm
from models.schema import AgentSchema,JudgeSchema
# from langchain_core import HumanMessage
# from utils.database import DatabaseUtil


llm=switch_llm("medium")
llm_judge=llm.with_structured_output(JudgeSchema)
sql_query="delete * from users where age>10"

prompt=f""" You are a SQL Security Judge whose sole task is to determine whether a given SQL query is safe for execution in a strictly read-only context, meaning the query must only be used for data retrieval and must not modify the database in any way; specifically, the query must contain only read operations such as SELECT and must not contain any INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, REPLACE, MERGE, GRANT, REVOKE, EXEC, EXECUTE, CALL, or any other DDL, DML, DCL, or TCL command, including within subqueries, CTEs, stored procedures, triggers, or nested statements, nor may it invoke functions or mechanisms that could indirectly modify the database such as SELECT INTO OUTFILE, SELECT FOR UPDATE, or user-defined functions with side effects, and the SQL query must not contain any commands that could change the structure or contents of the database; if the SQL query is safe, respond with 'Yes', otherwise respond with 'No' followed by a comment explaining your decision, and your response must follow this exact format: first line either 'Yes' or 'No', and if 'No', a second line containing the explanation comment. Here is the sql query to evaluate {sql_query}"""

print(llm_judge.invoke(prompt))