"""Deterministic safety checks that run BEFORE the LLM judge.

Cheap, fast, and fail-closed: if any forbidden keyword is found, the query is
rejected without an LLM call.
"""
import re
from dataclasses import dataclass

from config.settings import ForbiddenKeywords, SQLConfig


@dataclass(frozen=True)
class SafetyVerdict:
    """Result of the deterministic safety pre-check."""
    is_safe: bool
    reason: str = ""


# Strip string literals and comments so they can't hide keywords
_STRING_LITERAL_RE = re.compile(r"'(?:[^']|'')*'")
_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)


def _strip_noise(sql: str) -> str:
    """Remove string literals and comments to prevent keyword-in-string bypasses."""
    sql = _BLOCK_COMMENT_RE.sub(" ", sql)
    sql = _LINE_COMMENT_RE.sub(" ", sql)
    sql = _STRING_LITERAL_RE.sub("''", sql)
    return sql


def precheck_sql(sql: str) -> SafetyVerdict:
    """Run all deterministic safety checks. Returns a verdict."""
    if not sql or not sql.strip():
        return SafetyVerdict(False, "Empty SQL query.")

    if len(sql) > SQLConfig.MAX_SQL_LENGTH:
        return SafetyVerdict(
            False,
            f"SQL exceeds max length ({SQLConfig.MAX_SQL_LENGTH} chars).",
        )

    cleaned = _strip_noise(sql).upper()

    # Word-boundary match — avoids matching keywords inside identifiers
    for keyword in ForbiddenKeywords().keywords:
        pattern = rf"(?<![A-Z0-9_]){re.escape(keyword)}(?![A-Z0-9_])"
        if re.search(pattern, cleaned):
            return SafetyVerdict(False, f"Forbidden keyword detected: {keyword}")

    # Must begin with SELECT or WITH (CTE starting with SELECT)
    stripped = cleaned.lstrip()
    if not (stripped.startswith("SELECT") or stripped.startswith("WITH")):
        return SafetyVerdict(
            False,
            "Query must start with SELECT or WITH.",
        )

    # Multiple statements are not allowed
    if ";" in cleaned.rstrip().rstrip(";"):
        return SafetyVerdict(False, "Multiple statements are not allowed.")

    return SafetyVerdict(True)