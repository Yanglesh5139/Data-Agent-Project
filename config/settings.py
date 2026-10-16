"""Application-wide configuration and LLM tier definitions."""
from enum import Enum
from dataclasses import dataclass


class LLMTier(str, Enum):
    """Named LLM tiers used across the agent graph."""
    LOW = "low"       # cheap/fast — curation, final formatting
    MEDIUM = "medium" # balanced — SQL generation, safety judge
    HIGH =   "high"       #for advanced tasks

class SQLConfig:
    """SQL generation and safety constraints."""
    SCHEMA_NAME: str = "public"
    DEFAULT_ROW_LIMIT: int = 10
    MAX_SQL_LENGTH: int = 10_000  # reject absurdly long queries


@dataclass(frozen=True)
class ForbiddenKeywords:
    """Keywords that must never appear in a read-only query.

    Sorted longest-first so multi-word phrases match before their parts.
    """
    keywords: tuple[str, ...] = (
        # DML
        "INSERT", "UPDATE", "DELETE", "MERGE", "REPLACE", "UPSERT",
        # DDL
        "DROP", "ALTER", "TRUNCATE", "CREATE", "RENAME",
        # DCL
        "GRANT", "REVOKE",
        # TCL
        "COMMIT", "ROLLBACK", "SAVEPOINT", "BEGIN",
        # Execution / side-effects
        "EXEC", "EXECUTE", "CALL",
        "INTO OUTFILE", "INTO DUMPFILE", "FOR UPDATE", "FOR SHARE",
        "COPY ", "PG_READ_FILE", "PG_WRITE_FILE", "LO_IMPORT", "LO_EXPORT",
        "DBLINK", "SET ROLE", "SET SESSION AUTHORIZATION",
    )