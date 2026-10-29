from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_LOG_DIR = os.environ.get("ETL_LOG_DIR", os.path.join(_PROJECT_ROOT, "logs"))
_LOG_LEVEL_CONSOLE = os.environ.get("ETL_LOG_LEVEL", "INFO").upper()
_LOG_LEVEL_FILE = os.environ.get("ETL_LOG_FILE_LEVEL", "DEBUG").upper()
_USE_JSON = os.environ.get("ETL_LOG_JSON", "0").lower() in ("1", "true", "yes")

_LOGGER_ROOT_NAME = "etl"
_CONFIGURED = False  # guard against duplicate handlers on re-import


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

class ConsoleFormatter(logging.Formatter):
    """Human-friendly, colorized console output."""

    COLORS = {
        "DEBUG": "\033[36m",     # cyan
        "INFO": "\033[32m",      # green
        "WARNING": "\033[33m",   # yellow
        "ERROR": "\033[31m",     # red
        "CRITICAL": "\033[41m",  # red background
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        original = record.levelname
        record.levelname = f"{color}{original:<8}{self.RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original


class JsonFormatter(logging.Formatter):
    """One JSON object per line — ideal for ingestion by ELK/Loki/etc."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
            "process": record.process,
            "thread": record.threadName,
        }
        # Attach any custom fields set via logger.info(..., extra={...})
        for key, value in record.__dict__.items():
            if key not in _RESERVED_LOG_ATTRS and not key.startswith("_"):
                try:
                    json.dumps(value)  # only include JSON-serializable values
                    payload[key] = value
                except (TypeError, ValueError):
                    payload[key] = repr(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


_RESERVED_LOG_ATTRS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "asctime", "taskName",
}


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_logging(
    log_dir: Optional[str] = None,
    console_level: Optional[str] = None,
    file_level: Optional[str] = None,
    json_file: Optional[bool] = None,
    force: bool = False,
) -> logging.Logger:
    """
    Configure the project-wide root logger once.

    All loggers obtained via `get_logger(...)` are children of `logging.getLogger("etl")`,
    so they inherit these handlers automatically.
    """
    global _CONFIGURED
    root = logging.getLogger(_LOGGER_ROOT_NAME)

    if _CONFIGURED and not force:
        return root

    # Reset handlers (in case of force-reconfigure)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    root.setLevel(logging.DEBUG)
    root.propagate = False  # do not bubble to Python's root logger

    resolved_dir = log_dir or _LOG_DIR
    resolved_console_level = (console_level or _LOG_LEVEL_CONSOLE).upper()
    resolved_file_level = (file_level or _LOG_LEVEL_FILE).upper()
    use_json = _USE_JSON if json_file is None else json_file

    os.makedirs(resolved_dir, exist_ok=True)

    # --- Console handler -----------------------------------------------------
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(resolved_console_level)
    console_handler.setFormatter(
        ConsoleFormatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(console_handler)

    # --- Daily rotating file handler (all logs, per-day file) ---------------
    daily_path = os.path.join(
        resolved_dir, f"etl_{datetime.now().strftime('%Y%m%d')}.log"
    )
    file_handler = logging.handlers.RotatingFileHandler(
        daily_path, maxBytes=20 * 1024 * 1024, backupCount=10, encoding="utf-8"
    )
    file_handler.setLevel(resolved_file_level)
    if use_json:
        file_handler.setFormatter(JsonFormatter())
    else:
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | "
                    "%(module)s.%(funcName)s:%(lineno)d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    root.addHandler(file_handler)

    # --- Separate errors-only file (easy triage) ----------------------------
    errors_path = os.path.join(resolved_dir, "errors.log")
    error_handler = logging.handlers.RotatingFileHandler(
        errors_path, maxBytes=20 * 1024 * 1024, backupCount=10, encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | "
                "%(module)s.%(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(error_handler)

    _CONFIGURED = True
    root.info(
        "Logging configured | dir=%s | console=%s | file=%s | json=%s",
        resolved_dir, resolved_console_level, resolved_file_level, use_json,
    )
    return root


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a child logger of the project root logger.

    Call `setup_logging()` once at app startup before using any logger.
    If you forget, this will auto-configure with defaults on first use.
    """
    if not _CONFIGURED:
        setup_logging()
    suffix = name or "app"
    # Prefix with "etl." so all project loggers share the same handlers.
    full_name = suffix if suffix.startswith(f"{_LOGGER_ROOT_NAME}.") else f"{_LOGGER_ROOT_NAME}.{suffix}"
    return logging.getLogger(full_name)


# ---------------------------------------------------------------------------
# Convenience context helper
# ---------------------------------------------------------------------------

def log_context(**fields: Any) -> Dict[str, Any]:

    """
    Build an `extra=` dict to enrich log records.

    Example:
        logger.info("user logged in", extra=log_context(user_id=42, tenant="acme"))
    """
    return {"extra_fields": fields}


def _truncate(value, limit: int = 500) -> str:
    """Shorten long strings/objects so log lines stay readable."""
    text = str(value)
    return text if len(text) <= limit else text[:limit] + f"... [{len(text) - limit} chars truncated]"