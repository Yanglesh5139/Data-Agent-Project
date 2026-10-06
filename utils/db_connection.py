import os
from utils.database import DatabaseUtil


def get_db_util() -> DatabaseUtil:
    """Build a DatabaseUtil from environment variables."""
    return DatabaseUtil({
        "host": os.getenv("HOST"),
        "port": os.getenv("PORT"),
        "dbname": os.getenv("DATABASE"),
        "user": os.getenv("USER"),
        "password": os.getenv("PASSWORD"),
    })