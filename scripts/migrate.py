#!/usr/bin/env python3
"""Database migration script"""

import asyncio
import os
import sys

from alembic import command
from alembic.config import Config

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_migrations():
    """Run database migrations"""
    alembic_cfg = Config("alembic.ini")
    
    # Override database URL from environment
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        alembic_cfg.set_main_option("sqlalchemy.url", db_url)
    
    command.upgrade(alembic_cfg, "head")
    print("Migrations completed successfully")


if __name__ == "__main__":
    run_migrations()