#!/usr/bin/env python3
"""Seed database with sample data"""

import asyncio
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import get_db_context
from core.models import Person


async def seed_data():
    """Seed database with sample data"""
    sample_persons = [
        {"id": "john_doe", "name": "John Doe"},
        {"id": "jane_smith", "name": "Jane Smith"},
        {"id": "bob_wilson", "name": "Bob Wilson"},
    ]
    
    async with get_db_context() as session:
        for person_data in sample_persons:
            person = Person(**person_data)
            session.add(person)
        
        await session.commit()
        print(f"Seeded {len(sample_persons)} persons")


if __name__ == "__main__":
    asyncio.run(seed_data())