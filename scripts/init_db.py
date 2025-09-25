#!/usr/bin/env python3
"""Initialize database with sample data"""

import asyncio
import os
import sys
from datetime import datetime
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select

from core.database import get_db_context, engine
from core.models import Base, Person, Face, Enrollment


async def create_tables():
    """Create all database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✓ Database tables created")


async def seed_sample_data():
    """Seed database with sample data"""
    async with get_db_context() as session:
        # Check if data already exists
        result = await session.execute(select(Person))
        if result.scalar_one_or_none():
            print("Database already contains data, skipping seed")
            return

        # Create sample persons
        sample_persons = [
            Person(id="john_doe", name="John Doe"),
            Person(id="jane_smith", name="Jane Smith"),
            Person(id="bob_wilson", name="Bob Wilson"),
            Person(id="alice_brown", name="Alice Brown"),
            Person(id="charlie_davis", name="Charlie Davis"),
        ]

        session.add_all(sample_persons)
        await session.commit()   # ✅ لازم تسجل الأشخاص الأول

        # Create sample enrollments
        enrollments = [
            Enrollment(
                id=uuid.uuid4(),
                person_id=person.id,
                face_count=0,
                status="pending",
                created_at=datetime.utcnow()
            )
            for person in sample_persons
        ]

        session.add_all(enrollments)
        await session.commit()

        print(f"✓ Seeded {len(sample_persons)} sample persons and {len(enrollments)} enrollments")


async def verify_database():
    """Verify database connection and tables"""
    try:
        async with get_db_context() as session:
            # Test query
            result = await session.execute(select(Person))
            persons = result.scalars().all()

            print(f"✓ Database connection successful")
            print(f"  Found {len(persons)} persons in database")

            # Show table counts
            from sqlalchemy import func

            person_count = await session.execute(select(func.count(Person.id)))
            face_count = await session.execute(select(func.count(Face.id)))
            enrollment_count = await session.execute(select(func.count(Enrollment.id)))

            print(f"\nTable Statistics:")
            print(f"  Persons:     {person_count.scalar()}")
            print(f"  Faces:       {face_count.scalar()}")
            print(f"  Enrollments: {enrollment_count.scalar()}")

    except Exception as e:
        print(f"✗ Database verification failed: {e}")
        sys.exit(1)


async def main():
    """Main initialization function"""
    print("Initializing Face Recognition Database\n")

    # Create tables
    await create_tables()

    # Seed sample data
    await seed_sample_data()

    # Verify setup
    await verify_database()

    print("\n✓ Database initialization complete!")


if __name__ == "__main__":
    asyncio.run(main())
