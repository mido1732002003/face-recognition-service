# path: scripts/delete_all_persons.py
import asyncio
from sqlalchemy import text
from core.database import get_db_context

async def reset_db():
    async with get_db_context() as session:
        await session.execute(text("TRUNCATE TABLE faces CASCADE;"))
        await session.execute(text("TRUNCATE TABLE enrollments CASCADE;"))
        await session.execute(text("TRUNCATE TABLE persons CASCADE;"))
        await session.commit()
        print("[✓] All persons, faces, and enrollments deleted.")

if __name__ == "__main__":
    asyncio.run(reset_db())
