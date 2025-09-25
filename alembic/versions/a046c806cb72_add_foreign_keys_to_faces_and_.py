"""add foreign keys to faces and enrollments"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "a046c806cb72"
down_revision = "001"  # ← هنا لازم يكون رقم آخر revision عندك (راجع اسم ملف أول migration)
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add foreign key to faces.person_id → persons.id
    op.create_foreign_key(
        "fk_faces_person_id",
        "faces",
        "persons",
        ["person_id"],
        ["id"],
        ondelete="CASCADE",
    )

    # Add foreign key to enrollments.person_id → persons.id
    op.create_foreign_key(
        "fk_enrollments_person_id",
        "enrollments",
        "persons",
        ["person_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    # Drop foreign keys
    op.drop_constraint("fk_faces_person_id", "faces", type_="foreignkey")
    op.drop_constraint("fk_enrollments_person_id", "enrollments", type_="foreignkey")
