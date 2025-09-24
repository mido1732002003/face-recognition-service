"""Initial schema

Revision ID: 001
Revises: 
Create Date: 2025-01-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create persons table
    op.create_table(
        'persons',
        sa.Column('id', sa.String(length=100), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=True),
        sa.Column('metadata', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_person_created_at', 'persons', ['created_at'], unique=False)

    # Create faces table
    op.create_table(
        'faces',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('person_id', sa.String(length=100), nullable=False),
        sa.Column('embedding_id', sa.Integer(), nullable=False),
        sa.Column('quality_score', sa.Float(), nullable=True),
        sa.Column('image_path', sa.String(length=500), nullable=True),
        sa.Column('image_hash', sa.String(length=64), nullable=True),
        sa.Column('face_bbox', sa.Text(), nullable=True),
        sa.Column('landmarks', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['person_id'], ['persons.id'], ondelete='CASCADE')
    )
    op.create_index('idx_face_person_id', 'faces', ['person_id'], unique=False)
    op.create_index('idx_face_embedding_id', 'faces', ['embedding_id'], unique=False)
    op.create_index('idx_face_created_at', 'faces', ['created_at'], unique=False)

    # Create enrollments table
    op.create_table(
        'enrollments',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('person_id', sa.String(length=100), nullable=False),
        sa.Column('face_count', sa.Integer(), nullable=False, default=0),
        sa.Column('status', sa.String(length=50), nullable=False, default='pending'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_enrollment_person_id', 'enrollments', ['person_id'], unique=False)
    op.create_index('idx_enrollment_status', 'enrollments', ['status'], unique=False)
    op.create_index('idx_enrollment_created_at', 'enrollments', ['created_at'], unique=False)


def downgrade() -> None:
    op.drop_index('idx_enrollment_created_at', table_name='enrollments')
    op.drop_index('idx_enrollment_status', table_name='enrollments')
    op.drop_index('idx_enrollment_person_id', table_name='enrollments')
    op.drop_table('enrollments')
    
    op.drop_index('idx_face_created_at', table_name='faces')
    op.drop_index('idx_face_embedding_id', table_name='faces')
    op.drop_index('idx_face_person_id', table_name='faces')
    op.drop_table('faces')
    
    op.drop_index('idx_person_created_at', table_name='persons')
    op.drop_table('persons')