import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, Index, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(AsyncAttrs, DeclarativeBase):
    pass


class Person(Base):
    __tablename__ = "persons"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    metadata: Mapped[Optional[dict]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    faces: Mapped[list["Face"]] = relationship(back_populates="person", cascade="all, delete-orphan")

    __table_args__ = (Index("idx_person_created_at", "created_at"),)


class Face(Base):
    __tablename__ = "faces"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    person_id: Mapped[str] = mapped_column(String(100), nullable=False)
    embedding_id: Mapped[int] = mapped_column(nullable=False)  # FAISS index ID
    quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    image_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    image_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    face_bbox: Mapped[Optional[dict]] = mapped_column(Text, nullable=True)
    landmarks: Mapped[Optional[dict]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    person: Mapped["Person"] = relationship(back_populates="faces")

    __table_args__ = (
        Index("idx_face_person_id", "person_id"),
        Index("idx_face_embedding_id", "embedding_id"),
        Index("idx_face_created_at", "created_at"),
    )


class Enrollment(Base):
    __tablename__ = "enrollments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    person_id: Mapped[str] = mapped_column(String(100), nullable=False)
    face_count: Mapped[int] = mapped_column(default=0)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("idx_enrollment_person_id", "person_id"),
        Index("idx_enrollment_status", "status"),
        Index("idx_enrollment_created_at", "created_at"),
    )