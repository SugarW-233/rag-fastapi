from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class DocumentRecord(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        String(32),
        primary_key=True,
    )

    file_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    stored_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    file_hash: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        index=True,
        nullable=False,
    )
    file_size: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    page_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    chunk_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    status: Mapped[str] = mapped_column(
        String(20),
        default="processing",
        index=True,
        nullable=False,
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=func.now(),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
