from __future__ import annotations

from datetime import datetime
from uuid import uuid4
from sqlalchemy import DateTime, Integer, String, ForeignKey, Numeric, BigInteger, func
from sqlalchemy.orm import Mapped, mapped_column, relationship, DeclarativeBase


class Base(DeclarativeBase):
    pass


class Instrument(Base):
    __tablename__ = "instruments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(
        String(32), unique=True, nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=True)
    exchange: Mapped[str] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    prices: Mapped[list[Price]] = relationship(
        "Price", back_populates="instrument", cascade="all, delete-orphan"
    )


class Price(Base):
    __tablename__ = "prices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instrument_id: Mapped[int] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    open: Mapped[Numeric] = mapped_column(Numeric(20, 6), nullable=False)
    high: Mapped[Numeric] = mapped_column(Numeric(20, 6), nullable=False)
    low: Mapped[Numeric] = mapped_column(Numeric(20, 6), nullable=False)
    close: Mapped[Numeric] = mapped_column(Numeric(20, 6), nullable=False)
    volume: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    instrument: Mapped[Instrument] = relationship("Instrument", back_populates="prices")


class Watchlist(Base):
    __tablename__ = "watchlist"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instrument_id: Mapped[int] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    owner: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    added_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    instrument: Mapped[Instrument] = relationship("Instrument")


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: uuid4().hex)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    sessions: Mapped[list[UserSession]] = relationship(
        "UserSession", back_populates="user", cascade="all, delete-orphan"
    )


class UserSession(Base):
    __tablename__ = "user_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    user: Mapped[User] = relationship("User", back_populates="sessions")
