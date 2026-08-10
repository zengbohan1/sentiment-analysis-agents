"""SQLAlchemy 异步数据层：PostgreSQL 主库 + MySQL 从库（可选）+ SQLite 测试兜底。"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from ..config import Settings

# 统一命名约束，便于多库迁移
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=NAMING_CONVENTION)


class Database:
    """双数据源封装：主库（PostgreSQL / SQLite）与可选 MySQL 从库。

    - 主库：任务、报告、评测结果等核心数据。
    - MySQL（可选）：舆情数据采集落库（MindSpider 对接），配置为空时复用主库。
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.main_engine: AsyncEngine | None = None
        self.mysql_engine: AsyncEngine | None = None
        self.main_session: async_sessionmaker[AsyncSession] | None = None
        self.mysql_session: async_sessionmaker[AsyncSession] | None = None

    async def connect(self) -> None:
        self.main_engine = create_async_engine(
            self._settings.database_url, pool_pre_ping=True, pool_size=10, max_overflow=20
        )
        self.main_session = async_sessionmaker(self.main_engine, expire_on_commit=False)
        if self._settings.database_url_mysql:
            self.mysql_engine = create_async_engine(
                self._settings.database_url_mysql, pool_pre_ping=True
            )
            self.mysql_session = async_sessionmaker(self.mysql_engine, expire_on_commit=False)

    async def create_all(self) -> None:
        """建表（测试/开发环境使用；生产建议 Alembic 迁移）。"""
        if self.main_engine is None:
            await self.connect()
        assert self.main_engine is not None
        # 延迟导入避免循环引用
        from .. import models  # noqa: F401

        async with self.main_engine.begin() as conn:
            await conn.run_sync(metadata.create_all)

    async def disconnect(self) -> None:
        for engine in (self.main_engine, self.mysql_engine):
            if engine is not None:
                await engine.dispose()

    @asynccontextmanager
    async def session(self, use_mysql: bool = False) -> AsyncIterator[AsyncSession]:
        maker = self.mysql_session if use_mysql and self.mysql_session else self.main_session
        if maker is None:
            raise RuntimeError("数据库未连接，请先调用 connect()")
        async with maker() as sess:
            yield sess


_db: Database | None = None


def get_db(settings: Settings | None = None) -> Database:
    global _db
    if _db is None:
        _db = Database(settings or Settings())
    return _db
