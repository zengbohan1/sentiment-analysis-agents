"""Redis 缓存：热点查询 / 分析结果缓存；宕机或未连接时自动降级为进程内缓存。"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from ..config import Settings


class Cache:
    """统一缓存接口，Redis 后端 + 内存兜底。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._redis = None
        self._memory: dict[str, tuple[float, Any]] = {}
        self._memory_lock = asyncio.Lock()
        self._enabled = True

    async def connect(self) -> None:
        try:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(
                self._settings.redis_url, decode_responses=True, socket_connect_timeout=1
            )
            await self._redis.ping()
        except Exception:
            # 宕机/未启动：静默降级到内存缓存
            self._redis = None

    async def get(self, key: str) -> Any | None:
        if self._redis is not None:
            try:
                raw = await self._redis.get(key)
                return json.loads(raw) if raw is not None else None
            except Exception:
                self._redis = None  # 连接中断 → 降级
        async with self._memory_lock:
            item = self._memory.get(key)
            if item is None:
                return None
            expire_at, value = item
            if expire_at < time.monotonic():
                self._memory.pop(key, None)
                return None
            return value

    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        if self._redis is not None:
            try:
                await self._redis.set(key, json.dumps(value, ensure_ascii=False), ex=ttl)
                return
            except Exception:
                self._redis = None
        async with self._memory_lock:
            self._memory[key] = (time.monotonic() + ttl, value)

    async def delete(self, key: str) -> None:
        if self._redis is not None:
            try:
                await self._redis.delete(key)
                return
            except Exception:
                self._redis = None
        async with self._memory_lock:
            self._memory.pop(key, None)

    async def incr(self, key: str, amount: int = 1, ttl: int = 3600) -> int:
        """计数器（任务数统计等）。"""
        val = await self.get(key)
        new = (val if isinstance(val, (int, float)) else 0) + amount
        await self.set(key, new, ttl)
        return int(new)

    async def close(self) -> None:
        if self._redis is not None:
            try:
                await self._redis.aclose()
            except Exception:
                pass
            self._redis = None
