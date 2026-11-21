"""SQLite 公共连接与事务辅助：保证连接必然关闭、统一事务提交/回滚与 WAL 初始化。"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator

DEFAULT_SQLITE_TIMEOUT_SEC = 5.0


def open_sqlite_connection(
    db_path: str,
    *,
    timeout: float = DEFAULT_SQLITE_TIMEOUT_SEC,
    foreign_keys: bool = False,
    row_factory: Any = sqlite3.Row,
) -> sqlite3.Connection:
    """打开一条配置好超时、row_factory 与可选外键约束的 SQLite 连接。

    注意：``PRAGMA journal_mode=WAL`` 是数据库文件级持久化属性，应仅在
    ``init_db()`` / ``init_schema()`` 中通过 :func:`enable_wal_mode` 执行一次，
    避免每次短连接重复探测写锁。
    """
    conn = sqlite3.connect(db_path, timeout=timeout, check_same_thread=False)
    try:
        if row_factory is not None:
            conn.row_factory = row_factory
        if foreign_keys:
            conn.execute("PRAGMA foreign_keys=ON")
        return conn
    except Exception:
        conn.close()
        raise


def enable_wal_mode(conn: sqlite3.Connection) -> None:
    """在初始化建表期一次性开启 WAL 日志模式。"""
    conn.execute("PRAGMA journal_mode=WAL")


@contextmanager
def sqlite_connection(
    db_path: str,
    *,
    timeout: float = DEFAULT_SQLITE_TIMEOUT_SEC,
    foreign_keys: bool = False,
    row_factory: Any = sqlite3.Row,
    transactional: bool = False,
) -> Iterator[sqlite3.Connection]:
    """托管 SQLite 连接生命周期：可选事务提交/回滚，并在退出时必然 ``close()``。"""
    conn = open_sqlite_connection(
        db_path,
        timeout=timeout,
        foreign_keys=foreign_keys,
        row_factory=row_factory,
    )
    try:
        if transactional:
            with conn:
                yield conn
        else:
            yield conn
    finally:
        conn.close()
