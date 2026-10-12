# Copyright (c) Opendatalab. All rights reserved.
"""API 与 Router 共用的终态资源保留配置。"""

from __future__ import annotations

import os

DEFAULT_RETENTION_SECONDS = 86400
RETENTION_SCAN_INTERVAL_SECONDS = 300
RETENTION_ENV = "MINERU_API_RETENTION_SECONDS"


def resolve_retention_seconds(value: int | None = None) -> int:
    """显式参数优先于环境变量；零禁用回收，负数和非法值立即报错。"""
    if value is None:
        try:
            value = int(os.environ.get(RETENTION_ENV, str(DEFAULT_RETENTION_SECONDS)))
        except ValueError as exc:
            raise ValueError(f"{RETENTION_ENV} must be a non-negative integer") from exc
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("retention_seconds must be a non-negative integer")
    return value


__all__ = ["DEFAULT_RETENTION_SECONDS", "RETENTION_SCAN_INTERVAL_SECONDS", "RETENTION_ENV", "resolve_retention_seconds"]
