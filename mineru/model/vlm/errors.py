# Copyright (c) Opendatalab. All rights reserved.
"""仅识别推理引擎明确报告的不可恢复错误，不依赖模糊日志关键词。"""

from __future__ import annotations


class EngineDeadError(RuntimeError):
    """表示本地引擎已死亡，当前进程不能继续接受推理。"""


def is_fatal_engine_error(error: BaseException) -> bool:
    """遍历异常链和异常组，优先匹配 vLLM 的明确死亡类型。"""
    pending = [error]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        if isinstance(current, EngineDeadError):
            return True
        for cls in type(current).__mro__:
            if cls.__module__.startswith("vllm.") and cls.__name__ in {"EngineDeadError", "AsyncEngineDeadError"}:
                return True
        if any(cls.__name__ == "BaseExceptionGroup" for cls in type(current).__mro__):
            pending.extend(item for item in getattr(current, "exceptions", ()) if isinstance(item, BaseException))
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        elif current.__context__ is not None and not current.__suppress_context__:
            pending.append(current.__context__)
    return False


__all__ = ["EngineDeadError", "is_fatal_engine_error"]
