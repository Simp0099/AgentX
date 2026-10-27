"""Shared timeout/result handling for forensic tools."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from agent.schemas import ToolResult


def run_bounded(
    tool: str, fn: Callable, timeout: float | None = None, executor: Callable | None = None
) -> ToolResult:
    started = time.monotonic()
    try:
        if timeout is None:
            from agent.config import get_config

            timeout = float(get_config()["forensics"]["tool_timeout_seconds"])
        pool = (executor or ThreadPoolExecutor)(max_workers=1)
        try:
            future = pool.submit(fn)
            try:
                value = future.result(timeout=timeout)
            except TimeoutError:
                future.cancel()
                pool.shutdown(wait=False, cancel_futures=True)
                return ToolResult(
                    tool,
                    "timeout",
                    error="tool timed out",
                    duration_ms=(time.monotonic() - started) * 1000,
                )
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        return ToolResult(
            tool,
            "ok",
            value if isinstance(value, dict) else {"value": value},
            duration_ms=(time.monotonic() - started) * 1000,
        )
    except Exception as exc:  # tool boundary must not leak exceptions
        return ToolResult(
            tool, "error", error=str(exc), duration_ms=(time.monotonic() - started) * 1000
        )
