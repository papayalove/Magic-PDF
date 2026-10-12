"""把共享便捷任务接口适配到 Router 的公共资源和上游 V1 服务。"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from contextlib import ExitStack
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ...parser.api_server import ApiServerError, CreateJobRequest, ErrorDetail
from ...parser.task_api import TERMINAL_TASK_STATUSES, TaskUpload
from .proxy import RouterProxyError, request_upstream
from .resources import ResourceRegistry, SourceFileStore
from .workers import WorkerPool

SubmitJob = Callable[[dict[str, Any], Request, dict[int, TaskUpload]], Awaitable[Response]]
ReadJob = Callable[[str, Request], Awaitable[Response]]


def _response_data(response: Response) -> dict[str, Any]:
    """读取内部服务 JSON，错误保留原 HTTP 状态和 V1 error 字段。"""
    try:
        data = json.loads(bytes(response.body))
        if not isinstance(data, dict):
            raise ValueError("Response must be an object")
    except ValueError as exc:
        status = response.status_code if response.status_code >= 400 else 502
        raise RouterProxyError(status, "invalid_upstream_response", "Upstream response is not a JSON object") from exc
    if response.status_code >= 400:
        raw = data.get("error") or {}
        raise ApiServerError(
            response.status_code,
            ErrorDetail(
                type=raw.get("type", "api_error"),
                code=raw.get("code"),
                message=raw.get("message", "Upstream error"),
                param=raw.get("param"),
            ),
        )
    return data


class RouterTaskBackend:
    """复用 Router 的既有服务函数，所有产物访问均检查调用方资源 scope。"""

    def __init__(self, request: Request, submit: SubmitJob, get: ReadJob, cancel: ReadJob, owner_scope: str) -> None:
        """保存显式回调和当前 Router 资源，不建立独立任务队列。"""
        self.request = request
        self.owner_scope = owner_scope
        self.submit_job = submit
        self.get_job = get
        self.cancel_job = cancel
        self.sources: SourceFileStore = request.app.state.source_store
        self.registry: ResourceRegistry = request.app.state.registry
        self.pool: WorkerPool = request.app.state.worker_pool
        self._input_pins = ExitStack()

    def release_uploads(self) -> None:
        """关闭预检阶段持有的输入引用，传输或校验失败也会释放。"""
        self._input_pins.close()

    async def find_upload(self, sha256sum: str, filename: str) -> TaskUpload | None:
        """只复用本调用方在 Router 中仍持有的源字节。"""
        stored = self.sources.find_hash(self.owner_scope, sha256sum)
        if stored is None:
            return None
        self._input_pins.enter_context(self.sources.pin(stored))
        return TaskUpload(stored.path, filename, stored.bytes, stored.sha256sum, stored.mime_type)

    async def submit(self, body: CreateJobRequest, uploads: dict[int, TaskUpload]) -> dict[str, Any]:
        """委托共享提交服务，源字节传输也纳入相同的负载预占。"""
        return _response_data(await self.submit_job(body.model_dump(exclude_none=True), self.request, uploads))

    async def get(self, task_id: str) -> dict[str, Any]:
        """使用既有查询服务更新终态、输出映射和副本回收。"""
        return _response_data(await self.get_job(task_id, self.request))

    async def cancel(self, task_id: str) -> dict[str, Any]:
        """使用既有取消入口，返回完整公共任务状态。"""
        _response_data(await self.cancel_job(task_id, self.request))
        return await self.get(task_id)

    async def wait(self, task_id: str, timeout: float) -> dict[str, Any]:
        """每秒查询所属 worker，预算包含查询时间且不取消远端任务。"""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                route = self.registry.find("job", task_id)
                if route is None or route.owner_scope != self.owner_scope:
                    raise RouterProxyError(404, "job_not_found", f"Job {task_id} not found")
                return dict(route.metadata["payload"])
            try:
                payload = await asyncio.wait_for(self.get(task_id), timeout=remaining)
            except asyncio.TimeoutError:
                route = self.registry.find("job", task_id)
                if route is None or route.owner_scope != self.owner_scope:
                    raise RouterProxyError(404, "job_not_found", f"Job {task_id} not found")
                return dict(route.metadata["payload"])
            if payload["status"] in TERMINAL_TASK_STATUSES:
                return payload
            await asyncio.sleep(min(1, max(0, deadline - time.monotonic())))

    async def read_output(self, file_id: str) -> bytes:
        """解析公共 File 路由并沿用上游产物下载边界。"""
        route = self.registry.find("file", file_id)
        if route is None or route.owner_scope != self.owner_scope:
            raise RouterProxyError(404, "file_not_found", f"File {file_id} not found")
        response = await request_upstream(
            self.pool,
            self.pool.get(route.worker_id),
            "GET",
            f"/v1/files/{route.upstream_id}/content",
            request=self.request,
            follow_redirects=True,
        )
        if response.status_code >= 400:
            _response_data(Response(response.content, status_code=response.status_code))
        return response.content


def install_task_error_handlers(app: FastAPI) -> None:
    """让共享任务输入错误与直接 API server 使用相同的 V1 envelope。"""

    @app.exception_handler(ApiServerError)
    async def task_error(_request: Request, exc: ApiServerError) -> JSONResponse:
        """保留共享错误的 HTTP 状态和明确错误码。"""
        return JSONResponse({"error": exc.error.model_dump()}, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        """将路由查询参数校验错误统一转换为 400。"""
        return JSONResponse(
            {"error": {"type": "invalid_request_error", "code": "invalid_request", "message": str(exc)}}, status_code=400
        )


__all__ = ["RouterTaskBackend", "install_task_error_handlers"]
