"""两类 V1 服务共用的便捷任务输入、结果和同步等待接口。"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import tempfile
import zipfile
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NoReturn, Protocol

from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..errors import MineruError
from ..utils.async_utils import run_sync
from .api_server import (
    AccessLevel,
    CreateJobRequest,
    FileIdSource,
    FileStore,
    JobAsyncResponse,
    JobFileEntry,
    JobFileResult,
    JobStore,
    _raise_api_error,
    submit_parse_job,
)
from .page_range import normalize_page_range_input

MAX_TASK_FILE_BYTES = 200 * 1024 * 1024
MAX_TASK_FILES = 100
TERMINAL_TASK_STATUSES = frozenset({"completed", "partial", "failed", "canceled"})
ResponseFormat = Literal["json", "zip"]


@dataclass(frozen=True)
class TaskUpload:
    """表示已校验的输入字节，文件名视图独立于共享缓存路径。"""

    path: Path
    filename: str
    bytes: int
    sha256sum: str
    mime_type: str = "application/octet-stream"


class TaskBackend(Protocol):
    """便捷接口只依赖显式任务和文件操作，不维护独立队列。"""

    def release_uploads(self) -> None:
        """释放请求预检取得的输入租约，包括校验失败和取消路径。"""
        ...

    async def find_upload(self, sha256sum: str, filename: str) -> TaskUpload | None:
        """查找当前调用方可复用的源字节。"""
        ...

    async def submit(self, body: CreateJobRequest, uploads: dict[int, TaskUpload]) -> dict[str, Any]:
        """接收源字节并通过既有任务服务提交。"""
        ...

    async def get(self, task_id: str) -> dict[str, Any]:
        """查询既有任务，返回标准 Job 数据。"""
        ...

    async def cancel(self, task_id: str) -> dict[str, Any]:
        """取消既有任务而不影响其他任务。"""
        ...

    async def wait(self, task_id: str, timeout: float) -> dict[str, Any]:
        """有界等待，等待方超时不能取消后台任务。"""
        ...

    async def read_output(self, file_id: str) -> bytes:
        """通过当前服务的授权边界读取产物。"""
        ...


class TaskStatusResponse(JobAsyncResponse):
    """沿用 Job 字段，序列化为 task_id 并增加 V1 查询地址。"""

    job_id: str = Field(alias="task_id")
    status_url: str
    result_url: str


class TaskFileContent(JobFileResult):
    """内联文本产物，同时保留既有文件状态、错误和资源引用。"""

    content: dict[str, Any] = Field(default_factory=dict)


class TaskResultResponse(TaskStatusResponse):
    """描述可直接消费的 JSON 解析结果。"""

    files: list[TaskFileContent]


class UploadTaskOptions(BaseModel):
    """上传解析参数仅来自查询串，保证哈希命中时无需读取表单。"""

    model_config = ConfigDict(extra="forbid")
    tier: Literal["flash", "basic", "standard", "advanced"] | None = None
    ocr_mode: Literal["auto", "txt", "ocr"] = "auto"
    page_range: str = ""
    output_formats: list[str] = Field(default_factory=lambda: ["markdown"])
    filename: str | None = Field(default=None, min_length=1)
    sha256sum: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


def _invalid(message: str, *, code: str = "invalid_request", status_code: int = 400, param: str | None = None) -> NoReturn:
    """将输入和产物错误转换为两端一致的 V1 错误结构。"""
    _raise_api_error(
        status_code,
        error_type="api_error" if status_code >= 500 else "invalid_request_error",
        code=code,
        message=message,
        param=param,
    )


def task_payload(job: dict[str, Any]) -> dict[str, Any]:
    """重写任务名称与链接，保留上游语义树和公共资源标识。"""
    payload = copy.deepcopy(job)
    task_id = str(payload.pop("job_id"))
    base = f"/v1/tasks/{task_id}"
    payload.update(task_id=task_id, status_url=base, result_url=f"{base}/result", links={"self": base, "cancel": base})
    return payload


async def _save_upload(chunks: AsyncIterator[bytes], path: Path, filename: str, mime_type: str) -> TaskUpload:
    """逐块写入并计算实际哈希，超过大小限制立即终止。"""
    hasher = hashlib.sha256()
    size = 0
    with path.open("wb") as target:
        async for chunk in chunks:
            size += len(chunk)
            if size > MAX_TASK_FILE_BYTES:
                _invalid("File exceeds 200 MiB", code="file_too_large", status_code=413)
            target.write(chunk)
            hasher.update(chunk)
    if not size:
        _invalid("Empty files are not supported", param="files")
    return TaskUpload(path, filename, size, hasher.hexdigest(), mime_type)


async def _file_chunks(file: UploadFile) -> AsyncIterator[bytes]:
    """限制单次读取大小，不把 multipart 文件整体放入内存。"""
    while chunk := await file.read(1024 * 1024):
        yield chunk


async def _prepare_submission(
    request: Request,
    backend: TaskBackend,
    directory: Path,
    *,
    sync: bool,
    force_zip: bool,
) -> tuple[CreateJobRequest, dict[int, TaskUpload], bool]:
    """统一处理 JSON、表单和原始字节，命中时完全不读取请求主体。"""
    media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    params = dict(request.query_params)
    if sync:
        params.pop("response_format", None)
        params.pop("wait_timeout", None)
    uploads: dict[int, TaskUpload] = {}
    skipped = False
    try:
        if media_type == "application/json":
            if params:
                _invalid("JSON parse options must be supplied in the request body")
            body = CreateJobRequest.model_validate(await request.json())
        else:
            if "output_formats" in params:
                params["output_formats"] = request.query_params.getlist("output_formats")
            options = UploadTaskOptions.model_validate(params)
            options.page_range = normalize_page_range_input(options.page_range)
            if not options.output_formats or any(
                fmt not in {"markdown", "middle_json", "structured_content", "zip"} for fmt in options.output_formats
            ):
                _invalid("Unsupported output format", code="unsupported_output_format", param="output_formats")
            if media_type == "application/octet-stream":
                if options.filename is None:
                    _invalid("filename is required for raw uploads", param="filename")
                size_header = request.headers.get("content-length")
                if size_header is None:
                    _invalid("Content-Length is required for raw uploads", status_code=411)
                declared_size = int(size_header)
                if declared_size <= 0 or declared_size > MAX_TASK_FILE_BYTES:
                    _invalid("Invalid raw file size", code="file_too_large", status_code=413)
                cached = await backend.find_upload(options.sha256sum, options.filename) if options.sha256sum else None
                if cached is not None:
                    if cached.bytes != declared_size:
                        _invalid("Declared size does not match cached bytes", code="upload_size_mismatch")
                    uploads[0] = cached
                    skipped = True
                else:
                    upload = await _save_upload(request.stream(), directory / "0", options.filename, media_type)
                    if upload.bytes != declared_size:
                        _invalid("Declared size does not match uploaded bytes", code="upload_size_mismatch")
                    if options.sha256sum and upload.sha256sum != options.sha256sum:
                        _invalid("SHA-256 mismatch", code="file_hash_mismatch")
                    uploads[0] = upload
            elif media_type == "multipart/form-data":
                if options.filename is not None or options.sha256sum is not None:
                    _invalid("filename and sha256sum are only supported for raw uploads")
                async with request.form(max_files=MAX_TASK_FILES, max_fields=0) as form:
                    files = form.getlist("files")
                    if not files or any(key != "files" for key in form):
                        _invalid("Supply one or more files fields", param="files")
                    for index, file in enumerate(files):
                        if not isinstance(file, UploadFile) or not file.filename:
                            _invalid("Every files field must contain a named file", param="files")
                        if file.size is not None and file.size > MAX_TASK_FILE_BYTES:
                            _invalid("File exceeds 200 MiB", code="file_too_large", status_code=413)
                        uploads[index] = await _save_upload(
                            _file_chunks(file),
                            directory / str(index),
                            file.filename,
                            file.content_type or "application/octet-stream",
                        )
            else:
                _invalid("Use application/json, multipart/form-data or application/octet-stream", status_code=415)
            body = CreateJobRequest(
                files=[
                    JobFileEntry(source=FileIdSource(file_id=f"upload-{index}"), page_range=options.page_range or None)
                    for index in uploads
                ],
                tier=options.tier,
                ocr_mode=options.ocr_mode,
                output_formats=options.output_formats,
            )
        for entry in body.files:
            entry.page_range = normalize_page_range_input(entry.page_range) or None
        if body.callback is not None:
            _invalid("Webhook callbacks are not supported by this service", param="callback")
        if any(fmt not in {"markdown", "middle_json", "structured_content", "zip"} for fmt in body.output_formats):
            _invalid("Unsupported output format", code="unsupported_output_format", param="output_formats")
        if force_zip and "zip" not in body.output_formats:
            body.output_formats.append("zip")
        return body, uploads, skipped
    except StarletteHTTPException as exc:
        _invalid(str(exc.detail), status_code=exc.status_code)
        raise AssertionError("unreachable") from exc
    except MineruError as exc:
        _invalid(str(exc), code=exc.code, param=exc.param)
        raise AssertionError("unreachable") from exc
    except (ValidationError, ValueError) as exc:
        _invalid(str(exc))
        raise AssertionError("unreachable") from exc


async def build_task_result(job: dict[str, Any], backend: TaskBackend, response_format: ResponseFormat) -> Response:
    """只读取既有产物并共享结果封装，禁止为了下载而重新解析。"""
    payload = task_payload(job)
    status = str(job["status"])
    if status not in TERMINAL_TASK_STATUSES:
        return JSONResponse(payload, status_code=202)
    if status in {"failed", "canceled"}:
        return JSONResponse(payload, status_code=409)
    if response_format == "zip":
        if "zip" not in job.get("output_formats", []):
            _invalid("Request zip in output_formats when submitting the task", code="unsupported_output_format")
        archives: list[tuple[int, bytes]] = []
        for index, file in enumerate(job.get("files", [])):
            if file.get("status") == "completed":
                ref = (file.get("output_files") or {}).get("zip")
                if not ref:
                    _invalid("ZIP output is missing", status_code=502, code="invalid_response")
                archives.append((index, await backend.read_output(ref["file_id"])))
        if len(job["files"]) == 1:
            data = archives[0][1]
        else:
            data = await run_sync(_combine_archives, archives, payload)
        return Response(
            data,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{payload["task_id"]}.zip"',
                "X-MinerU-Task-Id": payload["task_id"],
                "X-MinerU-Task-Status": status,
            },
        )
    for file in payload["files"]:
        file["content"] = {}
        if file.get("status") != "completed":
            continue
        for fmt in job.get("output_formats", []):
            if fmt in {"markdown", "middle_json", "structured_content"} and not (file.get("output_files") or {}).get(fmt):
                _invalid(f"{fmt} output is missing", status_code=502, code="invalid_response")
        for fmt, ref in (file.get("output_files") or {}).items():
            if ref and fmt in {"markdown", "middle_json", "structured_content"}:
                data = await backend.read_output(ref["file_id"])
                file["content"][fmt] = data.decode("utf-8") if fmt == "markdown" else json.loads(data)
    return JSONResponse(payload)


def _combine_archives(archives: list[tuple[int, bytes]], payload: dict[str, Any]) -> bytes:
    """按输入序号封装批量产物，拒绝源 ZIP 中的越界路径。"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as target:
        target.writestr("manifest.json", json.dumps(payload, ensure_ascii=False))
        for index, data in archives:
            with zipfile.ZipFile(io.BytesIO(data)) as source:
                for info in source.infolist():
                    path = Path(info.filename)
                    if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
                        _invalid("Invalid ZIP member path", status_code=502, code="invalid_response")
                    target.writestr(f"{index + 1:04d}/{info.filename}", source.read(info))
    return buffer.getvalue()


def _request_schema() -> dict[str, Any]:
    """展开模型引用，为手动读取主体的两端生成相同 OpenAPI 输入定义。"""
    schema = CreateJobRequest.model_json_schema()
    definitions = schema.pop("$defs", {})

    def expand(value: Any) -> Any:
        """展开当前请求模型的非递归定义，避免引用未注册的组件。"""
        if isinstance(value, dict):
            if "$ref" in value:
                return expand(definitions[value["$ref"].rsplit("/", 1)[-1]])
            if "discriminator" in value:
                value = dict(value)
                value["discriminator"] = {"propertyName": value["discriminator"]["propertyName"]}
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        return value

    return {
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {"schema": expand(schema)},
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["files"],
                        "properties": {
                            "files": {
                                "type": "array",
                                "maxItems": MAX_TASK_FILES,
                                "items": {"type": "string", "format": "binary"},
                            }
                        },
                    }
                },
                "application/octet-stream": {"schema": {"type": "string", "format": "binary"}},
            },
        },
        "parameters": [
            {"name": name, "in": "query", "required": False, "schema": value}
            for name, value in UploadTaskOptions.model_json_schema()["properties"].items()
        ]
        + [
            {
                "name": "Content-Length",
                "in": "header",
                "required": False,
                "description": "Required for application/octet-stream; must match actual file bytes.",
                "schema": {"type": "integer", "minimum": 1, "maximum": MAX_TASK_FILE_BYTES},
            }
        ],
    }


def build_task_router(factory: Callable[[Request], TaskBackend]) -> APIRouter:
    """注册相同 V1 路由，后台实现由显式工厂选择。"""
    router = APIRouter(prefix="/v1", tags=["Tasks"])

    @router.post("/tasks", status_code=202, response_model=TaskStatusResponse, openapi_extra=_request_schema())
    async def create_task(request: Request) -> Response:
        """提交任务后立即返回公共 ID，缓存命中时不读取请求主体。"""
        backend = factory(request)
        try:
            with tempfile.TemporaryDirectory(prefix="mineru-task-") as directory:
                body, uploads, skipped = await _prepare_submission(
                    request, backend, Path(directory), sync=False, force_zip=False
                )
                job = await backend.submit(body, uploads)
        finally:
            backend.release_uploads()
        return JSONResponse(
            task_payload(job),
            status_code=202,
            headers={"Connection": "close"} if skipped and request.scope.get("http_version") == "1.1" else None,
        )

    @router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
    async def get_task(task_id: str, request: Request) -> Response:
        """查询与 V1 Job 共用的任务记录。"""
        return JSONResponse(task_payload(await factory(request).get(task_id)))

    @router.delete("/tasks/{task_id}")
    async def cancel_task(task_id: str, request: Request) -> Response:
        """通过后台已有取消机制取消任务。"""
        return JSONResponse(task_payload(await factory(request).cancel(task_id)))

    @router.get(
        "/tasks/{task_id}/result",
        response_model=None,
        responses={
            200: {
                "model": TaskResultResponse,
                "content": {"application/zip": {"schema": {"type": "string", "format": "binary"}}},
            },
            202: {"model": TaskStatusResponse},
        },
    )
    async def get_result(task_id: str, request: Request, response_format: ResponseFormat = "json") -> Response:
        """根据任务终态返回内联 JSON 或既有 ZIP 产物。"""
        backend = factory(request)
        return await build_task_result(await backend.get(task_id), backend, response_format)

    @router.post(
        "/file_parse",
        response_model=None,
        openapi_extra=_request_schema(),
        responses={
            200: {
                "model": TaskResultResponse,
                "content": {"application/zip": {"schema": {"type": "string", "format": "binary"}}},
            },
            202: {"model": TaskStatusResponse},
        },
    )
    async def file_parse(
        request: Request, response_format: ResponseFormat = "json", wait_timeout: int = Query(default=300, ge=1, le=3600)
    ) -> Response:
        """同步包装共享异步任务，等待超时不取消后台工作。"""
        backend = factory(request)
        try:
            with tempfile.TemporaryDirectory(prefix="mineru-task-") as directory:
                body, uploads, skipped = await _prepare_submission(
                    request, backend, Path(directory), sync=True, force_zip=response_format == "zip"
                )
                job = await backend.submit(body, uploads)
        finally:
            backend.release_uploads()
        job = await backend.wait(job["job_id"], wait_timeout)
        response = await build_task_result(job, backend, response_format)
        if skipped and request.scope.get("http_version") == "1.1":
            response.headers["Connection"] = "close"
        return response

    return router


class ApiTaskBackend:
    """把便捷接口适配到 API server 唯一的文件存储和 JobStore。"""

    def __init__(self, request: Request) -> None:
        """保留已通过鉴权的请求及当前应用资源。"""
        self.request = request
        self.files: FileStore = request.app.state.file_store
        self.jobs: JobStore = request.app.state.job_store
        self._cached_pins: list[str] = []
        self.access: AccessLevel = "registered" if request.app.state.api_key else "anonymous"

    def release_uploads(self) -> None:
        """释放当前请求的哈希预检引用，失败和取消同样经过此处。"""
        for sha in self._cached_pins:
            self.files.unpin_cached_blob(sha)
        self._cached_pins.clear()

    async def find_upload(self, sha256sum: str, filename: str) -> TaskUpload | None:
        """原子查找并持有可复用字节，直到提交或输入校验完成。"""
        path = self.files.pin_cached_blob(sha256sum)
        if path is None:
            return None
        self._cached_pins.append(sha256sum)
        return TaskUpload(path, filename, path.stat().st_size, sha256sum)

    async def submit(self, body: CreateJobRequest, uploads: dict[int, TaskUpload]) -> dict[str, Any]:
        """注册源字节后调用共享任务提交函数，失败时回滚文件视图。"""
        registered: list[str] = []
        try:
            for index, upload in uploads.items():
                file_id = self.files._new_file_id()
                registered.append(file_id)
                await run_sync(self.files.register_source_file, upload.filename, upload.path, upload.sha256sum, file_id)
                body.files[index].source = FileIdSource(file_id=file_id)
            rec = await submit_parse_job(body, self.request, self.files, self.jobs, self.access)
            return self.jobs.build_response(rec, access_level=self.access).model_dump(by_alias=True)
        except BaseException:
            for file_id in registered:
                if file_id in self.files._files:
                    self.files.delete_file(file_id)
            raise

    async def get(self, task_id: str) -> dict[str, Any]:
        """返回现有 Job 的规范数据。"""
        return self.jobs.build_response(self.jobs.get(task_id), access_level=self.access).model_dump(by_alias=True)

    async def cancel(self, task_id: str) -> dict[str, Any]:
        """取消同一份 Job 并返回当前完整状态。"""
        self.jobs.cancel(task_id)
        return await self.get(task_id)

    async def wait(self, task_id: str, timeout: float) -> dict[str, Any]:
        """等待清理完成事件，不把 HTTP 请求取消传播到解析。"""
        await self.jobs.wait_for_terminal(task_id, timeout)
        return await self.get(task_id)

    async def read_output(self, file_id: str) -> bytes:
        """沿用文件存储的产物下载权限检查。"""
        return await run_sync(self.files.read_file_data, file_id)


__all__ = [
    "ApiTaskBackend",
    "TaskBackend",
    "TaskResultResponse",
    "TaskStatusResponse",
    "TaskUpload",
    "build_task_router",
    "task_payload",
]
