"""通过两类真实服务验证 V1 便捷接口、源字节复用和异步生命周期。"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import threading
import time
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from mineru.kit.router import RouterSettings
from mineru.kit.router import create_app as create_router
from mineru.parser import api_server, task_api

HTML = b"<!doctype html><html><body><h1>Shared tasks</h1><p>Preserved content.</p></body></html>"


class _WorkerTransport(httpx.AsyncBaseTransport):
    """通过 ASGI 驱动两个真实 API server，避免 fake 允许源文件公开下载。"""

    def __init__(self, apps: dict[str, FastAPI]) -> None:
        """保存测试 worker 及其真实路由传输。"""
        self.apps = apps
        self.transports = {host: httpx.ASGITransport(app=app) for host, app in apps.items()}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """按主机路由到真实 worker 应用。"""
        return await self.transports[request.url.host].handle_async_request(request)

    async def aclose(self) -> None:
        """关闭前等待真实后台任务清理，不遗留推理运行时租约。"""
        for app in self.apps.values():
            await app.state.job_store.shutdown()


@pytest.fixture(params=["api", "router"])
def service(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[tuple[TestClient, FastAPI]]:
    """对直接 API 和双 worker Router 运行同一份公开接口合约。"""
    if request.param == "api":
        app = api_server.create_app(upload_dir=str(tmp_path / "direct"), tier="flash")
    else:
        workers = {
            host: api_server.create_app(upload_dir=str(tmp_path / host), tier="flash") for host in ("worker-a", "worker-b")
        }
        app = create_router(
            RouterSettings(
                upstream_urls=tuple(f"http://{host}" for host in workers), local_gpus="none", worker_refresh_interval_seconds=0
            ),
            transport=_WorkerTransport(workers),
        )
    with TestClient(app) as client:
        yield client, app


def _terminal(client: TestClient, task_id: str) -> dict[str, Any]:
    """有界查询真实任务，任何失败均在测试中直接暴露。"""
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        response = client.get(f"/v1/tasks/{task_id}/result")
        assert response.status_code in {200, 202, 409}, response.text
        if response.status_code != 202:
            return response.json()
        time.sleep(0.01)
    pytest.fail("Task did not reach a terminal state")


def _raw(
    client: TestClient,
    data: bytes = HTML,
    *,
    name: str = "source.html",
    sha: str | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """提交原始字节，显式声明大小并保留调用方身份。"""
    params = {"filename": name, "tier": "flash"}
    if sha is not None:
        params["sha256sum"] = sha
    return client.post(
        "/v1/tasks", params=params, headers={"content-type": "application/octet-stream", **(headers or {})}, content=data
    )


@pytest.mark.parametrize("source_type", ["multipart", "raw", "inline"])
def test_task_sources_return_inline_results_and_v1_links(service: tuple[TestClient, FastAPI], source_type: str) -> None:
    """三类输入均通过真实解析并返回可直接消费的协议和 V1 链接。"""
    client, _ = service
    formats = ["markdown", "middle_json", "structured_content"]
    if source_type == "inline":
        import base64

        response = client.post(
            "/v1/tasks",
            json={
                "files": [{"source": {"type": "inline", "name": "source.html", "data": base64.b64encode(HTML).decode()}}],
                "tier": "flash",
                "output_formats": formats,
            },
        )
    elif source_type == "raw":
        response = client.post(
            "/v1/tasks",
            params=[("filename", "source.html"), ("tier", "flash"), *(("output_formats", fmt) for fmt in formats)],
            content=HTML,
            headers={"content-type": "application/octet-stream"},
        )
    else:
        response = client.post(
            "/v1/tasks",
            params=[("tier", "flash"), *(("output_formats", fmt) for fmt in formats)],
            files={"files": ("source.html", HTML, "text/html")},
        )
    assert response.status_code == 202, response.text
    task = response.json()
    assert "job_id" not in task
    assert task["status_url"] == f"/v1/tasks/{task['task_id']}"
    assert task["result_url"] == f"/v1/tasks/{task['task_id']}/result"
    assert task["links"]["cancel"] == task["status_url"]
    result = _terminal(client, task["task_id"])
    assert result["status"] == "completed", result
    content = result["files"][0]["content"]
    assert "Preserved content" in content["markdown"]
    assert content["middle_json"]["schema"] == "docvortex.middle"
    assert content["structured_content"]["pages"][0]["page_idx"] == 0
    assert client.post("/tasks").status_code == 404
    assert client.post("/file_parse").status_code == 404


def test_sync_defaults_and_zip_wrap_the_same_job(service: tuple[TestClient, FastAPI]) -> None:
    """省略 tier 的 HTML 自动使用 Flash，同步 ZIP 来自同一份任务产物。"""
    client, _ = service
    response = client.post("/v1/file_parse", files={"files": ("source.html", HTML)})
    assert response.status_code == 200, response.text
    task_id = response.json()["task_id"]
    job = client.get(f"/v1/parse/jobs/{task_id}")
    assert job.status_code == 200
    assert job.json()["job_id"] == task_id
    assert "Shared tasks" in response.json()["files"][0]["content"]["markdown"]
    archive = client.post("/v1/file_parse?response_format=zip&tier=flash", files={"files": ("source.html", HTML)})
    assert archive.status_code == 200, archive.text
    assert archive.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(archive.content)) as zf:
        assert "middle_json.json" in zf.namelist()
    missing = client.get(f"/v1/tasks/{task_id}/result?response_format=zip")
    assert missing.status_code == 400


def test_file_id_reuse_and_source_download_boundary(service: tuple[TestClient, FastAPI]) -> None:
    """新入口的源文件可被 V1 和便捷入口复用，仍不能公开下载。"""
    client, _ = service
    first = _terminal(client, _raw(client).json()["task_id"])
    file_id = first["files"][0]["file_id"]
    body = {"files": [{"source": {"type": "file_id", "file_id": file_id}}], "tier": "flash"}
    for route in ("/v1/file_parse", "/v1/parse/jobs"):
        response = client.post(route, json=body)
        assert response.status_code in {200, 202}, response.text
    assert client.get(f"/v1/files/{file_id}/content").status_code == 403


class _UnreadableBody(httpx.SyncByteStream):
    """缓存命中时读取主体应立即导致测试失败。"""

    def __iter__(self) -> Iterator[bytes]:
        """拒绝任何主体读取，验证提前返回并非上传后去重。"""
        raise AssertionError("Cached source body must not be read")
        yield b""  # pragma: no cover


def test_hash_hit_skips_body_and_preserves_new_filename(service: tuple[TestClient, FastAPI]) -> None:
    """命中哈希时无需发送主体，而且文件名视图不会沿用旧文件名。"""
    client, _ = service
    first = _raw(client)
    assert first.status_code == 202
    _terminal(client, first.json()["task_id"])
    response = client.post(
        "/v1/tasks",
        params={"filename": "renamed.html", "tier": "flash", "sha256sum": hashlib.sha256(HTML).hexdigest()},
        headers={"content-type": "application/octet-stream", "content-length": str(len(HTML)), "expect": "100-continue"},
        content=_UnreadableBody(),
    )
    assert response.status_code == 202, response.text
    assert response.headers["connection"] == "close"
    result = _terminal(client, response.json()["task_id"])
    assert result["files"][0]["name"] == "renamed.html"


def test_hash_size_errors_do_not_create_jobs(service: tuple[TestClient, FastAPI], monkeypatch: pytest.MonkeyPatch) -> None:
    """错误哈希、声明大小与大小上限均在任务创建前被拒绝。"""
    client, app = service
    assert _raw(client, sha="0" * 64).status_code == 400
    assert _raw(client, sha="invalid").status_code == 400
    initial = _raw(client)
    assert initial.status_code == 202
    result = _terminal(client, initial.json()["task_id"])
    wrong_size = client.post(
        "/v1/tasks",
        params={"filename": "source.html", "tier": "flash", "sha256sum": hashlib.sha256(HTML).hexdigest()},
        headers={"content-type": "application/octet-stream", "content-length": str(len(HTML) + 1)},
        content=_UnreadableBody(),
    )
    assert wrong_size.status_code == 400
    monkeypatch.setattr(task_api, "MAX_TASK_FILE_BYTES", 8)
    assert _raw(client).status_code == 413
    count = len(app.state.job_store._jobs) if hasattr(app.state, "job_store") else len(app.state.registry.list("job"))
    assert count == 1
    assert result["status"] == "completed"


def test_partial_and_duplicate_names_have_distinct_zip_directories(service: tuple[TestClient, FastAPI]) -> None:
    """批量同名与部分失败均保留输入顺序，ZIP manifest 不隐藏错误。"""
    client, _ = service
    response = client.post(
        "/v1/file_parse?response_format=zip&tier=flash&ocr_mode=txt",
        files=[
            ("files", ("same.html", HTML)),
            ("files", ("same.html", b"<p>Second file</p>")),
            ("files", ("broken.pdf", b"not a pdf")),
        ],
    )
    assert response.status_code == 200, response.text
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["status"] == "partial"
        assert manifest["files"][2]["error"]
        assert any(name.startswith("0001/") for name in zf.namelist())
        assert any(name.startswith("0002/") for name in zf.namelist())
        assert not any(name.startswith("0003/") for name in zf.namelist())


def test_sync_timeout_leaves_one_background_task(service: tuple[TestClient, FastAPI], monkeypatch: pytest.MonkeyPatch) -> None:
    """等待超时返回同一任务，该任务随后继续完成且不会重复解析。"""
    client, _ = service
    release = threading.Event()
    original = api_server.parse_async
    calls = 0

    async def delayed(*args: Any, **kwargs: Any) -> Any:
        """通过可控闸门模拟超过等待预算的真实解析。"""
        nonlocal calls
        calls += 1
        while not release.is_set():
            await asyncio.sleep(0.01)
        return await original(*args, **kwargs)

    monkeypatch.setattr(api_server, "parse_async", delayed)
    try:
        response = client.post("/v1/file_parse?wait_timeout=1&tier=flash", files={"files": ("source.html", HTML)})
        assert response.status_code == 202, response.text
        task_id = response.json()["task_id"]
        assert response.json()["status_url"] == f"/v1/tasks/{task_id}"
    finally:
        release.set()
    assert _terminal(client, task_id)["status"] == "completed"
    assert calls == 1


def test_query_errors_and_openapi_match(service: tuple[TestClient, FastAPI]) -> None:
    """校验输入预算和公开文档，包括三种内容类型及完整 V1 前缀。"""
    client, _ = service
    for query in ("wait_timeout=0", "wait_timeout=3601", "response_format=unsupported"):
        response = client.post(f"/v1/file_parse?{query}", json={"files": []})
        assert response.status_code == 400, response.text
    spec = client.get("/openapi.json").json()
    for path in ("/v1/tasks", "/v1/tasks/{task_id}", "/v1/tasks/{task_id}/result", "/v1/file_parse"):
        assert path in spec["paths"]
    schema = spec["paths"]["/v1/tasks"]["post"]["requestBody"]["content"]
    assert set(schema) == {"application/json", "multipart/form-data", "application/octet-stream"}
    assert "#/$defs/" not in json.dumps(schema)
    assert "application/zip" in spec["paths"]["/v1/file_parse"]["post"]["responses"]["200"]["content"]
    task_api.TaskStatusResponse.model_validate(_raw(client).json())


def test_invalid_ranges_empty_and_too_many_uploads(service: tuple[TestClient, FastAPI]) -> None:
    """输入数量和页范围在两端均返回明确的 V1 错误，不进入解析。"""
    client, _ = service
    for response in (
        client.post("/v1/tasks?page_range=invalid", files={"files": ("source.html", HTML)}),
        client.post("/v1/tasks", files={"files": ("source.html", b"")}),
        client.post("/v1/tasks", files=[("files", (f"file-{index}.html", b"x")) for index in range(101)]),
        client.post("/v1/tasks", files={"unknown": ("source.html", HTML)}),
    ):
        assert response.status_code == 400, response.text
        assert response.json()["error"]["code"]


def test_cancellation_and_failed_results_are_terminal(
    service: tuple[TestClient, FastAPI], monkeypatch: pytest.MonkeyPatch
) -> None:
    """取消使用共享 Job，失败结果不被误报成 200 或重新提交。"""
    client, _ = service
    release = threading.Event()
    original = api_server.parse_async

    async def delayed(*args: Any, **kwargs: Any) -> Any:
        """保持解析运行，以便通过公开取消接口验证后台任务归属。"""
        while not release.is_set():
            await asyncio.sleep(0.01)
        return await original(*args, **kwargs)

    monkeypatch.setattr(api_server, "parse_async", delayed)
    try:
        task = _raw(client).json()
        canceled = client.delete(task["status_url"])
        assert canceled.status_code == 200, canceled.text
        assert canceled.json()["status"] == "canceled"
        result = client.get(task["result_url"])
        assert result.status_code == 409
        assert result.json()["status"] == "canceled"
        assert client.delete(task["status_url"]).status_code == 409
    finally:
        release.set()
        monkeypatch.setattr(api_server, "parse_async", original)
    failed = client.post("/v1/file_parse?tier=flash&ocr_mode=txt", files={"files": ("broken.pdf", b"broken")})
    assert failed.status_code == 409, failed.text
    assert failed.json()["status"] == "failed"
    assert failed.json()["files"][0]["error"]


def test_router_cached_uploads_keep_scope_and_alias_references(tmp_path: Path) -> None:
    """Router 秒传隔离身份；删除旧别名后新别名仍支持跨 worker 复制。"""
    workers = {host: api_server.create_app(upload_dir=str(tmp_path / host), tier="flash") for host in ("worker-a", "worker-b")}
    app = create_router(
        RouterSettings(
            upstream_urls=tuple(f"http://{host}" for host in workers), local_gpus="none", worker_refresh_interval_seconds=0
        ),
        transport=_WorkerTransport(workers),
    )
    headers = {"authorization": "Bearer tenant-a"}
    with TestClient(app) as client:
        task = _raw(client, headers=headers).json()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            result = client.get(task["result_url"], headers=headers)
            if result.status_code == 200:
                break
            time.sleep(0.01)
        first_id = result.json()["files"][0]["file_id"]
        body = {
            "filename": "renamed.html",
            "bytes": len(HTML),
            "mime_type": "text/html",
            "sha256sum": hashlib.sha256(HTML).hexdigest(),
        }
        reused = client.post("/v1/uploads", headers=headers, json=body)
        assert reused.status_code == 200, reused.text
        assert reused.json()["status"] == "completed"
        second_id = reused.json()["file"]["id"]
        assert client.post("/v1/uploads", headers={"authorization": "Bearer tenant-b"}, json=body).json()["status"] == "pending"
        assert client.get(task["status_url"], headers={"authorization": "Bearer tenant-b"}).status_code == 404
        assert client.delete(f"/v1/files/{first_id}", headers=headers).status_code == 200
        stored = app.state.source_store.find_file(second_id)
        assert stored is not None and stored.path.is_file()
        parsed = client.post(
            "/v1/file_parse",
            headers=headers,
            json={"tier": "flash", "files": [{"source": {"type": "file_id", "file_id": second_id}}]},
        )
        assert parsed.status_code == 200, parsed.text
        assert "Preserved content" in parsed.json()["files"][0]["content"]["markdown"]


def test_api_source_registration_cancel_rolls_back_view(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """磁盘线程完成时 HTTP 已取消，预先分配的源文件视图仍必须回滚。"""
    app = api_server.create_app(upload_dir=str(tmp_path / "api"), tier="flash")
    source = tmp_path / "source.html"
    source.write_bytes(HTML)
    entered, release = threading.Event(), threading.Event()
    original = api_server.FileStore.register_source_file

    def delayed(store: api_server.FileStore, filename: str, path: Path, sha256sum: str, file_id: str | None = None) -> str:
        """在线程注册完成后阻塞返回，复现取消时丢失返回 ID 的窗口。"""
        result = original(store, filename, path, sha256sum, file_id)
        entered.set()
        release.wait(5)
        return result

    monkeypatch.setattr(api_server.FileStore, "register_source_file", delayed)

    async def run() -> None:
        """取消提交方并等待线程清理，检查文件与任务索引。"""
        backend = task_api.ApiTaskBackend(Request({"type": "http", "app": app, "headers": []}))
        body = api_server.CreateJobRequest(
            files=[api_server.JobFileEntry(source=api_server.FileIdSource(file_id="pending"))], tier="flash"
        )
        work = asyncio.create_task(
            backend.submit(body, {0: task_api.TaskUpload(source, source.name, len(HTML), hashlib.sha256(HTML).hexdigest())})
        )
        while not entered.is_set():
            await asyncio.sleep(0.01)
        work.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await work
        assert not app.state.file_store._files
        assert not app.state.job_store._jobs
        await app.state.job_store.shutdown()

    asyncio.run(run())
