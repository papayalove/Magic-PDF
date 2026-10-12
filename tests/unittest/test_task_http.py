"""使用真实 TCP、curl 和两个 API worker 验证上传省流量及产物一致性。"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import shutil
import socket
import subprocess
import threading
import time
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit

import httpx
import pytest
from test_http_api_example_script import _Server

from mineru.kit.router import RouterSettings
from mineru.kit.router import create_app as create_router
from mineru.parser import api_server
from mineru.parser.api_server import create_app


@pytest.fixture(params=["api", "router"])
def live_service(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[str]:
    """通过真实 Uvicorn 提供直接服务或双 worker Router。"""
    servers: list[_Server] = []
    try:
        if request.param == "api":
            service = _Server(create_app(upload_dir=str(tmp_path / "api"), tier="flash", api_key="test-key"))
        else:
            for index in range(2):
                worker = _Server(create_app(upload_dir=str(tmp_path / f"worker-{index}"), tier="flash", api_key="test-key"))
                worker.start()
                servers.append(worker)
            service = _Server(
                create_router(
                    RouterSettings(
                        upstream_urls=tuple(worker.url for worker in servers),
                        local_gpus="none",
                        worker_refresh_interval_seconds=0,
                    )
                )
            )
        service.start()
        servers.append(service)
        yield service.url
    finally:
        for server in reversed(servers):
            server.stop()


def _wait(client: httpx.Client, task_id: str) -> dict:
    """通过真实 HTTP 有界查询任务结果。"""
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        response = client.get(f"/v1/tasks/{task_id}/result")
        assert response.status_code in {200, 202}, response.text
        if response.status_code == 200:
            return response.json()
        time.sleep(0.01)
    pytest.fail("HTTP task did not complete")


@pytest.mark.skipif(shutil.which("curl") is None, reason="curl is required for real upload accounting")
def test_expect_continue_skips_bytes_and_checks_auth(live_service: str, tmp_path: Path) -> None:
    """curl 未命中时实际上传、命中时上传量为零，并经过真实鉴权。"""
    source = tmp_path / "source.html"
    data = b"<h1>Upload accounting</h1><p>" + b"content " * 32000 + b"</p>"
    source.write_bytes(data)
    query = urlencode({"filename": source.name, "sha256sum": hashlib.sha256(data).hexdigest(), "tier": "flash"})
    result_path = tmp_path / "result.json"
    command = [
        "curl",
        "--silent",
        "--show-error",
        "--http1.1",
        "--max-time",
        "20",
        "--expect100-timeout",
        "5",
        "-H",
        "Expect: 100-continue",
        "-H",
        "Content-Type: application/octet-stream",
        "-H",
        "Authorization: Bearer test-key",
        "-X",
        "POST",
        "--upload-file",
        str(source),
        "--output",
        str(result_path),
        "--write-out",
        "%{http_code} %{size_upload}",
        f"{live_service}/v1/tasks?{query}",
    ]
    with httpx.Client(
        base_url=live_service, headers={"authorization": "Bearer test-key"}, timeout=20, trust_env=False
    ) as client:
        first = subprocess.run(command, capture_output=True, text=True, check=True)
        status, uploaded = first.stdout.split()
        assert status == "202", result_path.read_text()
        assert int(uploaded) == len(data)
        task = _wait(client, json.loads(result_path.read_text())["task_id"])
        assert "Upload accounting" in task["files"][0]["content"]["markdown"]
        second = subprocess.run(command, capture_output=True, text=True, check=True)
        assert second.stdout.split() == ["202", "0"]
        assert _wait(client, json.loads(result_path.read_text())["task_id"])["status"] == "completed"
        invalid = client.post(
            "/v1/tasks",
            headers={"authorization": "Bearer wrong-key"},
            json={"tier": "flash", "files": [{"source": {"type": "file_id", "file_id": task["files"][0]["file_id"]}}]},
        )
        assert invalid.status_code in {401, 404}


def _archive_members(data: bytes) -> dict[str, bytes]:
    """比较实际成员内容，排除 ZIP 时间戳和压缩包装差异。"""
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        return {name: zf.read(name) for name in zf.namelist()}


def test_disconnect_after_submission_keeps_background_job(live_service: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """客户端在已接单后关闭真实 TCP，后台解析仍完成且只执行一次。"""
    release = threading.Event()
    original = api_server.parse_async
    calls = 0

    async def delayed(*args: Any, **kwargs: Any) -> Any:
        """保持同步请求等待，允许测试先观察已创建的后台任务。"""
        nonlocal calls
        calls += 1
        while not release.is_set():
            await asyncio.sleep(0.01)
        return await original(*args, **kwargs)

    monkeypatch.setattr(api_server, "parse_async", delayed)
    address = urlsplit(live_service)
    body = b"<h1>Disconnected client</h1>"
    connection = socket.create_connection((address.hostname, address.port), timeout=5)
    try:
        headers = (
            f"POST /v1/file_parse?filename=source.html&tier=flash HTTP/1.1\r\nHost: {address.netloc}\r\n"
            f"Authorization: Bearer test-key\r\nContent-Type: application/octet-stream\r\nContent-Length: {len(body)}\r\n\r\n"
        )
        connection.sendall(headers.encode() + body)
        with httpx.Client(
            base_url=live_service, headers={"authorization": "Bearer test-key"}, timeout=5, trust_env=False
        ) as client:
            deadline = time.monotonic() + 10
            jobs = []
            while time.monotonic() < deadline:
                jobs = client.get("/v1/parse/jobs").json()["data"]
                if jobs:
                    break
                time.sleep(0.01)
            assert len(jobs) == 1
            connection.close()
            release.set()
            result = _wait(client, jobs[0]["job_id"])
            assert result["status"] == "completed"
            assert "Disconnected client" in result["files"][0]["content"]["markdown"]
            assert calls == 1
    finally:
        release.set()
        connection.close()


@pytest.mark.parametrize("mode,format", [("sync", "json"), ("async", "zip")])
@pytest.mark.skipif(shutil.which("curl") is None, reason="curl is required for the documented shell example")
def test_documented_task_script(live_service: str, tmp_path: Path, mode: str, format: str) -> None:
    """执行文档中的真实脚本，验证 URL 编码、鉴权和两种结果形式。"""
    source = tmp_path / "测试 文档.html"
    source.write_text("<h1>Script output</h1><p>中文内容</p>", encoding="utf-8")
    output = tmp_path / f"result.{format}"
    env = dict(
        os.environ,
        MINERU_API_URL=live_service,
        MINERU_API_KEY="test-key",
        MINERU_TIER="flash",
        MODE=mode,
        RESPONSE_FORMAT=format,
        RESULT_FILE=str(output),
        OUTPUT_FORMATS="markdown",
        PAGE_RANGE="",
        OCR_MODE="txt",
        REQUEST_TIMEOUT="30",
        EXPECT_TIMEOUT="10",
        MAX_POLLS="30",
        POLL_INTERVAL="0.01",
    )
    script = Path(__file__).resolve().parents[2] / "scripts" / "http_task_example.sh"
    completed = subprocess.run(["bash", str(script), str(source)], env=env, capture_output=True, text=True, timeout=40)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    if format == "json":
        result = json.loads(output.read_text())
        assert result["files"][0]["name"] == source.name
        assert "中文内容" in result["files"][0]["content"]["markdown"]
    else:
        assert "middle_json.json" in _archive_members(output.read_bytes())


@pytest.mark.parametrize("document", ["html", "pdf"])
def test_real_documents_match_v1_output_and_zip(live_service: str, document: str) -> None:
    """真实 HTML/PDF 的便捷 JSON 和 ZIP 与既有 V1 产物逐项一致。"""
    if document == "pdf":
        path = Path(__file__).resolve().parents[2] / "examples" / "demo1.pdf"
        if not path.is_file():
            pytest.skip("Local real-PDF sample examples/demo1.pdf is unavailable")
        content = path.read_bytes()
        name, page_range = path.name, "1-2"
    else:
        content = b"<h1>HTTP equality</h1><p>Same source and output.</p>"
        name, page_range = "source.html", ""
    formats = ["markdown", "middle_json", "structured_content", "zip"]
    with httpx.Client(
        base_url=live_service, headers={"authorization": "Bearer test-key"}, timeout=60, trust_env=False
    ) as client:
        submitted = client.post(
            "/v1/tasks",
            params=[
                ("tier", "flash"),
                ("ocr_mode", "txt"),
                ("page_range", page_range),
                *(("output_formats", fmt) for fmt in formats),
            ],
            files={"files": (name, content)},
        )
        assert submitted.status_code == 202, submitted.text
        result = _wait(client, submitted.json()["task_id"])
        file_id = result["files"][0]["file_id"]
        body = {
            "tier": "flash",
            "ocr_mode": "txt",
            "output_formats": formats,
            "files": [{"source": {"type": "file_id", "file_id": file_id}, "page_range": page_range or None}],
        }
        legacy = client.post("/v1/parse/jobs", json=body)
        assert legacy.status_code == 202, legacy.text
        baseline = _wait(client, legacy.json()["job_id"])
        assert baseline["files"][0]["content"] == result["files"][0]["content"]
        first_zip = client.get(result["result_url"], params={"response_format": "zip"})
        second_zip = client.get(baseline["result_url"], params={"response_format": "zip"})
        assert first_zip.status_code == second_zip.status_code == 200
        assert _archive_members(first_zip.content) == _archive_members(second_zip.content)
        # 同步包装同样只调用一次现有解析服务，且不改变共享协议或图片素材。
        sync = client.post("/v1/file_parse", json=body)
        assert sync.status_code == 200, sync.text
        assert sync.json()["files"][0]["content"] == result["files"][0]["content"]
