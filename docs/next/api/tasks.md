# V1 便捷解析接口

状态: Implemented
读者: 自部署 API 与 Router 使用者、服务端开发者
范围: 一次提交、异步结果、同步包装和源文件复用

`mineru-kit api-server` 和 `mineru-kit router` 提供相同的 `/v1/tasks`、`/v1/file_parse` 接口。
它们与 `/v1/parse/jobs` 共用任务：`task_id` 就是当前部署的 `job_id`，可以在两套查询接口之间交叉使用。
官方云服务是否支持这些便捷路由，应以该部署的接口文档为准。

## 路由

| 方法与路径 | 行为 |
|---|---|
| `POST /v1/tasks` | 提交后返回 `202`、任务状态及查询地址。 |
| `GET /v1/tasks/{task_id}` | 查询状态、进度和逐文件结果引用。 |
| `GET /v1/tasks/{task_id}/result` | 直接返回 JSON 结果，或通过 `response_format=zip` 获取 ZIP。 |
| `DELETE /v1/tasks/{task_id}` | 取消原有后台任务。 |
| `POST /v1/file_parse` | 提交相同的异步任务，等待并直接返回结果。 |

服务需要鉴权时，所有请求都携带 `Authorization: Bearer <key>`。上传解析参数只来自查询串；JSON 请求则通过主体传递解析参数。
不提供根路径 `/tasks`、`/file_parse`，也不接受旧 `backend`、`parse_method` 或旧输出协议。

## 一次请求获取结果

```bash
curl -sS 'http://127.0.0.1:8000/v1/file_parse?tier=flash&ocr_mode=txt' \
  -F 'files=@document.pdf' -o result.json

curl -sS 'http://127.0.0.1:8000/v1/file_parse?tier=flash&response_format=zip' \
  -F 'files=@document.pdf' -o result.zip
```

多个重复的 `files` 字段表示一个批量任务，整批交给一个 worker。上传方式的查询参数如下：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `tier` | 服务默认值 | `flash/basic/standard/advanced`，仍受服务能力约束。 |
| `ocr_mode` | `auto` | `auto/txt/ocr`。 |
| `page_range` | 整本 | PDF 页范围，批量文件共用；非 PDF 保持整本解析约束。 |
| `output_formats` | `markdown` | 重复参数选择 `markdown/middle_json/structured_content/zip`。 |
| `response_format` | `json` | 同步及结果接口支持 `json/zip`。 |
| `wait_timeout` | `300` | 同步等待秒数，范围 `1–3600`，包含排队及解析。 |

每文件最多 200 MiB，每任务最多 100 文件。需要逐文件页范围或不同来源时，使用与 [Parse Jobs](parse-jobs.md) 相同的 JSON：

```json
{
  "files": [{"source": {"type": "file_id", "file_id": "file-..."}, "page_range": "1-2"}],
  "tier": "flash",
  "ocr_mode": "txt",
  "output_formats": ["markdown", "middle_json", "zip"]
}
```

JSON 结果保留 `files[].status/error/output_files`，并在 `files[].content` 内联请求的文本和 JSON 产物。
Middle JSON 保留 `schema: "docvortex.middle"` 和版本身份；Structured Content 保留当前消费结构，不转换成旧 Content List。

`completed/partial` 返回 `200`；`failed/canceled` 返回 `409`。部分成功保留成功文件和失败项，不把整个任务误报为成功。
批量 ZIP 使用 `0001/`、`0002/` 等目录，并附带逐文件状态的 `manifest.json`。

## 异步任务与同步超时

```bash
curl -sS 'http://127.0.0.1:8000/v1/tasks?tier=flash&output_formats=markdown&output_formats=zip' \
  -F 'files=@document.pdf'

# 使用响应中提供的实际 task_id / result_url。
curl -sS 'http://127.0.0.1:8000/v1/tasks/job_.../result'
curl -sS 'http://127.0.0.1:8000/v1/tasks/job_.../result?response_format=zip' -o result.zip
```

未到终态的结果查询返回 `202`。异步 ZIP 下载要求提交时请求 `zip`，否则返回 `400 unsupported_output_format`；同步 ZIP 会自动请求该产物。

同步等待预算耗尽时返回 `202`，包含 `task_id/status_url/result_url`。任务继续运行；HTTP 断连也不取消已经提交的任务。
客户端应使用有界查询继续获取结果，不要重新提交同一请求，否则会创建另一份任务。

## 哈希与上传复用

新文件可直接 multipart 上传。该方式能复用存储，但文件字节仍经过网络。
重复解析可以直接提交已有 `file_id`，或使用 [Uploads](uploads-files.md) 的 SHA-256 预检。

原始字节入口允许在主体之前提交哈希；下面的客户端示例自动计算 SHA-256、编码文件名并支持同步/异步调用：

```bash
MINERU_API_URL=http://127.0.0.1:8000 \
  bash scripts/http_task_example.sh document.pdf

MINERU_API_URL=http://127.0.0.1:8002 MODE=async RESPONSE_FORMAT=zip \
  bash scripts/http_task_example.sh document.pdf
```

该方式使用 `Content-Type: application/octet-stream`、查询参数 `filename/sha256sum` 和必须的 `Content-Length`。
服务命中真实缓存字节时不读取主体；未命中时接收字节并核对实际大小和哈希。
客户端配合 `Expect: 100-continue`，且服务响应在客户端等待预算内到达时，可以完全跳过文件传输。
同步解析可能超过客户端的 Expect 等待预算；反向代理也可能自行确认并缓存上传，因此不能对所有部署保证零传输。

Router 仅复用当前调用方仍持有的私有源字节；仅 worker 已知该哈希不构成 Router 命中。
Router 的 V1 上传与便捷上传共享缓存；哈希命中后仍按任务负载选择 worker，必要时在 worker 间传输缓存字节。
源文件保持禁止公开下载，`file_id`、任务和上传索引仍是进程内状态，服务重启不提供恢复承诺。

## Router 调度

上传不再按凭证和 IP 固定分配。Router 先过滤健康和能力，再比较未完成任务及提交预占的负载；同负载轮询。
任务按 `(active_jobs / max_concurrent_jobs, active_jobs)` 比较，源文件归属仅在最低负载候选中享有优先权。
上传与任务的轮询游标独立，能力和状态查询不推进游标。

计数在复制或创建上游任务之前预占；失败、取消、终态和后台回收通过同一机制释放一次。
一个批量任务仍交给一个 worker。需要多个 worker 同时处理时，提交多个独立任务。
托管 worker 的容量来自 `--worker-concurrency`；外置 worker 当前按容量 1 计权，只统计通过本 Router 提交的任务。
现有命令默认获得新策略，不增加调度开关、动态服务发现或多副本共享资源索引。
