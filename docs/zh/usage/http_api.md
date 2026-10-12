# V1 HTTP API

自部署 V1 API 与 Python SDK、WebUI 使用同一套接口。本页说明一次完整请求周期——创建上传、上传字节、提交解析任务、轮询到终态、下载产物——并指向一个经过测试的示例脚本。Python 客户端见 [Python SDK](sdk_api.md)。

先启动本地服务，或将示例指向任意 V1 部署（包括官方云服务）：

```bash
mineru-kit api-server --host 127.0.0.1 --port 8000 --tier standard
```

## 一次请求解析与异步 tasks

API server 和 Router 都提供 `/v1/tasks`、`/v1/file_parse`，与现有 Job 共用后台解析和任务 ID：

```bash
curl -sS 'http://127.0.0.1:8000/v1/file_parse?tier=flash&ocr_mode=txt' \
  -F 'files=@document.pdf' -o result.json

curl -sS 'http://127.0.0.1:8000/v1/file_parse?tier=flash&response_format=zip' \
  -F 'files=@document.pdf' -o result.zip

MINERU_API_URL=http://127.0.0.1:8002 MODE=async RESPONSE_FORMAT=zip \
  bash scripts/http_task_example.sh document.pdf
```

同步等待默认 300 秒，`wait_timeout` 支持 1–3600 秒；超时返回 `202` 和 `task_id/status_url/result_url`，任务继续执行。
异步提交使用 `POST /v1/tasks`，通过返回的 V1 地址查询状态和结果。JSON 内联 `files[].content`，同时保留逐文件错误和产物引用。
异步 ZIP 必须提交时请求 `output_formats=zip`；同步 ZIP 自动请求。部分成功返回 `200` 和 `partial`，批量 ZIP 含分目录产物及 manifest。

上传可以是 multipart 或原始字节，也可使用与 V1 Job 相同的 JSON 来源。上传解析参数放查询串，JSON 参数放主体。
每文件最多 200 MiB，每任务最多 100 文件；批量上传共用页范围，逐文件配置使用 JSON。原始字节需提供 `filename` 和 `Content-Length`。
上述脚本自动计算 SHA-256 并使用 `Expect: 100-continue`；命中缓存且及时响应时，可跳过主体传输。
代理缓冲和客户端等待预算会影响省流量效果；直接复用 `file_id` 或现有 uploads 预检仍可可靠避免重复上传。

Router 的 V1 上传和便捷上传共享按调用方隔离的源字节缓存。上传按负载分配、同负载轮询；任务在上游网络等待前预占名额。
文件归属仅在最低负载候选中享有优先权，哈希命中也不能把任务固定到繁忙 worker。
一个批量任务仍整体交给一台 API，需要多 worker 并行时提交多个独立任务。
外置 worker 当前按容量 1 计权；本地 worker 容量来自 `--worker-concurrency`。这些路由属于当前自部署服务，云端支持情况以具体部署为准。

## 经过测试的示例脚本

包含超时、响应校验、有界轮询、退出码和凭据隔离的完整闭环，以仓库中单一经过测试的脚本维护：

> [`scripts/http_api_example.sh`](https://github.com/opendatalab/MinerU/blob/master/scripts/http_api_example.sh)

```bash
export MINERU_API_URL=http://127.0.0.1:8000
export MINERU_API_KEY=secret-key        # 匿名本地访问时省略
./scripts/http_api_example.sh document.pdf
```

脚本行为由 `tests/unittest/test_http_api_example_script.py` 在真实 Flash 服务和脚本化 mock 服务上验证，无需下载模型。退出码面向自动化保持稳定：

| 码 | 含义 |
| --- | --- |
| `0` | 任务 completed，全部请求产物已保存 |
| `1` | 脚本、传输、协议、下载或本地写入失败 |
| `2` | 任务 partial：已先保存成功文件的产物 |
| `3` | 任务 failed |
| `4` | 任务 canceled |
| `124` | 轮询预算耗尽；打印 `job_id`，可继续轮询 |

## 鉴权与字节上传

服务以 `--api-key` 启动时，`/v1/*` 请求携带 `Authorization: Bearer $MINERU_API_KEY`；匿名本地访问可省略该头。

字节上传原样使用 `POST /v1/uploads` 返回的方法、URL 和请求头：

- 自部署服务返回同源 URL（如 `/v1/uploads/{id}/content`）。同源指 scheme + host + 有效端口，由客户端判断而非假设；同源上传在服务要求鉴权时必须携带 API 鉴权。
- 官方 API 返回自带授权的预签名对象存储 URL。**不要**向不同源的 URL 附加 MinerU API Key——那会泄露凭据。服务返回的上传头（`upload_headers`）始终原样保留。

示例脚本实现了与 Python SDK（`api_client._same_origin_upload_headers`）相同的规则：相对 URL 先按 API 基地址解析，再比较 scheme + host + 有效端口，仅对同源上传附加 MinerU Key。相对 URL 解析后即同源；同一主机的不同端口按不同源处理。

下载走 `GET /v1/files/{file_id}/content`，可能返回 `302` 重定向。示例跟随重定向但不使用 `--location-trusted`，凭据不会被重新发送到不同源（包括同主机不同端口；要求 curl >= 7.83.0）。

## 脚本刻意覆盖的请求周期细节

- **上传生命周期。** 创建上传时传 `sha256sum` 可启用完整性校验和秒传；命中已存在文件时响应直接为 `completed` 并内嵌 `file` 对象，跳过字节上传和完成两步。
- **HTTP 200 不等于合法响应。** 脚本在每一步校验 JSON 结构：ID 必须是非空字符串、状态必须属于支持集合；否则立即失败，而不是带着 null ID 进入轮询。
- **有界轮询。** `MAX_POLLS` 限制轮询请求次数（每次还包括请求超时和轮询间隔）；预算耗尽时脚本退出 `124` 并打印 `job_id`——任务未被取消，可直接继续轮询。
- **终态。** `completed`、`partial`、`failed`、`canceled` 都是终态。`partial` 表示部分文件成功：脚本先保存成功文件的产物、打印含错误的逐文件结果，然后才退出 `2`；`failed` 退出 `3`，`canceled` 退出 `4`。
- **原子下载。** 产物先写 `*.part`，成功后才移到最终文件名；`completed` 任务若有产物下载失败或缺失，不会退出 `0`。
- **产物。** 所有产物都是 `purpose:"parse_output"` 的文件；内容一律通过 `GET /v1/files/{file_id}/content`、用 `files[].output_files` 下的 id 获取。以 `GET /v1/health`（`features.sources`、`features.output_formats`）为当前部署实际能力的依据；渲染器支持的格式不代表 API 暴露。

## 生产部署注意事项

- 谨慎绑定 `--host`。默认 `127.0.0.1` 只监听回环；使用 `0.0.0.0` 暴露到网络时，应配合 `--api-key` 或带鉴权的反向代理。
- 存储与重启边界：文件字节保存在 `--upload-dir` 下（未配置时为临时目录，正常关闭即删除），但上传/文件/任务的资源索引是进程内状态。即使保留了目录，服务重启后旧的 `upload_id`/`file_id`/任务 ID 也不保证可用。该 API 不是可恢复的持久化任务服务；持久化阅读、索引和缓存需求使用文档库——但文档库不是这套上传/任务接口的透明持久化替代品。
- `--concurrency` 限制并发解析任务数；超出后表现为 `queued` 时间变长，而不是失败。
- `GET /v1/usage` 报告用量。用 `/v1/health` 和 `/docs`（启用时）的 OpenAPI 文档区分服务端故障与客户端错误；重试 `failed` 任务前先查看服务端日志。

便捷解析使用 `/v1/file_parse`、`/v1/tasks`；根路径旧接口不提供，旧客户端迁移见[迁移指南](../reference/migration_4.md)。
