#!/usr/bin/env bash
# V1 便捷解析：同一个原始字节请求携带哈希，命中时可跳过主体传输。
# MODE=sync|async，RESPONSE_FORMAT=json|zip，OUTPUT_FORMATS=markdown,middle_json,...
# 成功输出到 RESULT_FILE；部分成功退出 2，失败退出 1，预算耗尽退出 124。
set -euo pipefail

[ $# -eq 1 ] && [ -f "$1" ] || { echo 'usage: bash scripts/http_task_example.sh <file>' >&2; exit 1; }
: "${MINERU_API_URL:?Set MINERU_API_URL to the API or Router base URL}"
TASK_SOURCE=$1
TASK_MODE=${MODE:-sync}
TASK_FORMAT=${RESPONSE_FORMAT:-json}
TASK_WAIT=${WAIT_TIMEOUT:-300}
TASK_MAX_POLLS=${MAX_POLLS:-300}
TASK_RESULT=${RESULT_FILE:-result.$TASK_FORMAT}
case "$TASK_MODE" in sync|async) ;; *) echo 'MODE must be sync or async' >&2; exit 1 ;; esac
case "$TASK_FORMAT" in json|zip) ;; *) echo 'RESPONSE_FORMAT must be json or zip' >&2; exit 1 ;; esac
TASK_RESPONSE=$(mktemp)
TASK_HEADERS=$(mktemp)
trap 'rm -f "$TASK_RESPONSE" "$TASK_HEADERS"' EXIT
TASK_AUTH=()
if [ -n "${MINERU_API_KEY:-}" ]; then TASK_AUTH=(-H "Authorization: Bearer $MINERU_API_KEY"); fi

# 用标准库编码查询串，文件名和页范围不进入 shell 求值。
TASK_URL=$(python3 - "$TASK_SOURCE" "$TASK_MODE" "$TASK_FORMAT" "$TASK_WAIT" <<'PY'
import hashlib
import os
import pathlib
import sys
from urllib.parse import urlencode

path = pathlib.Path(sys.argv[1])
hasher = hashlib.sha256()
with path.open('rb') as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b''):
        hasher.update(chunk)
query = [('filename', path.name), ('sha256sum', hasher.hexdigest())]
if os.environ.get('MINERU_TIER'):
    query.append(('tier', os.environ['MINERU_TIER']))
query.extend([('ocr_mode', os.environ.get('OCR_MODE', 'auto')), ('page_range', os.environ.get('PAGE_RANGE', ''))])
formats = [value.strip() for value in os.environ.get('OUTPUT_FORMATS', 'markdown').split(',')]
if sys.argv[3] == 'zip' and 'zip' not in formats:
    formats.append('zip')
query.extend(('output_formats', value) for value in formats)
route = '/v1/file_parse' if sys.argv[2] == 'sync' else '/v1/tasks'
if sys.argv[2] == 'sync':
    query.extend([('response_format', sys.argv[3]), ('wait_timeout', sys.argv[4])])
print(os.environ['MINERU_API_URL'].rstrip('/') + route + '?' + urlencode(query))
PY
)

TASK_CODE=$(curl --silent --show-error --http1.1 --connect-timeout 10 --max-time "${REQUEST_TIMEOUT:-600}" \
  --expect100-timeout "${EXPECT_TIMEOUT:-310}" -H 'Expect: 100-continue' -H 'Content-Type: application/octet-stream' \
  ${TASK_AUTH[@]+"${TASK_AUTH[@]}"} -X POST --upload-file "$TASK_SOURCE" \
  --output "$TASK_RESPONSE" --dump-header "$TASK_HEADERS" --write-out '%{http_code}' "$TASK_URL")

TASK_POLLS=0
while [ "$TASK_CODE" = 202 ]; do
  TASK_ID=$(python3 - "$TASK_RESPONSE" <<'PY'
import json
import sys
from urllib.parse import quote

with open(sys.argv[1], encoding='utf-8') as source:
    task_id = json.load(source)['task_id']
print(quote(task_id, safe=''))
PY
)
  if [ "$TASK_POLLS" -ge "$TASK_MAX_POLLS" ]; then
    echo "Polling budget exhausted; task_id=$TASK_ID continues running" >&2
    exit 124
  fi
  TASK_POLLS=$((TASK_POLLS + 1))
  sleep "${POLL_INTERVAL:-1}"
  TASK_CODE=$(curl --silent --show-error --connect-timeout 10 --max-time "${REQUEST_TIMEOUT:-600}" \
    ${TASK_AUTH[@]+"${TASK_AUTH[@]}"} --output "$TASK_RESPONSE" --dump-header "$TASK_HEADERS" \
    --write-out '%{http_code}' "${MINERU_API_URL%/}/v1/tasks/$TASK_ID/result?response_format=$TASK_FORMAT")
done
if [ "$TASK_CODE" != 200 ]; then cat "$TASK_RESPONSE" >&2; exit 1; fi
mv -- "$TASK_RESPONSE" "$TASK_RESULT"
echo "Saved $TASK_RESULT"
python3 - "$TASK_RESULT" "$TASK_FORMAT" <<'PY'
import json
import sys
import zipfile

if sys.argv[2] == 'json':
    with open(sys.argv[1], encoding='utf-8') as source:
        status = json.load(source)['status']
else:
    with zipfile.ZipFile(sys.argv[1]) as archive:
        status = json.loads(archive.read('manifest.json'))['status'] if 'manifest.json' in archive.namelist() else 'completed'
raise SystemExit(2 if status == 'partial' else 0)
PY
