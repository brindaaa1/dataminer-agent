#!/usr/bin/env bash
# 一键启动工作台：装依赖、构建前端（第一次或前端有改动时）、启动服务并打开浏览器。
# 需要 Python 3.10+ 和 Node.js；用真实模型时在 .env 里填 key（格式见 .env.example），不填也能用 mock 走完整流程。
set -e
cd "$(dirname "$0")"
pip install -q -r requirements.txt
if [ ! -d web/dist ] || [ -n "$(find web/src web/index.html -newer web/dist -print -quit)" ]; then
  (cd web && npm install --silent && npm run build)
fi
URL=http://127.0.0.1:8000
(sleep 3 && (open "$URL" || xdg-open "$URL")) >/dev/null 2>&1 &
exec uvicorn server.app:app --port 8000
