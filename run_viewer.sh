#!/usr/bin/env bash
# Launch the browser point cloud viewer on http://127.0.0.1:8770
set -euo pipefail
cd "$(dirname "$0")"
PY="./.venv/bin/python"
[ -x "$PY" ] || PY="python3"
exec "$PY" -u viewer_server.py "$@"
