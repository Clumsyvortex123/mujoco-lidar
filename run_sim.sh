#!/usr/bin/env bash
# Launch the demo simulator. Extra args pass straight through, e.g.
#   ./run_sim.sh --headless --rings 16 --azimuth-step 2
set -euo pipefail
cd "$(dirname "$0")"
PY="./.venv/bin/python"
[ -x "$PY" ] || PY="python3"
exec "$PY" -u sim.py "$@"
