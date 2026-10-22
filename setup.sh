#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-openvino.txt
.venv/bin/python -m lazyrig.probe
printf '\nSetup complete. Run ./start-server.sh and open http://127.0.0.1:8765\n'
