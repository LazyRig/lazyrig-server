#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  printf 'Run ./setup.sh first.\n'
  exit 1
fi
printf 'Starting LazyRig Server...\n'
exec .venv/bin/python -u -m lazyrig --mode openvino "$@"
