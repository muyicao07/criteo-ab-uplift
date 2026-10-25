#!/usr/bin/env bash
set -uo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "$PROJECT_DIR/run.sh" "$@"
PROJECT_STATUS=$?
if [[ -t 0 ]]; then
  printf '\nPress Return to close this window.\n'
  read -r || true
fi
exit "$PROJECT_STATUS"
