#!/usr/bin/env bash
# All mode lifecycle symlinks execute this file. Keep the ORIGINAL invocation path.
set -euo pipefail
MODE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$MODE_DIR/../.." && pwd)"
ACTION="$(basename -- "${BASH_SOURCE[0]}" .sh)"
case "$ACTION" in
  launch) ACTION=_run ;;
  start|status|stop|force-stop) ;;
  *) echo "Invoke this through a registered mode lifecycle symlink" >&2; exit 2 ;;
esac
exec "$PROJECT_DIR/mgr" "$ACTION" "$(basename -- "$MODE_DIR")" "$@"
