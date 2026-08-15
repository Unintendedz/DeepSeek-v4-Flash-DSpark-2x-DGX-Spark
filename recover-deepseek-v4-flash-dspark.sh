#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.dspark}"

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

PROJECT_NAME="${PROJECT_NAME:-deepseek-v4-flash}"
LOCK_FILE="${XDG_RUNTIME_DIR:-/tmp}/dspark-vllm-recovery.lock"

head_container() {
  docker ps -q \
    --filter "label=com.docker.compose.project=$PROJECT_NAME" \
    --filter "label=com.docker.compose.service=vllm-dspark" \
    | head -n 1
}

health_status() {
  docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{end}}' "$1" 2>/dev/null || true
}

container="$(head_container)"
[ -n "$container" ] || exit 0
[ "$(health_status "$container")" = "unhealthy" ] || exit 0

exec 9>"$LOCK_FILE"
flock -n 9 || exit 0

# Another check may have recovered the API while this invocation waited.
container="$(head_container)"
[ -n "$container" ] || exit 0
[ "$(health_status "$container")" = "unhealthy" ] || exit 0

printf '%s DSpark API is unhealthy; restarting both nodes.\n' "$(date -Is)"
ENV_FILE="$ENV_FILE" "$SCRIPT_DIR/stop-deepseek-v4-flash-dspark.sh"
ENV_FILE="$ENV_FILE" "$SCRIPT_DIR/start-deepseek-v4-flash-dspark.sh"
