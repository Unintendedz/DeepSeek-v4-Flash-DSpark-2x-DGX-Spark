#!/usr/bin/env bash
# Apply the runtime text patch twice in a disposable stock image, then compile
# every modified vLLM module. No GPU or Mooncake service is required.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${DSPARK_VLLM_IMAGE:-ghcr.io/anemll/dspark-vllm-gx10:0.1.1}"
PATCH="$ROOT/patches/hotfix-dsv4-mooncake-eagle-lookup.py"

docker run --rm --entrypoint bash \
  -v "$PATCH:/opt/hotfix-dsv4-mooncake-eagle-lookup.py:ro" \
  "$IMAGE" -lc '
    set -e
    python3 /opt/hotfix-dsv4-mooncake-eagle-lookup.py
    python3 /opt/hotfix-dsv4-mooncake-eagle-lookup.py
    python3 -m py_compile \
      /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py \
      /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py \
      /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/utils.py
  '
