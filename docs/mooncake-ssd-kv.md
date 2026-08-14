# Experimental Mooncake SSD-backed KV tier

This optional alpha profile adds a local-NVMe external KV tier to the pinned
Anemll `0.1.1` two-node DSpark runtime. It is for long conversations whose GPU
KV no longer remains in memory between turns: Mooncake stores reusable blocks
on SSD and vLLM reloads a verified prefix before recomputing only the tail.

It does **not** make model generation faster, compress the conversation, or
change model weights. Cold SSD recovery still has visible TTFT; the benefit is
avoiding a full prefill when the same long prefix returns.

## Exact compatibility pins

- image: `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`
- vLLM in that image: `0.25.2.dev0+g752a3a504.d20260714`
- Mooncake: `0.3.12.post1`, CUDA 13, CPython 3.12, aarch64
- cache geometry: TP=2, `nvfp4_ds_mla`, block size 256

Stage these two wheels on **both** nodes under the same host directory. Wheels
are deliberately not committed to this repository.

| File | SHA-256 |
|---|---|
| `mooncake_transfer_engine_cuda13-0.3.12.post1-cp312-cp312-manylinux_2_28_aarch64.whl` | `31664c1aedfcb5938c984ce132f95c42ef523709b7e104b23010c8cd2fec9cbb` |
| `msgpack-1.2.1-cp312-cp312-manylinux2014_aarch64.manylinux_2_17_aarch64.manylinux_2_28_aarch64.whl` | `60926b75d00c8e816ef98f3034f484a8bc64242d66839cef4cf7e503142316a0` |

Verify before launch:

```bash
sha256sum ~/dspark-mooncake-wheels/*.whl
```

## Configure both nodes

Create the SSD directory and copy the two wheels to both nodes. Copy
[`mooncake-standalone.example.json`](mooncake-standalone.example.json) to
`~/dspark-local/mooncake-standalone.json`, then replace `<HEAD_ROCE_IP>` with
the head node's fabric IP. The paths must resolve identically on both nodes.

Add this to `.env.dspark` on the head:

```env
ENABLE_KV_SSD=1
GPU_MEMORY_UTILIZATION_SSD=0.75
DSPARK_MOONCAKE_WHEELS=${HOME}/dspark-mooncake-wheels
DSPARK_MOONCAKE_CONFIG=${HOME}/dspark-local/mooncake-standalone.json
DSPARK_KV_SSD_DIR=${HOME}/.cache/dspark-kv-ssd

# 512 GiB logical quota per node; lower this for a smaller SSD.
MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES=549755813888
MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE=549755813888
MOONCAKE_OFFLOAD_BUCKET_KEYS_LIMIT=16
```

The launcher checks the config and both wheel types on both nodes before it
starts containers. It installs them offline inside each disposable container,
starts one Mooncake master plus one client per node, and applies the pinned
vLLM compatibility hotfix only when `ENABLE_KV_SSD=1`.

Start normally:

```bash
./start-deepseek-v4-flash-dspark.sh
```

For patch compatibility alone, without a GPU serve:

```bash
bash scripts/validate-mooncake-ssd-hotfix.sh
```

## Verify the live tier

The Mooncake client health endpoint is local to each node:

```bash
curl -fsS http://127.0.0.1:9300/health
curl -fsS http://127.0.0.1:9300/metrics | grep -E 'offload|storage|bytes'
du -sh ~/.cache/dspark-kv-ssd
```

Use one long conversation twice. The first request populates SSD; after GPU KV
eviction, a later request with the same prefix should log
`Recovered persisted Mooncake DSpark boundary` and non-zero storage reads.
Judge recovery by TTFT and loaded-prefix length, not output tok/s.

Illustrative live results from one 2x DGX Spark deployment on 2026-08-14:

| Prompt tokens | Verified SSD boundary | SSD read | Cold-recovery TTFT |
|---:|---:|---:|---:|
| 262,838 | 262,144 | 1,114,818,432 bytes | 13.42 s |
| 338,030 | 335,872 | 1,421,766,528 bytes | 25.94 s |

Subsequent warm turns in those sessions produced roughly 1.46-3.33 s TTFT.
These are operational observations, not a cross-machine benchmark.

## Limits and rollback

- This runtime patch targets exact source anchors in Anemll `0.1.1`; it refuses
  to patch an unknown image instead of guessing.
- Loads are synchronous and fail closed: an incomplete TP load raises before
  inference. It does not yet fall back cleanly to full recomputation.
- `GPU_MEMORY_UTILIZATION_SSD=0.75` trades some in-memory KV capacity for the
  Mooncake staging arena on GB10 unified memory.
- Cold NVMe recovery is slower than an in-memory prefix hit, and SSD capacity,
  endurance, and filesystem latency matter.
- KV files contain prompt-derived state. Protect the directory like model
  traffic and erase it according to your data-retention policy.
- The cache namespace includes the model revision and KV geometry. Do not reuse
  it across incompatible weights, quantization, TP, or block sizes.

Rollback is a full serve restart with:

```env
ENABLE_KV_SSD=0
```

The old SSD files are not deleted automatically.
