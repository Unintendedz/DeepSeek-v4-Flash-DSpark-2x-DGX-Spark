#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recover conservative Mooncake hits and complete SSD loads reliably.

The vLLM snapshot in the Anemll 0.1.1 image can reduce a valid external
hybrid-cache hit to zero while applying the EAGLE last-block rule.  The same
keys are visible to Mooncake and are loadable; only the coordinator's final
hit length is wrong.

When that happens, ask the same coordinator for the non-EAGLE prefix, then
walk back to a producer boundary that was really persisted.  The distinction
matters for hybrid caches: the producer intentionally keeps SWA tails every
``VLLM_PREFIX_CACHE_RETENTION_INTERVAL`` tokens, so subtracting one ordinary
cache block can select an interior boundary whose SWA keys do not exist.

The returned boundary is strictly below the stored tail.  The remaining
prefill recreates the hidden state required by DSpark, and a second
coordinator pass verifies that every hybrid group needed by the receive-side
load mask exists before the boundary is returned.

This image also queues a load from ``get_finished()`` and immediately returns.
The request then enters ``WAITING_FOR_REMOTE_KVS``; although the background
thread finishes the SSD read, the completion notification can be lost between
the two TP workers and the request waits forever.  For this optional SSD
profile, wait for the receive queue in that load-only executor step.  Both TP
ranks still read in parallel, and the scheduler receives completion from both
ranks in the same collective response.  Since the stock output aggregator can
still carry its per-rank counter across calls and lose that edge, receive-side
Mooncake completions are unioned inside that already-all-ranks collective.
Send-side completion counting is intentionally left unchanged.

Finally, the image exposes ``load_async=false`` on the scheduler but leaves
the worker implementation as a no-op plus an assertion.  Complete that
existing synchronous path: both TP workers load before their model forward,
so no request enters the broken remote-KV waiting state.  Any incomplete or
failed load raises before inference instead of using partially populated KV.
The connector facade in this snapshot is also a no-op, so wire it to the
worker implementation.

Idempotent.  Called by the compose entrypoint before ``exec vllm serve``.
"""
from pathlib import Path


P = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/"
    "kv_connector/v1/mooncake/store/worker.py"
)
AGG_P = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/"
    "kv_connector/utils.py"
)
CONNECTOR_P = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/"
    "kv_connector/v1/mooncake/store/connector.py"
)
MARK = "# [mooncake-eagle-lookup-hotfix] conservative stored boundary"
COMPLETION_MARK = "# [mooncake-ssd-completion-hotfix] finish load in this step"
AGG_MARK = "# [mooncake-ssd-completion-hotfix] collective already joined TP loads"
SYNC_MARK = "# [mooncake-ssd-sync-load-hotfix] load before model forward"
CONNECTOR_SYNC_MARK = "# [mooncake-ssd-sync-load-hotfix] delegate to worker"

ANCHOR = (
    "        _masks, hit_length = self.coord.find_longest_cache_hit(\n"
    "            block_hashes, token_len, ExternalCachedBlockPool(exists_set)\n"
    "        )\n"
    "        return hit_length\n"
)

REPLACEMENT = (
    "        _masks, hit_length = self.coord.find_longest_cache_hit(\n"
    "            block_hashes, token_len, ExternalCachedBlockPool(exists_set)\n"
    "        )\n"
    "\n"
    f"        {MARK}\n"
    "        # This image can collapse a valid hybrid DSpark/EAGLE hit to 0\n"
    "        # while popping the look-ahead block.  First derive the available\n"
    "        # prefix without the EAGLE pop.  Then step back to a boundary the\n"
    "        # producer actually persisted: SWA retention is coarse, so merely\n"
    "        # subtracting one cache block can choose an interior boundary with\n"
    "        # missing SWA keys and make the async load fail.\n"
    "        if self.coord.use_eagle and hit_length == 0:\n"
    "            _plain_masks, plain_hit_length = (\n"
    "                self.coord.find_longest_cache_hit(\n"
    "                    block_hashes,\n"
    "                    token_len,\n"
    "                    ExternalCachedBlockPool(exists_set),\n"
    "                    apply_eagle=False,\n"
    "                )\n"
    "            )\n"
    "            retention = self.coord.retention_interval or self.coord.lcm_block_size\n"
    "            conservative_hit = (plain_hit_length - 1) // retention * retention\n"
    "            while conservative_hit > 0:\n"
    "                _verified_masks, verified_hit = (\n"
    "                    self.coord.find_longest_cache_hit(\n"
    "                        block_hashes,\n"
    "                        conservative_hit,\n"
    "                        ExternalCachedBlockPool(exists_set),\n"
    "                        apply_eagle=False,\n"
    "                    )\n"
    "                )\n"
    "                if verified_hit >= conservative_hit:\n"
    "                    logger.info(\n"
    "                        \"Recovered persisted Mooncake DSpark boundary: \"\n"
    "                        \"eagle_hit=%d plain_hit=%d returned_hit=%d \"\n"
    "                        \"retention=%d\",\n"
    "                        hit_length,\n"
    "                        plain_hit_length,\n"
    "                        conservative_hit,\n"
    "                        retention,\n"
    "                    )\n"
    "                    return conservative_hit\n"
    "                conservative_hit -= retention\n"
    "        return hit_length\n"
)

COMPLETION_ANCHOR = (
    "        # Check completion of previously queued transfers\n"
    "        done_sending = (\n"
)

COMPLETION_REPLACEMENT = (
    f"        {COMPLETION_MARK}\n"
    "        # Loads are issued above from this same method.  Waiting here\n"
    "        # avoids losing an edge-triggered completion between TP ranks.\n"
    "        # The receive queue is per-rank, so the two SSD reads remain\n"
    "        # parallel across the distributed executor.\n"
    "        self.recv_request_queue.join()\n"
    "\n"
    "        # Check completion of previously queued transfers\n"
    "        done_sending = (\n"
)

AGG_ANCHOR = (
    "            update_finished_set(\n"
    "                kv_output.finished_recving, self._recv_remaining_count, finished_recving\n"
    "            )\n"
)

AGG_REPLACEMENT = (
    f"            {AGG_MARK}\n"
    "            # The Mooncake SSD worker waits for its receive queue before\n"
    "            # returning, and collective_rpc gathers every TP response\n"
    "            # before this aggregator runs.  Union the edge-triggered load\n"
    "            # notifications here; retain normal per-rank counting above\n"
    "            # for asynchronous send/store completion.\n"
    "            for req_id in kv_output.finished_recving or ():\n"
    "                finished_recving.add(req_id)\n"
    "                self._recv_remaining_count.pop(req_id, None)\n"
)

START_LOAD_ANCHOR = (
    "    def start_load_kv(\n"
    "        self,\n"
    "        metadata: MooncakeStoreConnectorMetadata,\n"
    "    ):\n"
    '        """No-op: loads are issued in get_finished() for overlap."""\n'
    "        pass\n"
)

START_LOAD_REPLACEMENT = (
    "    def start_load_kv(\n"
    "        self,\n"
    "        metadata: MooncakeStoreConnectorMetadata,\n"
    "    ):\n"
    '        """Load before forward when the scheduler requests sync mode."""\n'
    f"        {SYNC_MARK}\n"
    "        if self.load_async:\n"
    "            return\n"
    "\n"
    "        expected: set[str] = set()\n"
    "        for request in metadata.requests:\n"
    "            load_spec = request.load_spec\n"
    "            if load_spec is None or not load_spec.can_load:\n"
    "                continue\n"
    "            load_spec.token_len = load_spec.kvpool_cached_tokens\n"
    "            expected.add(request.req_id)\n"
    "            self.recv_request_queue.put(request)\n"
    "\n"
    "        if not expected:\n"
    "            return\n"
    "\n"
    "        # Each TP rank owns a receive queue; collective execution keeps\n"
    "        # the ranks parallel while each one waits for its local SSD read.\n"
    "        self.recv_request_queue.join()\n"
    "        completed: set[str] = set()\n"
    "        invalid_block_ids: set[int] = set()\n"
    "        for recv_thread in self.kv_recv_threads:\n"
    "            completed |= recv_thread.get_and_clear_finished_requests()\n"
    "            invalid_block_ids |= (\n"
    "                recv_thread.get_and_clear_block_ids_with_load_errors()\n"
    "            )\n"
    "        missing = expected - completed\n"
    "        if missing or invalid_block_ids:\n"
    "            raise RuntimeError(\n"
    "                \"Mooncake synchronous SSD load did not complete safely: \"\n"
    "                f\"missing_requests={sorted(missing)} \"\n"
    "                f\"invalid_blocks={len(invalid_block_ids)}\"\n"
    "            )\n"
)

GET_FINISHED_LOAD_ANCHOR = (
    "        # Issue async loads\n"
    "        for request in meta.requests:\n"
    "            load_spec = request.load_spec\n"
    "            if load_spec is None or not load_spec.can_load:\n"
    "                continue\n"
    "\n"
    "            load_spec.token_len = load_spec.kvpool_cached_tokens\n"
    "            self.recv_request_queue.put(request)\n"
    "\n"
    '        assert self.load_async, "load_async must be True for better performance."\n'
)

GET_FINISHED_LOAD_REPLACEMENT = (
    "        # Issue async loads. Sync-mode loads were completed before forward.\n"
    "        if self.load_async:\n"
    "            for request in meta.requests:\n"
    "                load_spec = request.load_spec\n"
    "                if load_spec is None or not load_spec.can_load:\n"
    "                    continue\n"
    "\n"
    "                load_spec.token_len = load_spec.kvpool_cached_tokens\n"
    "                self.recv_request_queue.put(request)\n"
)

CONNECTOR_START_LOAD_ANCHOR = (
    "    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:\n"
    "        # No-op: loads are issued in get_finished() for compute overlap.\n"
    "        pass\n"
)

CONNECTOR_START_LOAD_REPLACEMENT = (
    "    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:\n"
    f"        {CONNECTOR_SYNC_MARK}\n"
    "        assert self.connector_worker is not None\n"
    "        metadata = self._get_connector_metadata()\n"
    "        assert isinstance(metadata, MooncakeStoreConnectorMetadata)\n"
    "        self.connector_worker.start_load_kv(metadata)\n"
)


def main() -> None:
    src = P.read_text()
    changed = False

    if MARK not in src:
        if ANCHOR not in src:
            raise SystemExit("Mooncake lookup anchor not found; refusing to patch")
        src = src.replace(ANCHOR, REPLACEMENT, 1)
        changed = True

    if COMPLETION_MARK not in src:
        if COMPLETION_ANCHOR not in src:
            raise SystemExit("Mooncake completion anchor not found; refusing to patch")
        src = src.replace(COMPLETION_ANCHOR, COMPLETION_REPLACEMENT, 1)
        changed = True

    if SYNC_MARK not in src:
        if START_LOAD_ANCHOR not in src:
            raise SystemExit("Mooncake sync-load anchor not found; refusing to patch")
        if GET_FINISHED_LOAD_ANCHOR not in src:
            raise SystemExit(
                "Mooncake get-finished load anchor not found; refusing to patch"
            )
        src = src.replace(START_LOAD_ANCHOR, START_LOAD_REPLACEMENT, 1)
        src = src.replace(
            GET_FINISHED_LOAD_ANCHOR, GET_FINISHED_LOAD_REPLACEMENT, 1
        )
        changed = True

    if changed:
        P.write_text(src)
        print(f"[mooncake-ssd-hotfix] patched {P}")
    else:
        print(f"[mooncake-ssd-hotfix] already applied to {P}")

    agg_src = AGG_P.read_text()
    if AGG_MARK not in agg_src:
        if AGG_ANCHOR not in agg_src:
            raise SystemExit("Mooncake aggregator anchor not found; refusing to patch")
        AGG_P.write_text(agg_src.replace(AGG_ANCHOR, AGG_REPLACEMENT, 1))
        print(f"[mooncake-ssd-hotfix] patched {AGG_P}")
    else:
        print(f"[mooncake-ssd-hotfix] already applied to {AGG_P}")

    connector_src = CONNECTOR_P.read_text()
    if CONNECTOR_SYNC_MARK not in connector_src:
        if CONNECTOR_START_LOAD_ANCHOR not in connector_src:
            raise SystemExit(
                "Mooncake connector sync-load anchor not found; refusing to patch"
            )
        CONNECTOR_P.write_text(
            connector_src.replace(
                CONNECTOR_START_LOAD_ANCHOR, CONNECTOR_START_LOAD_REPLACEMENT, 1
            )
        )
        print(f"[mooncake-ssd-hotfix] patched {CONNECTOR_P}")
    else:
        print(f"[mooncake-ssd-hotfix] already applied to {CONNECTOR_P}")


if __name__ == "__main__":
    main()
