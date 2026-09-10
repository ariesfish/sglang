# fix.3(C): materialize interleave-CP KV pages before PD transfer.
#
# Problem: under DSA prefill-CP round-robin split (--cp-strategy interleave /
# dsa_prefill_cp_mode=round-robin-split), each CP rank's page pool only holds
# ITS OWN token rows (position % cp_size == cp_rank): ~1/8 of every page is
# written, the rest is allocated-but-unwritten. The PD sender's
# filter_kv_indices_for_cp_rank assigns each rank a CONTIGUOUS page block, so
# the shipped block is ~7/8 unwritten rows -> decode assembles garbage KV at
# most positions while all byte/page accounting checks out.
#
# Fix (option C): before the send, fill the unwritten rows on every rank:
#   1) each rank packs the rows it owns (per page, in-page positions
#      j % cp_size == cp_rank),
#   2) one intra-node all_gather over the attn_cp group,
#   3) each rank scatters the gathered peer rows into its own page slots.
# After this every rank holds the complete page set and the existing
# contiguous-block per-rank transfer is correct. Cost: one NVLink all-gather
# of the request's KV pages (~ms scale) per send_kv_chunk call; RDMA volume
# and the compute-side CP win are unchanged.
#
# Scope / caveats (v1):
#   - Only the main KV page groups (k/v latent buffers) and the DSA index
#     cache buffers are materialized. Request-slot state (mamba, dsa-tail,
#     aux) is not page-indexed and already handled by the existing CP paths.
#   - Ownership assumes the batch is a single request chunk (the long-prompt
#     regime CP targets). Multi-request batched chunks keep today's behavior
#     (no worse); see the per-call page-count symmetry guard.
#   - Only round-robin-split mode (interleave). Zigzag layouts need a
#     different ownership map.
#   - Requires page_size % cp_size == 0 (64 % 8 == 0).

import logging
import os
import time
from typing import List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default) not in ("0", "false", "False")


class _PagedByteBuf:
    """A contiguous pool buffer viewed as [num_page_slots, page_size, per_tok]."""

    __slots__ = ("view3", "page_size", "per_tok", "n_slots", "page_bytes")

    def __init__(self, tensor: torch.Tensor, page_bytes: int, page_size: int):
        assert tensor.is_contiguous()
        self.page_size = int(page_size)
        self.page_bytes = int(page_bytes)
        self.per_tok = int(page_bytes) // self.page_size
        if self.page_size * self.per_tok != int(page_bytes):
            raise ValueError(
                f"non-uniform page layout: page_bytes={page_bytes} "
                f"page_size={page_size}"
            )
        raw = tensor.view(torch.uint8)
        n_slots = raw.numel() // int(page_bytes)
        raw = raw[: n_slots * int(page_bytes)]  # guard: only the backed span
        self.n_slots = n_slots
        self.view3 = raw.reshape(self.n_slots, self.page_size, self.per_tok)


class CPKVMaterializer:
    """Fills interleave-CP KV pages across attn_cp ranks before PD transfer."""

    def __init__(
        self,
        buf_specs: List[tuple],
        page_size: int,
        cp_size: int,
        cp_rank: int,
    ):
        # buf_specs: list of (tensor, page_bytes) in transfer-order
        self.enabled = False
        self.reason = None
        self.page_size = int(page_size)
        self.cp_size = int(cp_size)
        self.cp_rank = int(cp_rank)

        if not _env_flag("SGLANG_CP_KV_MATERIALIZE"):
            self.reason = "env disabled"
            return
        if self.cp_size <= 1:
            self.reason = "cp_size<=1"
            return
        if self.page_size % self.cp_size != 0 or self.page_size < self.cp_size:
            self.reason = f"page_size {self.page_size} not divisible by cp {self.cp_size}"
            return
        try:
            if not os.environ.get("SGLANG_CP_MV_SKIP_MODE_CHECK"):
                from sglang.srt.layers.attention.dsa.utils import (
                    is_dsa_prefill_cp_round_robin_split,
                )

                if not is_dsa_prefill_cp_round_robin_split():
                    self.reason = "not round-robin-split CP mode"
                    return
        except Exception as e:  # pragma: no cover
            self.reason = f"round-robin probe failed: {e}"
            return

        try:
            self.bufs: List[_PagedByteBuf] = []
            for tensor, page_bytes in buf_specs:
                if tensor is None or page_bytes is None:
                    continue
                self.bufs.append(_PagedByteBuf(tensor, page_bytes, self.page_size))
        except Exception as e:
            self.reason = f"buffer view failed: {e}"
            return

        self.enabled = bool(self.bufs)
        if not self.enabled:
            self.reason = "no buffers"
            return
        col_dev = self.bufs[0].view3.device
        self.own_cols = torch.arange(
            self.cp_rank, self.page_size, self.cp_size, device=col_dev, dtype=torch.long
        )
        self.peer_cols = {
            q: torch.arange(
                q, self.page_size, self.cp_size, device=col_dev, dtype=torch.long
            )
            for q in range(self.cp_size)
        }
        total_mb = sum(b.n_slots * b.page_bytes for b in self.bufs) / 1e6
        logger.info(
            "[cp-kv-materialize] enabled: %d bufs, page=%d, cp=%d, pool=%.0fMB",
            len(self.bufs), self.page_size, self.cp_size, total_mb,
        )

    def _pack(self, page_ids, n):
        """Pack this rank's owned rows across all buffers -> (pack, m)."""
        packs = []
        m = self.own_cols.numel()  # page_size // cp_size
        for b in self.bufs:
            sel = b.view3.index_select(0, page_ids)  # [n, page, per_tok]
            packs.append(sel[:, self.own_cols, :].reshape(n * m * b.per_tok))
        pack = torch.cat(packs) if len(packs) > 1 else packs[0]
        if not pack.is_contiguous():
            pack = pack.contiguous()
        return pack, m

    def _scatter(self, q, page_ids, n, m, gathered, per_pack):
        """Write peer rank q's rows from the flat gather result into our pool.

        Pack layout is flat per-buffer chunks [n, m, per_tok_b] in bufs order.
        """
        cols = self.peer_cols[q]
        flat_base = q * per_pack
        offset = 0
        for b in self.bufs:
            sz = n * m * b.per_tok
            chunk = gathered[
                flat_base + offset : flat_base + offset + sz
            ].view(n, m, b.per_tok)
            tmp = b.view3.index_select(0, page_ids)  # writable copy
            tmp[:, cols, :] = chunk
            b.view3.index_copy_(0, page_ids, tmp)  # write back into pool
            offset += sz
        assert offset == per_pack

    # ------------------------------------------------------------------
    def materialize(self, page_ids: torch.Tensor) -> None:
        """page_ids: 1-D long tensor of local page slot ids for this request."""
        if not self.enabled:
            return
        from sglang.srt.layers.dp_attention import attn_cp_all_gather_into_tensor

        n = int(page_ids.numel()) if torch.is_tensor(page_ids) else len(page_ids)
        if n == 0:
            return
        dev = self.bufs[0].view3.device
        if torch.is_tensor(page_ids):
            page_ids = page_ids.to(dev, non_blocking=False).long()
        else:
            page_ids = torch.as_tensor(
                np.asarray(page_ids), dtype=torch.long, device=dev
            )

        # ---- symmetry guard: every rank must hold the same page count ----
        cnt = torch.tensor([n], dtype=torch.long, device=dev)
        cnt_all = torch.empty(self.cp_size, dtype=torch.long, device=dev)
        attn_cp_all_gather_into_tensor(cnt_all, cnt)
        if int((cnt_all != n).sum()) != 0:
            logger.warning(
                "[cp-kv-materialize] page-count mismatch across CP ranks "
                f"({cnt_all.tolist()}); skipping materialize for this request"
            )
            return

        # ---- overflow guard: page slots must exist in every buffer ----
        max_slot = int(page_ids.max().item())
        min_slots = min(b.n_slots for b in self.bufs)
        if max_slot >= min_slots:
            logger.error(
                "[cp-kv-materialize] page slot overflow: max=%d but smallest "
                "buffer holds %d slots; disabling",
                max_slot,
                min_slots,
            )
            self.enabled = False
            return

        t0 = time.perf_counter()

        # ---- pack own rows across all buffers into one contiguous buffer ----
        pack, m = self._pack(page_ids, n)

        # ---- one all_gather over the attn_cp group ----
        gathered = torch.empty(
            self.cp_size * pack.numel(), dtype=pack.dtype, device=pack.device
        )
        attn_cp_all_gather_into_tensor(gathered, pack)

        # ---- scatter peer rows into our own page slots ----
        per_pack = pack.numel()
        for q in range(self.cp_size):
            if q == self.cp_rank:
                continue  # our own rows are already in place
            self._scatter(q, page_ids, n, m, gathered, per_pack)

        if torch.cuda.is_available():
            torch.cuda.current_stream().synchronize()
        logger.info(
            "[cp-kv-materialize] filled %d pages, pack=%.1fMB, bufs=%d, %.1fms",
            n,
            pack.numel() / 1e6,
            len(self.bufs),
            (time.perf_counter() - t0) * 1000.0,
        )
