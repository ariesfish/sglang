"""b12x NSA/MSA indexer integration for SM120.

Drop-in replacement for the DeepGEMM paged-MQA-logits path in
``dsa_indexer.py::_get_topk_paged``. Returns the same ``(q_rows, width_tokens)
float32 logits`` tensor that sglang's ``topk_transform`` consumes, so the
existing topk + page_table transform pipeline is reused unchanged.

Why logits (not pre-topk indices):
  sglang ``topk_transform`` (``dsa_topk_backend.py``) does BOTH topk selection
  AND the page_table_1 gather (``fast_topk_transform_fused`` /
  ``flashinfer.top_k_page_table_transform``). b12x ``index_topk_fp8`` does its
  own topk and returns raw logits-space indices, which would skip the
  page_table transform. So we use ``paged_decode_logits`` to produce logits and
  hand them to the existing ``topk_transform`` -- minimal intrusion, no
  duplicate transform logic.

K-cache layout alignment (verified by .scratch/test_b12x_mla_contract.py):
  sglang ``get_index_k_with_scale_buffer`` returns a buffer viewed as
  ``(N, page_size=64, 1, head_dim_with_sf=132)`` = 128 FP8 + 4 bytes FP32 scale.
  b12x ``INDEX_HEAD_DIM=128`` + ``+4`` scale = ``64*132`` row width. Identical
  physical byte layout, no reformat.

GLM vs DSV4:
  b12x indexer is model-agnostic for the paged path (same byte layout). GLM
  uses ``score_mode=NSA_RELU_SUM`` (default) -- no per-model branch needed
  here. MSA (``MSA_BILINEAR``) is a different scorer; not used by GLM-5.2.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Re-export for convenience so callers can import everything from one place.
from b12x.attention.indexer import (  # noqa: E402
    IndexerContiguousMetadata,
    IndexerPagedDecodeMetadata,
    contiguous_logits,
    paged_decode_logits,
    prepare_paged_indexer_metadata,
    build_paged_mqa_schedule_metadata,
)


class B12xIndexerRunner:
    """b12x paged indexer, replacing DeepGEMM fp8_paged_mqa_logits.

    Built once per ``DsaIndexer`` (mirrors how DeepGEMM reuses ``sm_count``).
    The schedule metadata is built per-forward by sglang (capture-safe, already
    done in ``init_forward_metadata``) and passed in via ``IndexerPagedDecodeMetadata``,
    so this runner holds no per-call mutable state.
    """

    def __init__(self, sm_count: int, index_topk: int):
        self.sm_count = sm_count
        self.index_topk = index_topk

    def get_paged_decode_logits(
        self,
        *,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        index_k_cache: torch.Tensor,
        real_page_table: torch.Tensor,
        cache_seqlens_int32: torch.Tensor,
        schedule_metadata: Optional[torch.Tensor] = None,
        page_size: int = 64,
        q_offset: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute paged MQA logits over the FP8 index cache.

        Returns ``(q_rows, width_tokens)`` float32 logits, matching the shape
        DeepGEMM ``fp8_paged_mqa_logits`` produces, for downstream
        ``topk_transform``.

        Args:
          q_fp8: ``[q_rows, 1, INDEX_HEAD_DIM=128]`` float8_e4m3fn query.
            sglang passes ``q_fp8.unsqueeze(1)`` for the non-dg-native path;
            b12x expects ``[q_rows, heads, 128]`` so the same shape works
            (heads=1 for the indexer head).
          weights: ``[q_rows, 1, 128]`` -> squeezed to 2D by b12x internally.
          index_k_cache: sglang view ``(N, page_size, 1, 132)``; b12x reads it
            as ``(N, page_size*(128+4))`` uint8 -- same bytes.
          real_page_table: sglang ``metadata.get_page_table_64()`` =
            ``[q_rows, num_pages]`` int32 page indices.
          cache_seqlens_int32: per-query raw token length (NOT clamped to 1).
          schedule_metadata: ``(num_sms+1, 2)`` int32 from
            ``build_paged_mqa_schedule_metadata`` (replaces DeepGEMM
            ``get_paged_mqa_logits_metadata``). If None, built here (non-capture
            path only).
          q_offset: real (unpadded) q length; logits are computed for the full
            ``q_fp8`` rows but the caller slices ``[:q_offset]`` before
            ``topk_transform`` (same as DeepGEMM path). Kept for API symmetry.
        """
        # Trim to real q rows if caller passed padded hidden states (attn_tp>1
        # or MAX_LEN padding). Mirrors the DeepGEMM ``q_fp8[:q_offset]`` slice.
        if q_offset is not None and q_offset < q_fp8.shape[0]:
            q_fp8 = q_fp8[:q_offset]
            weights = weights[:q_offset]
            # real_page_table / cache_seqlens are already per-query (not padded
            # by q_offset), so no trim needed there.

        if schedule_metadata is None:
            # Non-capture fallback: build schedule here. Capture path must
            # pre-build and pass via metadata.paged_mqa_schedule_metadata.
            schedule_metadata = build_paged_mqa_schedule_metadata(
                cache_seqlens_int32,
                block_kv=page_size,
                num_sms=self.sm_count,
            )

        metadata = IndexerPagedDecodeMetadata(
            real_page_table=real_page_table,
            cache_seqlens_int32=cache_seqlens_int32,
            paged_mqa_schedule_metadata=schedule_metadata,
        )

        logits = paged_decode_logits(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            metadata=metadata,
            page_size=page_size,
            score_mode=0,  # IndexerScoreMode.NSA_RELU_SUM (GLM-5.2)
        )
        return logits

    def get_contiguous_logits(
        self,
        *,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        kv_fp8,
        ks: torch.Tensor,
        ke: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contiguous (ragged) MQA logits, replacing DeepGEMM fp8_mqa_logits.

        Used by the prefill/extend path (_get_topk_ragged + forward_indexer).
        Returns ``(q_rows, k_tokens)`` float32 logits matching DeepGEMM output.

        Contract (verified by reading b12x contiguous_logits_reference):
          b12x k_start/k_end = DeepGEMM ks/ke, both per-token int32 ranges over
          the contiguous kv_fp8[0] dimension. valid_mask = (pos >= k_start) &
          (pos < k_end) -> half-open [k_start, k_end), matching DeepGEMM.
          kv_fp8 = (k_quant [K,128] float8_e4m3fn, k_scale [K] float32) tuple,
          identical to the DeepGEMM kv_fp8 tuple sglang builds.
        """
        metadata = IndexerContiguousMetadata(
            k_start=ks.to(torch.int32).contiguous(),
            k_end=ke.to(torch.int32).contiguous(),
        )
        logits = contiguous_logits(
            q_fp8=q_fp8,
            weights=weights,
            kv_fp8=kv_fp8,
            metadata=metadata,
            score_mode=0,  # IndexerScoreMode.NSA_RELU_SUM (GLM-5.2)
        )
        return logits


def build_b12x_paged_schedule(
    seqlens_32_2d: torch.Tensor,
    blocksize: int,
    sm_count: int,
) -> torch.Tensor:
    """Drop-in for ``deep_gemm.get_paged_mqa_logits_metadata``.

    Same signature + return shape ``(num_sms+1, 2) int32`` on the input device.
    Used by the 4 call sites in dsa_backend.py + dsa_indexer.py that build
    schedule metadata outside CUDA graph capture.
    """
    return build_paged_mqa_schedule_metadata(
        seqlens_32_2d,
        block_kv=blocksize,
        num_sms=sm_count,
    )


__all__ = [
    "B12xIndexerRunner",
    "build_b12x_paged_schedule",
    "IndexerPagedDecodeMetadata",
]
