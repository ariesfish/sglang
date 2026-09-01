import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.moe.utils import (
    MoeA2ABackend,
    install_shared_experts_fusion_decision,
    is_shared_experts_fusion_disabled,
)
from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
from sglang.srt.runtime_context import (
    get_context,
    get_flags,
    get_parallel,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestGlm5NextSharedExpertFusionPolicy(CustomTestCase):
    """GLM-5.3-Flash fuses its shared expert into the routed MoE kernel,
    including under expert parallelism (DeepEP per-rank shared slots).

    Two gates must stay in lockstep (upstream 8b8676ba33): the decision the
    loader installs before any layer exists (``install_shared_experts_fusion_decision``
    consulting ``shared_experts_fusion_disable_reason``) drives the layer build
    (fused DeepseekV2MoE), while the wrapper's own
    ``determine_num_fused_shared_experts`` drives the weight-load remap
    ``mlp.shared_experts -> mlp.experts.<n_routed>``. A divergence builds fused
    layers but skips the remap, so the fused expert slot runs on uninitialized
    weights and every MoE layer output degenerates.
    """

    def setUp(self):
        super().setUp()
        cm = get_parallel().override(tp_size=8, moe_ep_size=8)
        cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)
        get_flags().moe.a2a_backend = MoeA2ABackend.DEEPEP
        self.addCleanup(
            lambda: setattr(get_flags().moe, "a2a_backend", None)
        )
        get_flags().moe.disable_shared_experts_fusion = None
        self.addCleanup(
            lambda: setattr(get_flags().moe, "disable_shared_experts_fusion", None)
        )

    def _publish(self, disable_fusion=False):
        override = get_context().override_server_args(
            disable_shared_experts_fusion=disable_fusion
        )
        override.install()
        self.addCleanup(override.restore)

    def _make_hf_config(self, n_shared_experts=1):
        return SimpleNamespace(
            text_config=SimpleNamespace(n_shared_experts=n_shared_experts)
        )

    def _cuda_caps(self):
        return (
            patch("sglang.srt.models.glm5_next._is_cuda", True),
            patch("sglang.srt.models.glm5_next._device_sm", 90),
        )

    def test_gate_allows_fusion_under_ep_and_deepep(self):
        """EP>1/DeepEP no longer vetoes fusion: the per-rank shared-slot path
        supports it (home-rank compute, no all-reduce double count)."""
        self._publish(disable_fusion=False)
        with (
            patch("sglang.srt.models.glm5_next._is_cuda", True),
            patch("sglang.srt.models.glm5_next._device_sm", 90),
        ):
            reason = (
                Glm5NextForConditionalGeneration.shared_experts_fusion_disable_reason(
                    self._make_hf_config(), None
                )
            )
            install_shared_experts_fusion_decision(
                Glm5NextForConditionalGeneration, self._make_hf_config(), None
            )
        self.assertIsNone(reason)
        # The runner that builds the layers therefore builds them fused.
        self.assertFalse(is_shared_experts_fusion_disabled())

    def test_wrapper_matches_layer_build_under_ep_and_deepep(self):
        """The weight-remap count must agree with the layer build decision.

        Guards the f609d67 regression: the wrapper gate resolved OFF at EP>1
        while the loader-installed decision resolved ON, so the fused expert
        slot never received its weights.
        """
        self._publish(disable_fusion=False)
        with (
            patch("sglang.srt.models.glm5_next._is_cuda", True),
            patch("sglang.srt.models.glm5_next._device_sm", 90),
        ):
            install_shared_experts_fusion_decision(
                Glm5NextForConditionalGeneration, self._make_hf_config(), None
            )
            wrapper = Glm5NextForConditionalGeneration.__new__(
                Glm5NextForConditionalGeneration
            )
            wrapper.config = SimpleNamespace(n_shared_experts=1)
            wrapper.quant_config = None
            wrapper.determine_num_fused_shared_experts()
        self.assertEqual(wrapper.num_fused_shared_experts, 1)
        self.assertFalse(is_shared_experts_fusion_disabled())

    def test_explicit_disable_flag_still_wins(self):
        self._publish(disable_fusion=True)
        with (
            patch("sglang.srt.models.glm5_next._is_cuda", True),
            patch("sglang.srt.models.glm5_next._device_sm", 90),
        ):
            install_shared_experts_fusion_decision(
                Glm5NextForConditionalGeneration, self._make_hf_config(), None
            )
            wrapper = Glm5NextForConditionalGeneration.__new__(
                Glm5NextForConditionalGeneration
            )
            wrapper.config = SimpleNamespace(n_shared_experts=1)
            wrapper.quant_config = None
            wrapper.determine_num_fused_shared_experts()
        self.assertTrue(is_shared_experts_fusion_disabled())
        self.assertEqual(wrapper.num_fused_shared_experts, 0)

    def test_gate_disables_without_shared_experts(self):
        self._publish(disable_fusion=False)
        with (
            patch("sglang.srt.models.glm5_next._is_cuda", True),
            patch("sglang.srt.models.glm5_next._device_sm", 90),
        ):
            reason = (
                Glm5NextForConditionalGeneration.shared_experts_fusion_disable_reason(
                    self._make_hf_config(n_shared_experts=None), None
                )
            )
        self.assertIn("No shared experts", reason)

    def test_gate_disables_off_cuda(self):
        self._publish(disable_fusion=False)
        with patch("sglang.srt.models.glm5_next._is_cuda", False):
            reason = (
                Glm5NextForConditionalGeneration.shared_experts_fusion_disable_reason(
                    self._make_hf_config(), None
                )
            )
        self.assertIn("CUDA", reason)

    def test_gate_disables_pre_sm80(self):
        self._publish(disable_fusion=False)
        with (
            patch("sglang.srt.models.glm5_next._is_cuda", True),
            patch("sglang.srt.models.glm5_next._device_sm", 75),
        ):
            reason = (
                Glm5NextForConditionalGeneration.shared_experts_fusion_disable_reason(
                    self._make_hf_config(), None
                )
            )
        self.assertIn("SM80", reason)

    def test_gate_vetoes_mixed_precision_quant(self):
        """A quant config that keeps the shared expert at a higher precision
        than the routed experts cannot fuse, same veto as DeepSeek's gate."""
        self._publish(disable_fusion=False)
        mixed = SimpleNamespace(
            get_name=lambda: "quark", can_fuse_shared_expert=lambda: False
        )
        with (
            patch("sglang.srt.models.glm5_next._is_cuda", True),
            patch("sglang.srt.models.glm5_next._device_sm", 90),
        ):
            reason = (
                Glm5NextForConditionalGeneration.shared_experts_fusion_disable_reason(
                    self._make_hf_config(), mixed
                )
            )
        self.assertIn("higher precision", reason)


if __name__ == "__main__":
    unittest.main()
