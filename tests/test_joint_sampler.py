"""CPU integration checks; uses actual context slicing, packing and fusion."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), "/workspace/ComfyUI"]
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options
comfy.options.enable_args_parsing()
import torch
import comfy.conds
import comfy.utils
from comfy.nested_tensor import NestedTensor
from comfy.sampler_helpers import convert_cond
from comfy.samplers import encode_model_conds
from comfy.context_windows import _sampler_sample_wrapper
from mmh3tools.common import frames_to_audio_t, frame_at_latent
from mmh3tools.nodes_joint_sampler import (
    MMH3JointContextHandler, MMH3JointWindowSampler, bind_window_conditioning,
    prepare_masks, _WINDOW_KEY,
)
from mmh3tools.nodes_windows import _plan

torch.set_num_threads(2)


class FakeModel:
    latent_format = SimpleNamespace(temporal_downscale_ratio=4)

    def resize_cond_for_context_window(self, *args, **kwargs):
        return None


class JointSamplerTests(unittest.TestCase):
    def setUp(self):
        self.length, self.overlap, self.frames, self.total_t, self.windows = _plan(
            651, 243, 39, "standard_static")
        self.video = torch.randn(1, 24, self.total_t, 3, 5)
        self.audio = torch.randn(1, 32, 2, frames_to_audio_t(self.frames))
        self.packed, self.shapes = comfy.utils.pack_latents([self.video, self.audio])
        self.mask_v = torch.ones(1, 1, self.total_t, 3, 5)
        self.mask_v[..., 0, 0] = 0
        self.mask_a = torch.zeros(1, 1, 2, self.audio.shape[3])
        self.raw = {"conds": [
            [[torch.tensor([i]), {"ref_marker": f"control-{i}"}]
             for _ in range(i + 1)] for i in range(len(self.windows))]}
        bound = bind_window_conditioning(self.raw, self.windows)
        def extra_conds(**kwargs):
            return {"latent_shapes": comfy.conds.CONDConstant(self.shapes),
                    "denoise_mask": comfy.conds.CONDRegular(self.mask_v),
                    "audio_denoise_mask": comfy.conds.CONDRegular(self.mask_a)}
        self.conds = [encode_model_conds(
            extra_conds, convert_cond(bound), self.packed, "cpu", "positive"), None]
        self.sigmas = torch.tensor([1., .988235294, .972972973, .952380952,
                                    .923076923, .87804878, .8, .631578947, 0.])
        self.options = {"transformer_options": {"sample_sigmas": self.sigmas,
                                                "mmh3_control_frame0": 999}}

    def handler(self, device="gpu"):
        return MMH3JointContextHandler(self.windows, self.length, self.overlap, device)

    def run_handler(self, handler, packed, sigma):
        calls = []
        state = handler._build_window_state(packed, self.conds, FakeModel())
        snapshots = packed.clone()
        def calc(model, conds, x, t, options):
            index = conds[0][0][_WINDOW_KEY]
            self.assertEqual(len(conds[0]), index + 1)
            self.assertTrue(all(c["ref_marker"] == f"control-{index}" for c in conds[0]))
            w = state.prepare_window(self.windows[index], model)
            v, a = comfy.utils.unpack_latents(x, conds[0][0]["model_conds"]["latent_shapes"].cond)
            torch.testing.assert_close(v, w.get_tensor(state.latents[0]), rtol=0, atol=0)
            aw = w.get_window_for_modality(1)
            torch.testing.assert_close(a, aw.get_tensor(state.latents[1]), rtol=0, atol=0)
            mc = conds[0][0]["model_conds"]
            torch.testing.assert_close(mc["denoise_mask"].cond, w.get_tensor(self.mask_v))
            torch.testing.assert_close(mc["audio_denoise_mask"].cond, aw.get_tensor(self.mask_a))
            self.assertEqual(options["transformer_options"]["mmh3_control_frame0"],
                             frame_at_latent(w.index_list[0]))
            torch.testing.assert_close(t, sigma, rtol=0, atol=0)
            torch.testing.assert_close(packed, snapshots, rtol=0, atol=0)
            self.assertIsNone(conds[1])
            calls.append(index)
            return [torch.full_like(x, (index + 1) * 10), torch.zeros_like(x)]
        result = handler.execute(calc, FakeModel(), self.conds, packed, sigma, self.options)
        self.assertEqual(sorted(calls), [0, 1, 2])
        self.assertNotIn("context_window", self.options["transformer_options"])
        self.assertEqual(self.options["transformer_options"]["mmh3_control_frame0"], 999)
        self.assertTrue(all(c["model_conds"]["latent_shapes"].cond == self.shapes
                            for c in self.conds[0]))
        return result, state

    def test_eight_steps_keep_windows_synchronized_and_fuse_both_modalities(self):
        x = self.packed.clone()
        handler = self.handler()
        for sigma in self.sigmas[:-1]:
            result, state = self.run_handler(handler, x, sigma[None])
            outputs = comfy.utils.unpack_latents(result[0], self.shapes)
            for m, out in enumerate(outputs):
                dim = 2 if m == 0 else 3
                expected = torch.zeros(out.shape[dim])
                counts = torch.zeros_like(expected)
                for i, w in enumerate(self.windows):
                    indices = state.prepare_window(w, FakeModel()).get_window_for_modality(m).index_list
                    for local, global_index in enumerate(indices):
                        weight = min(local + 1, len(indices) - local)
                        expected[global_index] += (i + 1) * 10 * weight
                        counts[global_index] += weight
                self.assertTrue(bool((counts > 0).all()))
                shape = [1] * out.ndim
                shape[dim] = len(expected)
                torch.testing.assert_close(out, (expected / counts).reshape(shape).expand_as(out))
            self.assertEqual(int(torch.count_nonzero(result[1])), 0)
            x = x * .9 + result[0] * .1

    def test_window_order_and_accumulator_placement(self):
        baseline, _ = self.run_handler(self.handler(), self.packed, self.sigmas[:1])
        reverse = self.handler("cpu")
        original = reverse.get_context_windows
        reverse.get_context_windows = lambda *args: list(reversed(original(*args)))
        result, _ = self.run_handler(reverse, self.packed, self.sigmas[:1])
        torch.testing.assert_close(result[0], baseline[0])

    def test_binding_is_exact_and_does_not_mutate_original(self):
        bound = bind_window_conditioning(self.raw, self.windows)
        self.assertEqual([c[1][_WINDOW_KEY] for c in bound], [0, 1, 1, 2, 2, 2])
        self.assertNotIn(_WINDOW_KEY, self.raw["conds"][0][0][1])
        with self.assertRaisesRegex(ValueError, "conditioning sets"):
            bind_window_conditioning({"conds": self.raw["conds"][:2]}, self.windows)
        with self.assertRaisesRegex(ValueError, "keyframe"):
            bind_window_conditioning({"conds": [[[None, {"minimax_keyframes": [1]}]]]}, self.windows[:1])

    def test_masks_preserve_audio_and_spatial_regions_without_pinning_overlap(self):
        latent = {"samples": NestedTensor([self.video, self.audio]),
                  "noise_mask": NestedTensor([self.mask_v, self.mask_a])}
        output = prepare_masks(latent, torch.ones(self.frames, 6, 10), None, "max")
        v, a = output["noise_mask"].unbind()
        torch.testing.assert_close(v, self.mask_v)
        torch.testing.assert_close(a, self.mask_a)
        self.assertTrue(bool((v[:, :, 60:72, 1:, :] == 1).all()))
        without = {"samples": latent["samples"]}
        self.assertNotIn("noise_mask", prepare_masks(without, None, None, "max"))
        output = prepare_masks(without, None, torch.zeros(self.frames, 1, 1), "max")
        v, a = output["noise_mask"].unbind()
        self.assertTrue(bool((v == 1).all()))
        self.assertTrue(bool((a == 0).all()))

    def test_node_is_registered(self):
        from mmh3tools import NODES
        self.assertIn(MMH3JointWindowSampler, NODES)
        self.assertEqual(MMH3JointWindowSampler.define_schema().node_id, "MMH3JointWindowSampler")

    def test_freenoise_wrapper_reuses_video_noise_without_changing_audio_or_first_window(self):
        video = torch.arange(self.total_t, dtype=torch.float32).reshape(1, 1, -1, 1, 1)
        video = video.expand_as(self.video).clone()
        original, _ = comfy.utils.pack_latents([video, self.audio])
        handler = MMH3JointContextHandler(self.windows, self.length, self.overlap, freenoise=True)
        guider = SimpleNamespace(conds={"positive": self.conds[0]})
        calls = []
        def executor(g, sigmas, extra_args, callback, noise):
            self.assertIs(g, guider)
            self.assertIs(sigmas, self.sigmas)
            calls.append(noise)
            return noise
        def run(seed, h=handler):
            return _sampler_sample_wrapper(
                executor, guider, self.sigmas,
                {"model_options": {"context_handler": h}, "seed": seed}, None, original.clone())
        rng = torch.random.get_rng_state()
        output = run(41)
        v, a = comfy.utils.unpack_latents(output, self.shapes)
        torch.testing.assert_close(v[:, :, :72], video[:, :, :72], rtol=0, atol=0)
        torch.testing.assert_close(a, self.audio, rtol=0, atol=0)
        torch.testing.assert_close(v[0, 0, 72:132, 0, 0].sort().values,
                                   torch.arange(60, dtype=torch.float32), rtol=0, atol=0)
        torch.testing.assert_close(run(41), output, rtol=0, atol=0)
        self.assertFalse(torch.equal(run(42), output))
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
        torch.testing.assert_close(run(41, self.handler()), original, rtol=0, atol=0)
        self.assertEqual(len(calls), 4)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
