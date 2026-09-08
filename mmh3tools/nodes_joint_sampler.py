"""Experimental full-denoise H3 sampling with shared, revisable overlap."""

import logging

import torch
from comfy.context_windows import (
    create_prepare_sampling_wrapper,
    create_sampler_sample_wrapper,
    get_matching_context_schedule,
    get_matching_fuse_method,
)
from comfy_api.latest import io
from comfy_extras.nodes_custom_sampler import Guider_Basic, SamplerCustomAdvanced

from .common import frame_at_latent, latents_to_frames, unpack_av
from .nodes_loop import per_row_mask_is_continuous
from .nodes_looping_sampler import (
    MASK_MODES, _audio_profile_to_mask, _mask_to_video_latent, _sliced_mask,
    _split_mask, _video_mask_to_audio,
)
from .nodes_multiprompt import MMH3CondSet
from .nodes_windows import MMH3ContextHandler, _plan, _window_frame_spans

_WINDOW_KEY = "mmh3_joint_window_index"


def bind_window_conditioning(cond_set, windows):
    conds = cond_set["conds"]
    if len(conds) != len(windows):
        raise ValueError(
            f"Joint sampler planned {len(windows)} windows but received {len(conds)} "
            "conditioning sets. Match total, window and overlap frames in the "
            "reference encoder and sampler.")
    bound = []
    for i, cond in enumerate(conds):
        if not cond:
            raise ValueError(f"Window {i + 1} has no conditioning.")
        for tokens, metadata in cond:
            if metadata.get("minimax_keyframes"):
                raise ValueError("Joint sampler does not support keyframe guides yet.")
            bound.append([tokens, {**metadata, _WINDOW_KEY: i}])
    return bound


class MMH3JointContextHandler(MMH3ContextHandler):
    def __init__(self, windows, length, overlap, accumulator_device="gpu", freenoise=False):
        super().__init__(
            context_schedule=get_matching_context_schedule("standard_static"),
            fuse_method=get_matching_fuse_method("pyramid"),
            context_length=length, context_overlap=overlap, context_stride=1,
            closed_loop=False, dim=2, freenoise=freenoise, causal_window_fix=False,
            split_conds_to_windows=False, accumulator_device=accumulator_device,
        )
        self.window_indices = {
            tuple(window.index_list): i for i, window in enumerate(windows)
        }

    def get_context_windows(self, model, x_in, model_options):
        windows = super().get_context_windows(model, x_in, model_options)
        if {tuple(w.index_list) for w in windows} != set(self.window_indices):
            raise ValueError("Sampling windows changed after conditioning was assigned.")
        return windows

    def get_resized_cond(self, cond_in, x_in, window, device=None):
        if cond_in is None:
            return None
        index = self.window_indices[tuple(window.index_list)]
        selected = [c for c in cond_in if c.get(_WINDOW_KEY) == index]
        if not selected:
            raise ValueError(f"Missing conditioning for joint window {index + 1}.")
        return super().get_resized_cond(selected, x_in, window, device)

    def evaluate_context_windows(self, calc_cond_batch, model, x_in, conds,
                                 timestep, enumerated_context_windows,
                                 model_options, window_state, total_windows=None,
                                 device=None, first_device=None):
        results = []
        for item in enumerated_context_windows:
            # Core writes context_window into these options. Keep each window's
            # control offset local, including on exceptions or cancelled renders.
            options = {**model_options, "transformer_options": {
                **model_options.get("transformer_options", {}),
                "mmh3_control_frame0": frame_at_latent(item[1].index_list[0]),
            }}
            results.extend(super().evaluate_context_windows(
                calc_cond_batch, model, x_in, conds, timestep, [item], options,
                window_state, total_windows, device, first_device))
        return results


def prepare_masks(latent, denoise_mask, audio_denoise_mask, mode):
    video, audio = unpack_av(latent, allow_video_only=True)
    total_t = video.shape[2]
    total_a = 0 if audio is None else audio.shape[3]
    mask_v, mask_a = _split_mask(latent)
    if denoise_mask is not None or audio_denoise_mask is not None:
        if not per_row_mask_is_continuous():
            raise RuntimeError("Joint sampler masks require H3 per-row mask support.")
    if denoise_mask is not None:
        mv = _mask_to_video_latent(
            denoise_mask.to(video.device), total_t, video.shape[3], video.shape[4], mode)
        mask_v = mv if mask_v is None else torch.minimum(mask_v.to(mv), mv)
    if audio_denoise_mask is not None:
        if audio is None:
            raise ValueError("An audio mask needs an AV latent containing audio.")
        am = _mask_to_video_latent(audio_denoise_mask.to(video.device), total_t, 1, 1, mode)
        profile = _video_mask_to_audio(am, total_t, total_a, mode)
        ma = _audio_profile_to_mask(profile, audio)
        mask_a = ma if mask_a is None else torch.minimum(mask_a.to(ma), ma)
    out = dict(latent)
    if mask_v is not None or mask_a is not None:
        out["noise_mask"] = _sliced_mask(
            video, audio, mask_v, mask_a, 0, total_t, 0, total_a)
    return out


class MMH3JointWindowSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3JointWindowSampler",
            display_name="MMH3 Joint Window Sampler (experimental)",
            category="MMH3Tools/sampling", is_experimental=True,
            description=("Sample all windows at each sigma, fuse overlap predictions, "
                         "then advance one shared latent. Full-denoise quality is unvalidated. "
                         "Use matching window settings on Reference MultiPrompt; fresh runs only."),
            inputs=[
                io.Model.Input("model"), io.Noise.Input("noise"),
                io.Sampler.Input("sampler"), io.Sigmas.Input("sigmas"),
                MMH3CondSet.Input("cond_set"), io.Latent.Input("latent"),
                io.Int.Input("window_frames", default=243, min=5, max=100000, step=17),
                io.Int.Input("overlap_frames", default=39, min=0, max=100000, step=17),
                io.Combo.Input("accumulator_device", options=["gpu", "cpu"], default="gpu"),
                io.Combo.Input("denoise_mask_mode", options=MASK_MODES, default="max"),
                io.Boolean.Input("freenoise", default=False, optional=True,
                                 tooltip="Reuse shuffled initial video noise across windows. Control noise and audio are unchanged."),
                io.Mask.Input("denoise_mask", optional=True),
                io.Mask.Input("audio_denoise_mask", optional=True),
            ],
            outputs=[io.Latent.Output(display_name="latent"),
                     io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, model, noise, sampler, sigmas, cond_set, latent,
                window_frames, overlap_frames, accumulator_device="gpu",
                denoise_mask_mode="max", denoise_mask=None, audio_denoise_mask=None,
                freenoise=False):
        video, _ = unpack_av(latent, allow_video_only=True)
        total_frames = latents_to_frames(video.shape[2])
        length, overlap, total_f, total_t, windows = _plan(
            total_frames, window_frames, overlap_frames, "standard_static")
        if total_t != video.shape[2] or any(w.index_list[0] % 5 for w in windows):
            raise ValueError("Joint sampler needs H3-aligned latent lengths and window starts.")
        conditioning = bind_window_conditioning(cond_set, windows)
        prepared = prepare_masks(latent, denoise_mask, audio_denoise_mask, denoise_mask_mode)
        patched = model.clone()
        patched.model_options["context_handler"] = MMH3JointContextHandler(
            windows, length, overlap, accumulator_device, freenoise=freenoise)
        create_prepare_sampling_wrapper(patched)
        if freenoise:
            create_sampler_sample_wrapper(patched)
        guider = Guider_Basic(patched)
        guider.set_conds(conditioning)
        spans = _window_frame_spans(windows, total_f)
        report = (
            f"Joint sampling: {len(windows)} windows, {total_f} frames; "
            f"window {length} / overlap {overlap} video latents.\n"
            f"Frame spans (inclusive): {spans}\n"
            f"Seed {noise.seed}; {len(sigmas) - 1} sampler intervals; "
            "one global noise draw, pyramid prediction fusion, CFG 1.\n"
            f"Initial video noise: {'FreeNoise shuffling' if freenoise else 'ordinary global noise'}.\n"
            "Overlap remains editable at every step. Input spatial/audio masks are preserved.\n"
            "Experimental: inspect continuity, motion adherence and skin texture."
        )
        logging.info("[MMH3JointWindowSampler] %s", report)
        output = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, prepared)[1]
        return io.NodeOutput(output, report)
