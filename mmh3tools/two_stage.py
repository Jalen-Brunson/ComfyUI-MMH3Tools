"""Context frames and a second stage, per chunk, for MMH3LoopingSampler.

CONTEXT FRAMES. The masked carry pins every carried row. `tail_custom` keeps only
the last `context_frames` of them pinned (rounded UP to whole H3 latent rows) and
hands the earlier ones back to the master's own mask, so the model redraws its own
lead-in up to a few hard frames instead of inheriting every artifact in the
previous chunk's tail. The redrawn rows are context only: the write-back starts at
the previous chunk's end, so the frames already rendered never change. Same row
math as ComfyUI-HybridWindows' `tail_context_rows`; copied, not imported, so
neither pack needs the other.

STAGE 2. Each chunk can finish with a second sampler pass: its own guider (so its
own MODEL, CFG and negative), sampler and SIGMAS. Stage 1's DENOISED estimate is
re-noised at stage2_sigmas[0] -- the looping sampler's phase-2 handoff, which is
what lets the second schedule be anything -- optionally after a learned latent
upscale. Upscaled chunks get FRESH video noise drawn on the target grid: lifting
the stage-1 noise (nearest resize) leaves it spatially correlated, which the
hybrid sampler measured as a corrupted high-res finish (2026-09-14). Audio is
never resized; it keeps stage 1's per-chunk noise.

TWO GRIDS. With an upscaler the chunk lives on two grids. Stage 2 pins the
previous chunk's FINISHED rows (the target-grid master); stage 1 pins either the
previous chunk's stage-1 estimate (`estimate`, no round trip) or its finished
rows area-downscaled (`downscaled_final`).
"""

import logging

import torch
import torch.nn.functional as F

import comfy.sample
from comfy.nested_tensor import NestedTensor

from .common import frame_at_latent

OVERLAP_PINS = ["full", "tail_custom"]
STAGE1_CARRY = ["estimate", "downscaled_final"]
UPSCALE_NONE = "none"
UPSCALE_PRECISIONS = ["fp32", "fp16", "bf16"]
# Stage-2 video noise on a new grid: the chunk's stage-1 seed plus this, so a re-roll
# (which moves the stage-1 seed) redraws both stages and no chunk's stage-2 draw can
# equal another chunk's stage-1 draw (those are base + chunk, a few hundred apart).
STAGE2_SEED_OFFSET = 1_000_003


def tail_context_rows(start, end, frames):
    """Smallest tail of whole H3 latent rows, ending at `end`, covering `frames` frames."""
    first = end
    frame_end = frame_at_latent(end)
    while first > start and frame_end - frame_at_latent(first) < frames:
        first -= 1
    return end - first


def upscale_choices():
    try:
        import folder_paths
        return [UPSCALE_NONE] + list(folder_paths.get_filename_list("latent_upscale_models"))
    except Exception:  # noqa: BLE001 - schema must build without the folder
        return [UPSCALE_NONE]


def target_size(video, megapixels, align=32):
    """Latent (h, w) the learned upscaler produces for `megapixels`.

    Mirrors the upscaler pack's own sizing (1024^2 pixels per MP, sides snapped to
    `align` pixels, /16 to latent), so the target grid is known before any call.
    """
    height, width = video.shape[-2:]
    target_h = (megapixels * 1024 * 1024 * height / width) ** 0.5
    target_w = target_h * width / height
    return tuple(max(1, round(round(v / align) * align / 16)) for v in (target_h, target_w))


def resize_frames(x, size, mode):
    """Spatially resize [B,C,T,H,W] frame by frame."""
    if tuple(x.shape[-2:]) == tuple(size):
        return x.clone()
    b, c, t, h, w = x.shape
    flat = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    flat = F.interpolate(flat.float(), size=tuple(size), mode=mode)
    return flat.reshape(b, t, c, *size).permute(0, 2, 1, 3, 4).contiguous().to(x.dtype)


def resize_mask(mask, size):
    return None if mask is None else resize_frames(mask, size, "nearest-exact")


def downscale_video(video, size):
    return resize_frames(video, size, "area")


def upscale_video(video, model_name, megapixels, precision, temporal_chunking, keep_loaded):
    """Learned 3D latent upscale of a chunk's video, through the installed node."""
    import nodes
    upscaler = nodes.NODE_CLASS_MAPPINGS.get("MinimaxH3LatentUpscaler3D")
    if upscaler is None:
        raise RuntimeError(
            "MMH3LoopingSampler: upscale_model is set but the MiniMax H3 latent upscaler "
            "node (MinimaxH3LatentUpscaler3D) is not installed. Install "
            "Comfyui_Minimax_h3_latent_Upscaler, or set upscale_model to none.")
    want = target_size(video, megapixels)
    out = upscaler.execute(
        {"samples": video}, model_name, {"mode": "megapixels", "megapixels": float(megapixels)},
        32, bool(temporal_chunking), not bool(keep_loaded),
        "cuda" if torch.cuda.is_available() else "cpu", precision)[0]["samples"]
    if tuple(out.shape[-2:]) != tuple(want) or out.shape[2] != video.shape[2]:
        raise RuntimeError(
            "MMH3LoopingSampler: the upscaler returned %s, expected %d latents at %s"
            % (tuple(out.shape), int(video.shape[2]), want))
    return out.to(device=video.device, dtype=torch.float32)


class Stage2Noise:
    """Noise for a chunk's second pass on a different grid.

    Video is drawn fresh on the target grid from `seed`; audio is exactly the noise
    stage 1 drew for this chunk (core draws a nested latent video-first from one
    generator, so it is regenerated from the stage-1 noise and the stage-1 latent).
    """

    def __init__(self, stage1_noise, stage1_latent, seed):
        self.stage1_noise = stage1_noise
        self.stage1_latent = stage1_latent
        self.seed = int(seed)

    def generate_noise(self, input_latent):
        video = input_latent["samples"].unbind()[0]
        noise_v = comfy.sample.prepare_noise(video, self.seed)
        first = self.stage1_noise.generate_noise(self.stage1_latent)
        parts = first.unbind() if isinstance(first, NestedTensor) else [first]
        if len(parts) < 2:
            return noise_v
        return NestedTensor([noise_v, parts[-1]])


def release_head(noise_mask, free, floor):
    """Hand carried video rows [0, free) back to `floor`; everything else unchanged."""
    if free <= 0 or noise_mask is None:
        return noise_mask
    if isinstance(noise_mask, NestedTensor):
        vm, am = noise_mask.unbind()
    else:
        vm, am = noise_mask, None
    vm = vm.clone()
    vm[:, :, :free] = floor.to(device=vm.device, dtype=vm.dtype)
    return NestedTensor([vm, am]) if am is not None else vm


def head_floor(in_mask_v, like, v0, free, accepted_t):
    """What released carry rows go back to: the master's own pin (a v2v keep region
    stays kept), except accepted rows, which release fully -- acceptance protects
    output bytes, and the write-back never touches rows below the previous end."""
    if in_mask_v is not None:
        floor = in_mask_v[:, :, v0:v0 + free].to(dtype=torch.float32).clone()
    else:
        floor = torch.ones([like.shape[0], 1, free] + list(like.shape[3:]), dtype=torch.float32)
    accepted = min(free, max(0, int(accepted_t) - int(v0)))
    if accepted:
        floor[:, :, :accepted] = 1.0
    return floor


def validate_sigmas(sigmas, label):
    if sigmas is None:
        return
    s = [float(x) for x in sigmas.flatten()]
    if len(s) < 2:
        raise ValueError("MMH3LoopingSampler: %s needs at least one step." % label)
    if any(x != x or x in (float("inf"), float("-inf")) for x in s):
        raise ValueError("MMH3LoopingSampler: %s contains a non-finite value." % label)
    if s[0] > 1.0 + 1e-6:
        raise ValueError("MMH3LoopingSampler: %s starts at %.6f; H3 sigmas are <= 1."
                         % (label, s[0]))
    if any(b >= a for a, b in zip(s, s[1:])):
        raise ValueError("MMH3LoopingSampler: %s must strictly descend." % label)
    if s[-1] != 0.0:
        raise ValueError("MMH3LoopingSampler: %s must end at 0 -- stage 2 finishes the chunk."
                         % label)


def describe_sigmas(sigmas):
    s = [float(x) for x in sigmas.flatten()]
    return "%d step%s from sigma %.4f" % (len(s) - 1, "" if len(s) == 2 else "s", s[0])


def log(msg, *args):
    logging.info("[MMH3LoopingSampler] " + msg, *args)
