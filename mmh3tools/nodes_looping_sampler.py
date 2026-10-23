"""Chained AV generation in one node: fill a clip chunk by chunk, one graph cost.

WHY ONE NODE. Driving N chunks from the graph costs a copy of every downstream
node per chunk -- sampler, decode, save -- and a graph that size is what starts
breaking ComfyUI. A Python loop inside one node costs the same whether it runs
4 chunks or 40. The trade is that nothing inside the loop can be a graph node,
so every prompt has to exist BEFORE sampling starts. That is what the cond_set
is: MMH3ReferenceMultiPrompt encodes all N up front, in one text-encoder load.

THE LATENT IS THE WHOLE CLIP. You hold a song of known length; you do not know
how many chunks that is. So the total is given and the chunk count is DERIVED,
the way LTXAVTools' looping sampler does it -- an earlier version of this node
took ONE CHUNK's latent and a chunk count, which made the total emergent and
every question about it ("how long will this be", "which chunk owns frame N",
"which slice of audio does chunk 3 get") an exercise in reconstruction.

The schedule comes from `_plan`, the SAME function MMH3WindowPlan and
MMH3SplitAudioToWindows use. So chunks and windows are the same thing by
construction: the prompt written against window 3's audio is the prompt chunk 3
renders, and `window_count` really is the chunk count rather than merely looking
like it.

AUDIO COMES FREE. The master latent holds the whole track, so each chunk slices
its own span out of it. No per-chunk audio input, no risk of every chunk getting
the first chunk's music.

TWO CARRY ROUTES.

`mask` masks the chunk's first `overlap` latents -- they already hold the
previous chunk's output, since chunks are slices of one master -- so the model
conditions on them without denoising them. Needs per-row masking (#15375).

`keyframe` passes those same latents as a GUIDE anchored at frame 0: re-injected
every step, never denoised, carrying a multi-step clip plus its audio at the same
cond_t. Needs #15439, and the guide-origin wrap when references are also present.

Neither needs a join. Chunks are written back into the master in place, so there
is no trim, no grid-safe cut, and no frames lost per seam -- all of which existed
only because the old shape concatenated separately-allocated chunks.

GUIDES ARE BUILT HERE, per chunk, and stale ones are stripped off incoming
conditioning first -- an upstream guide node or a cond cached from a previous
run would anchor this chunk to somebody else's frames. Straight from
LTXAVTools, where the same leak had the same cause.

GUIDER COPYING is the part that is easy to get wrong. copy.copy is shallow, so
new_g.original_conds is the SAME dict object as the source guider's, and
set_conds assigns into it -- chunk 0 would overwrite the BASE conditioning and
every later chunk would read chunk 0's conds back as "base". Rebinding the dict
per chunk is the fix. Learned in LTXAVTools, where the symptom was every chunk
getting chunk 0's speaker.
"""

import copy
import logging

import node_helpers
import torch
import comfy.utils
from comfy_api.latest import io
from comfy.nested_tensor import NestedTensor

from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced, SplitSigmas

from . import two_stage as TS
from .common import (AUDIO_LATENT_FPS, AUDIO_T_DIM, FPS, LATENTS_PER_GROUP,
                     LATENT_BASE, VAE_SPATIAL, VIDEO_T_DIM,
                     frame_at_latent, frames_to_audio_t, frames_to_latents,
                     latents_to_frames, pack_av, unpack_av)
from .nodes_loop import per_row_mask_is_continuous
from .nodes_multiprompt import MMH3CondSet
from .nodes_preview import begin_preview
from .nodes_windows import _audio_index_at, _plan, _window_frame_spans

_GUIDE_KEYS = ("minimax_keyframes", "minimax_frame_count")
MASK_MODES = ["max", "min", "mean", "last"]


def _strip_guide_keys(cond, label):
    """Remove keyframe bookkeeping from conditioning.

    This node registers ALL of its own guides, per chunk. Anything arriving
    pre-registered is stale -- an upstream guide node, or a cond cached from a
    previous run -- and would anchor this chunk to the wrong frames. Straight
    from LTXAVTools, where the same leak had the same cause.
    """
    out, stripped = [], False
    for t, d in cond:
        if any(k in d for k in _GUIDE_KEYS):
            d = {k: v for k, v in d.items() if k not in _GUIDE_KEYS}
            stripped = True
        out.append([t, d])
    if stripped:
        logging.info("[MMH3LoopingSampler] stripped stale keyframes from %s; this node "
                     "builds its own per chunk", label)
    return out


def _has_refs(cond):
    return any("minimax_refs" in d and d["minimax_refs"] for _t, d in cond)


def _guides_available():
    """Whether #15439's any-index guides are present.

    Detected, not assumed: stock raises on any anchor that is not first/last,
    and a guide carrying a multi-step clip is not expressible at all.
    """
    try:
        import inspect
        import comfy.ldm.minimax.model as mm
        return "only first/last keyframe anchors" not in inspect.getsource(
            mm.PackedLayout.__init__)
    except Exception:
        return False


def _guide_origin_correct():
    """Whether a guide anchors on the TARGET origin when references are present.

    #15439 anchors to text_len, but the target begins after the refs. See
    patch_guide_origin; this asks the patch first and probes only as a fallback.
    """
    from . import patch_guide_origin
    if patch_guide_origin.is_applied():
        return True
    try:
        import comfy.ldm.minimax.model as mm
        lay = mm.PackedLayout(
            8, 7, 4, 4, 8,
            keyframes=[{"resolved_frame_index": 0,
                        "latent": torch.zeros([1, 24, 1, 4, 4])}],
            refs=[{"kind": "image", "latent_h": 4, "latent_w": 4,
                   "latent": torch.zeros([1, 24, 1, 4, 4])}])
        seg = {k: a for a, _b, k in lay.segments}
        return abs(float(lay.position_ids[seg["cond"], 0])
                   - float(lay.position_ids[seg["video"], 0])) < 1e-6
    except Exception:
        return False


def _raw_conds(guider):
    """This guider's (positive, negative) in the form set_conds accepts.

    negative is None for a BasicGuider, which has no such key at all --
    Guider_Basic.set_conds takes ONE argument. Indexing for it raises KeyError,
    and calling set_conds with two raises TypeError.
    """
    if hasattr(guider, "raw_conds"):
        return guider.raw_conds
    conds = getattr(guider, "original_conds", {}) or {}
    return (conds.get("positive"), conds.get("negative"))


def _chunk_guider(guider, positive, frame0=None):
    """This chunk's guider: the wired one, with its POSITIVE replaced.

    `frame0` is published to the model as `mmh3_control_frame0` so anything reading
    the sequence per chunk -- the Fun ControlNet wrapper, for one -- knows WHERE in
    the clip this chunk starts. Core's controlnet picks its hint frames from index 0
    and caches by shape, and every chunk shares a shape, so without this every chunk
    is driven by the control video's opening frames with no error raised.
    """
    new_g = copy.copy(guider)
    # SHALLOW copy shares original_conds; set_conds assigns into it. Rebind
    # before touching it or chunk 0 clobbers the base conditioning.
    new_g.original_conds = dict(guider.original_conds)
    if frame0 is not None:
        # model_options is shared by the same shallow copy, so rebind BOTH levels
        # rather than writing the offset into the caller's guider.
        opts = dict(getattr(guider, "model_options", {}) or {})
        opts["transformer_options"] = dict(opts.get("transformer_options", {}))
        opts["transformer_options"]["mmh3_control_frame0"] = int(frame0)
        new_g.model_options = opts
    _, negative = _raw_conds(guider)
    if negative is None:
        new_g.set_conds(positive)              # Guider_Basic: no CFG, no negative
    else:
        new_g.set_conds(positive, negative)
    new_g.raw_conds = (positive, negative)
    return new_g


def _run_sampling(noise, guider, sampler, sigmas, av_init,
                  sampling_start_step, sampling_end_step,
                  phase2_sampler=None, phase2_guider=None, phase2_start_step=0,
                  carry_noise=None):
    """Run one chunk's schedule as sequential resample-continuation segments.

    NOTHING HERE WINDOWS A GUIDE. A keyframe guide is not a per-step influence:
    it is registered on the conditioning and re-injected every step, so it is
    structural for the whole chunk. Releasing it mid-schedule would mean changing
    the packed layout between steps, which is not expressible -- and the phase-2
    guider is deliberately given this chunk's FULL conditioning, so it cannot
    release it either. To drop a guide you need a separate pass whose
    conditioning never had it.

    sampling_start_step / sampling_end_step WINDOW THE SCHEDULE, using core
    SplitSigmas' own slicing (first output sigmas[:step+1], second output
    sigmas[step:], sharing the boundary sigma). Start skips the steps before it,
    re-noising the incoming latent to that sigma; end discards the ones after,
    leaving a partially denoised latent. Both are ABSOLUTE indices into the
    incoming schedule, so pass 1 `end N` hands off to pass 2 `start N` with no
    arithmetic on the user's part.

    The remaining split point is phase2_start_step, where the sampler/guider pair
    switches for dual-solver schedules. It is rebased onto the sliced schedule.

    Straight from LTXAVTools' looping sampler, which solved this first.
    """
    start = max(0, int(sampling_start_step))
    if start > 0:
        if sampling_end_step <= start:
            raise ValueError(
                "MMH3LoopingSampler: sampling_start_step (%d) must be below "
                "sampling_end_step (%d) -- that window contains no steps. These are "
                "absolute indices into the sigma schedule: start skips the steps "
                "before it, end drops the ones after."
                % (start, int(sampling_end_step)))
        _, sigmas = SplitSigmas().get_sigmas(sigmas, start)
        if len(sigmas) <= 1:
            raise ValueError(
                "MMH3LoopingSampler: sampling_start_step %d is at or past the end of "
                "the sigma schedule -- nothing left to sample. Lower it, or give the "
                "scheduler more steps." % start)

    use_phase2 = phase2_sampler is not None and int(phase2_start_step) > 0
    phase2_local = int(phase2_start_step) - start
    end_step = int(sampling_end_step) - start

    # LOCAL PATCH 2026-09-07 (carry_noise): the steps whose sigma >= until_sigma run as their
    # own segment with the carried rows at a fractional mask (H3 derives per-row timesteps
    # from the mask once per SamplerCustomAdvanced call, so a per-step mask change would
    # mislabel the rows -- a separate segment keeps labels and noise consistent); afterwards
    # the carried rows are restored byte-original and the remaining steps pin them exactly.
    cn_cut = None
    if carry_noise is not None and carry_noise[0] > 0.0 and carry_noise[2] > 0:
        k = 0
        while k < len(sigmas) and float(sigmas[k]) >= float(carry_noise[1]):
            k += 1
        if 0 < k < end_step:
            cn_cut = k
    cut_points = sorted(
        p for p in (({phase2_local} if use_phase2 else set()) | ({cn_cut} if cn_cut else set()))
        if 0 < p < end_step)

    segments = []
    remaining = sigmas
    prev = 0
    for p in cut_points:
        seg, remaining = SplitSigmas().get_sigmas(remaining, p - prev)
        segments.append((prev, seg))
        prev = p
    tail, _ = SplitSigmas().get_sigmas(remaining, end_step - prev)
    segments.append((prev, tail))

    current = av_init
    orig_mask = av_init.get("noise_mask")
    orig_head = None
    if cn_cut is not None:
        level, _until, carried = carry_noise
        ov, oa = unpack_av(av_init, "chunk input")
        orig_head = ov[:, :, :carried].clone()
    for seg_start, seg_sigmas in segments:
        if len(seg_sigmas) <= 1:
            continue
        seg_sampler, seg_guider = sampler, guider
        if use_phase2 and seg_start >= phase2_local:
            seg_sampler = phase2_sampler
            seg_guider = phase2_guider if phase2_guider is not None else guider
        seg_init = current
        noised = cn_cut is not None and seg_start < cn_cut
        if noised:
            seg_init = dict(current)
            nm = current.get("noise_mask")
            if nm is not None:
                parts = list(nm.unbind()) if isinstance(nm, NestedTensor) else [nm]
                vm = parts[0].clone()
                vm[:, :, :carried] = torch.maximum(vm[:, :, :carried], torch.full_like(vm[:, :, :carried], float(level)))
                parts[0] = vm
                seg_init["noise_mask"] = NestedTensor(parts) if isinstance(nm, NestedTensor) else vm
        _, current = SamplerCustomAdvanced().sample(
            noise, seg_guider, seg_sampler, seg_sigmas, seg_init)
        if noised:
            # the carried rows were partially re-estimated at high sigma: put the original
            # context back so the later (pinned) steps and the write-back see exact bytes
            cv, ca = unpack_av(current, "segment output")
            cv = cv.clone()
            cv[:, :, :carried] = orig_head.to(cv.dtype)
            current = pack_av(current, cv, ca, noise_mask=orig_mask)
    return current


def _chunk_noise(noise, index):
    """A distinct noise per chunk. Reusing one object gives every chunk the
    same noise, which reads as the model refusing to advance."""
    if index == 0 or not hasattr(noise, "seed"):
        return noise
    n = copy.copy(noise)
    n.seed = int(noise.seed) + index
    return n


def _parse_indices(text, total_frames):
    """Comma-separated GLOBAL pixel-frame indices; negatives count from the end.

    Resolved here rather than passed through: PackedLayout takes a negative
    literally, so cond_t would fall below text_len and collide with the text
    token positions. Out of range is an error -- silently dropping a keyframe
    the user asked for is worse than stopping.
    """
    out = []
    for piece in (text or "").split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            v = int(piece)
        except ValueError:
            raise ValueError(
                "MMH3LoopingSampler: keyframe_indices has %r in it, which is not a "
                "whole number. Expected something like '0, 60, -1'." % piece)
        if v < 0:
            v += total_frames
        if not 0 <= v < total_frames:
            raise ValueError(
                "MMH3LoopingSampler: keyframe index %s is outside the clip, which is "
                "%d frames (0-%d)." % (piece, total_frames, total_frames - 1))
        out.append(v)
    return out


def _fit_keyframe(px, tgt_w, tgt_h, is_opener):
    """Resize one keyframe still to the TARGET grid. -> (pixels, note or None).

    Keyframe rows share the target's spatial grid -- PackedLayout reads only the
    latent's TIME dim and sizes the segment from the target, so a still at any
    other resolution reserves the wrong number of rows and dies deep in the model
    with a broadcast error that names nothing. The other two keyframe paths already
    handle this (MMH3ImageKeyframe resizes internally, MMH3LatentKeyframe has an
    opt-in guard); this is the third, and it used to just encode whatever arrived.

    Resizing here rather than refusing is deliberate: a ladder runs the same still
    against 2-3 different target resolutions, and making the graph carry a resize
    per stage is busywork the node has the numbers to do itself.

    Aspect policy mirrors MMH3ImageKeyframe's 'auto', which is the stock node's:
    the frame-0 opener STRETCHES because it sets the clip's geometry, every later
    anchor CENTRE CROPS because it follows one already established. Identical
    results when the aspect already matches, which is the normal case.
    """
    src_h, src_w = int(px.shape[1]), int(px.shape[2])
    if (src_h, src_w) == (int(tgt_h), int(tgt_w)):
        return px, None
    crop = "disabled" if is_opener else "center"
    out = comfy.utils.common_upscale(
        px[..., :3].movedim(-1, 1), int(tgt_w), int(tgt_h), "lanczos", crop
    ).movedim(1, -1)
    return out, ("%dx%d -> %dx%d (%s)"
                 % (src_w, src_h, int(tgt_w), int(tgt_h),
                    "stretch" if is_opener else "centre crop"))


def _owner(spans, overlap_frames, g):
    """Which chunk RENDERS global frame g, and its local frame there.

    Chunks overlap, and a frame inside a chunk's overlap is carried from the
    previous chunk rather than generated, so anchoring a keyframe there paints
    a frame that is conditioned on, not drawn. Assign it to the LAST chunk whose
    NEW content covers it; chunk 0 has no carry so it owns its whole span.
    """
    owner = None
    for i, (a, b) in enumerate(spans):
        if not a <= g <= b:
            continue
        if i == 0 or (g - a) >= overlap_frames:
            owner = (i, g - a)
    if owner is None:
        for i, (a, b) in enumerate(spans):
            if a <= g <= b:
                return (i, g - a)
    return owner


class MMH3LoopingSampler(io.ComfyNode):
    """Fill one clip chunk by chunk, carrying each chunk's tail into the next."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3LoopingSampler",
            display_name="MiniMax H3 Looping Sampler",
            category="MMH3Tools/sampling",
            description=(
                "Fill a whole clip chunk by chunk in ONE node execution. The latent "
                "is the finished length; the chunk count is derived from it, so a "
                "127s track needs no arithmetic on your part.\n\n"
                "The schedule is the SAME one MMH3 Window Plan and MMH3 Split Audio "
                "to Windows compute, so chunk N renders the audio that window N's "
                "prompt was written against. Prompts must all exist before sampling "
                "starts -- wire a cond_set from MiniMax H3 Reference (Multi-Prompt)."
            ),
            inputs=[
                io.Noise.Input("noise"),
                io.Guider.Input(
                    "guider",
                    tooltip="Supplies the MODEL, the cfg, and the negative if it is a "
                            "CFG guider. Its POSITIVE is replaced every chunk from the "
                            "cond_set, so whatever is wired there is ignored -- wire "
                            "MMH3 Cond Select at index 0 so the graph is valid and says "
                            "what it means. A Basic Guider works too; it has no negative."),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                MMH3CondSet.Input("cond_set"),
                io.Latent.Input(
                    "latent",
                    tooltip="The WHOLE clip's AV latent -- the finished length, not one "
                            "chunk. Chunks are slices of it and are written back in "
                            "place, so the output is exactly this length.\n\n"
                            "If its audio half holds a real track (MMH3 Reference "
                            "(Multi-Prompt)'s use_input_audio), each chunk gets ITS OWN "
                            "span of it automatically."),
                io.Int.Input(
                    "chunk_frames", default=192, min=0, max=3600, step=17,
                    tooltip="New content per chunk, in frames at 24 fps. Snapped to the "
                            "5j+2 latent grid. The chunk COUNT falls out of this and the "
                            "clip length -- wire the same value MMH3 Window Plan gets "
                            "and the two schedules are identical.\n\n"
                            "0 = ONE CHUNK covering everything to be generated, sized "
                            "here: the region plus the carry plus any grid padding. One "
                            "chunk means one prompt, whatever the length works out to."),
                io.Int.Input(
                    "overlap_frames", default=22, min=0, max=3600, step=17,
                    tooltip="Frames each chunk carries from the previous one. Snapped so "
                            "the stride keeps every chunk on latent phase 0; an overlap "
                            "that is a multiple of 5 rather than 5m+2 walks the phase "
                            "0,2,4,1,3, which is a five-chunk beat."),
                io.Combo.Input(
                    "carry", options=["mask", "keyframe"], default="mask",
                    tooltip="HOW the previous chunk reaches the next.\n\n"
                            "'mask' masks the overlap latents, which already hold the "
                            "previous chunk's output, so the model conditions on them "
                            "without denoising them. Needs #15375.\n\n"
                            "'keyframe' passes them as a GUIDE anchored at frame 0 -- "
                            "re-injected every step, never denoised, carrying the audio "
                            "at the same coordinate. Needs #15439."),
                io.Float.Input(
                    "overlap_strength_video", default=1.0, min=0.0, max=1.0, step=0.05,
                    tooltip="1.0 preserves the carried region outright, 0.0 regenerates "
                            "it. `mask` carry only."),
                io.Float.Input(
                    "overlap_strength_audio", default=0.9, min=0.0, max=1.0, step=0.05,
                    tooltip="Noise mask for the carried AUDIO latents: mask = 1 - "
                            "strength. 1.0 pins them to the previous chunk's audio, 0.0 "
                            "regenerates them. Independent of the video strength. `mask` "
                            "carry only."),
                io.Int.Input(
                    "sampling_start_step", default=0, min=0, max=1000,
                    tooltip="Begin at this step, skipping the ones before it -- the "
                            "incoming latent is re-noised to that sigma and finished from "
                            "there. Same slice as core SplitSigmas' second output "
                            "(sigmas[step:]). Use it to continue a partial pass: set "
                            "sampling_end_step N on pass 1 and sampling_start_step N on "
                            "pass 2. 0 = start at the beginning. Step numbers are "
                            "ABSOLUTE indices into the incoming schedule.\n\n"
                            "Applies WITHIN every chunk, not across chunks."),
                io.Int.Input(
                    "sampling_end_step", default=1000, min=0, max=1000,
                    tooltip="Stop after this step; later ones are DISCARDED, leaving a "
                            "partially denoised latent. Same slice as core SplitSigmas' "
                            "first output (sigmas[:step+1]). Pairs with "
                            "sampling_start_step to window the schedule without wiring "
                            "SplitSigmas. Not a guide control -- a keyframe guide is "
                            "structural for the whole chunk and cannot be windowed. Step "
                            "numbers are ABSOLUTE."),
                io.Int.Input(
                    "phase2_start_step", default=0, min=0, max=1000,
                    tooltip="Schedule step where phase 2 takes over (0 = disabled). E.g. "
                            "4 on a 12-step schedule: steps 0-3 use the main "
                            "sampler/guider, steps 4+ use the phase-2 pair. Rebased onto "
                            "the window sampling_start_step leaves, so it stays an "
                            "ABSOLUTE index like the other two."),
                io.Sampler.Input(
                    "phase2_sampler", optional=True,
                    tooltip="Second-phase sampler for dual-solver schedules (e.g. a heavy "
                            "solver for the first steps, euler for the rest). Takes over "
                            "at phase2_start_step within EVERY chunk's schedule, "
                            "resample-style continuation."),
                io.Guider.Input(
                    "phase2_guider", optional=True,
                    tooltip="Guider for the second phase (e.g. cfg 1.0 while phase 1 runs "
                            "cfg 2.0). Like the main guider its POSITIVE is replaced every "
                            "chunk from the cond_set -- only its guidance settings apply. "
                            "Falls back to the main guider if unconnected."),
                io.Image.Input(
                    "keyframes", optional=True,
                    tooltip="A BATCH of stills to pin, one per index in keyframe_indices. "
                            "Encoded once here. Independent of `carry`. Resized to the "
                            "generation's resolution first -- keyframe rows share the "
                            "target grid, so a still at another size cannot be used as-is. "
                            "The frame-0 opener is stretched, later anchors are centre "
                            "cropped; every resize is named in the log and the report.\n\n"
                            "ZIPPED with `keyframe_indices`: image i goes at index i. "
                            "A count mismatch is not fatal -- the shorter list wins and "
                            "what was dropped is named, the same way extra prompts are "
                            "reported as unused."),
                io.String.Input(
                    "keyframe_indices", multiline=False, default="",
                    tooltip="Comma-separated frame indices into the WHOLE clip -- place a "
                            "shot where it belongs and this works out which chunk renders "
                            "it. Negatives count from the end. An index inside a chunk's "
                            "carried overlap goes to the chunk that actually draws it. "
                            "Ignored entirely when no `keyframes` are attached, so one "
                            "graph can be reused across passes that do and do not anchor.\n\n"
                            "Fewer or more of these than there are keyframe images is a "
                            "WARNING, not an error -- the extras are dropped and said so. "
                            "An index OUTSIDE the clip is still an error, because that is "
                            "a typo rather than a count."),
                io.Vae.Input(
                    "vae", optional=True,
                    tooltip="The H3 VIDEO vae, needed only to encode `keyframes`."),
                io.Mask.Input(
                    "denoise_mask", optional=True,
                    tooltip="Denoise mask for the VIDEO half over the WHOLE clip: WHITE "
                            "regenerates, BLACK keeps the input latent's content. The "
                            "AUDIO half is NOT affected -- mask it with "
                            "`audio_denoise_mask`.\n\n"
                            "Reduced onto the latent grid with the VAE's real geometry "
                            "and snapped to the DiT's 2x2 patch, so 32 pixels is the "
                            "finest feature it can express.\n\n"
                            "Merged keep-wins (elementwise min) with the mask the "
                            "latent already carried and with the overlap carry, then "
                            "sliced per chunk. Kept regions reproduce the input latent, "
                            "so an empty latent pins BLACK -- this is a v2v tool."),
                io.Combo.Input(
                    "denoise_mask_mode", options=MASK_MODES, default="max",
                    optional=True,
                    tooltip="How BOTH masks are reduced onto the latent grid, "
                            "spatially and across each latent's frame group. 'max' is "
                            "the union and the safe default; 'min' is the intersection; "
                            "'mean' gives fractional edges; 'last' takes each group's "
                            "final frame. `audio_denoise_mask` is pooled to 1x1, so "
                            "only the temporal grouping reaches it."),
                io.Mask.Input(
                    "audio_denoise_mask", optional=True,
                    tooltip="Denoise mask for the AUDIO half: WHITE regenerates, BLACK "
                            "keeps the input latent's audio. Only its TIME axis is "
                            "read -- each frame reduces to one value, mapped onto the "
                            "audio grid through the same boundary conversion the chunk "
                            "loop uses, so a span frozen here lines up with the "
                            "picture.\n\n"
                            "INDEPENDENT of `denoise_mask`: freeing video does not free "
                            "audio, or the reverse. Left unconnected, audio is masked "
                            "only by whatever the input latent already carried -- see "
                            "`preserve_masks` on MiniMax H3 Split AV, which is what "
                            "lets a `use_input_audio` pin survive a split and repack."),
                # LOCAL PATCH 2026-08-31 (pin_renorm, user-requested): appended LAST so
                # saved graphs keep their widget order.
                io.Float.Input(
                    "pin_renorm_video", default=0.0, min=0.0, max=1.0, step=0.05,
                    optional=True,
                    tooltip="Anti-drift carry renormalization ('pin_renorm'). Before each "
                            "chunk samples, the carried VIDEO head's per-channel latent "
                            "mean/std are matched to chunk 1's window (0 = off, 1 = full "
                            "renorm, between = blend). Applies to the CONTEXT the model "
                            "sees only: after sampling, the carried span is restored "
                            "byte-original in the master, so already-rendered frames are "
                            "never altered -- only the NEW rows inherit the anchored "
                            "statistics. `mask` carry only; audio untouched. The report "
                            "prints each head's sigma-ratio vs chunk 1: a climbing ratio "
                            "IS the drift this counters."),
                # LOCAL PATCH 2026-09-03 (band-aware pin_renorm, user-requested):
                # appended after pin_renorm_video, still ahead of the hidden
                # prior_av_latent socket, so saved graphs keep their widget order.
                io.Combo.Input(
                    "pin_renorm_mode", options=["global", "per_band"],
                    default="global", optional=True,
                    tooltip="'global' (original): one per-channel mean/std match over "
                            "the whole carried head. 'per_band': the head is split "
                            "into three spatial frequency bands (high <~2 latent px, "
                            "mid ~2-6, low = residual) and each band's per-channel "
                            "sigma is matched to chunk 1's band independently. Global "
                            "renorm caps total variance but is blind to energy "
                            "REDISTRIBUTING between bands -- measured 2026-09-03 as "
                            "mid-band mottle (skin blotch) doubling over ~10 chunks "
                            "while global sigma and high-frequency energy stayed flat. "
                            "Per-band pins the spectral profile itself. The report "
                            "prints per-band sigma-ratios (hi/mid/lo)."),
                # LOCAL PATCH 2026-09-07 (anti-drift research, user-approved plan): three
                # widgets appended after pin_renorm_mode so saved graphs keep their widget
                # order. All defaults = today's behaviour.
                io.Float.Input(
                    "carry_precomp", default=0.0, min=0.0, max=1.0, step=0.05, optional=True,
                    tooltip="Per-hop bias pre-compensation (anti-drift, context-only). Root cause "
                            "measured 2026-09-07: H3 adds a fixed-direction bias to the carried "
                            "context's latent channel means every hop and the loop integrates it. "
                            "With `reference_latent` wired (the SOURCE video's VAE latent aligned to "
                            "this master, e.g. the v2v source master), the carried head's per-channel "
                            "means are shifted toward the offset-from-reference chunk 0 had, by at "
                            "most `carry_precomp_cap` per channel per hop, BEFORE the chunk samples; "
                            "the master gets the original bytes back afterwards, so rendered frames "
                            "never change and the seam only ever carries one hop's bias. 0 = off, "
                            "1 = full correction (capped)."),
                io.Float.Input(
                    "carry_precomp_cap", default=0.05, min=0.0, max=1.0, step=0.01, optional=True,
                    tooltip="Maximum per-channel mean shift applied per hop by carry_precomp "
                            "(normalized-latent units; the measured per-hop drift is 0.02-0.05). "
                            "Keeps every correction seam-safe."),
                io.Float.Input(
                    "carry_noise_level", default=0.0, min=0.0, max=1.0, step=0.05, optional=True,
                    tooltip="Sigma-scheduled context noise (anti-drift). While sigma >= "
                            "`carry_noise_until_sigma` the carried rows are shown to the model at "
                            "this fractional mask value (row noise level = value*sigma, labels "
                            "consistent -- the schedule is split into segments so H3's per-row "
                            "timesteps match), so the exposure/tone decisions of the first steps "
                            "cannot copy the context's fine statistics; the carried rows are then "
                            "restored byte-original and pinned exactly for the remaining steps. "
                            "Inpaint/denoise masks are untouched (only the carried head rows are "
                            "affected). 0 = off."),
                io.Float.Input(
                    "carry_noise_until_sigma", default=0.96, min=0.0, max=1.0, step=0.01,
                    optional=True,
                    tooltip="carry_noise applies to the steps that START at a sigma >= this value "
                            "(the split lands on the schedule's own sigmas). 0.96 = the first 3 of "
                            "the 8 PDD steps (1.0, 0.988, 0.973); 0.95 = the first 4."),
                io.Latent.Input(
                    "reference_latent", optional=True,
                    tooltip="For carry_precomp: the SOURCE video's H3 VAE latent (video, audio "
                            "optional) aligned row-for-row with `latent` -- the v2v source master "
                            "(H3ContinueMaster source_master / MMH3PackAV of the source encode) or "
                            "a saved encode of the same footage."),
                io.Int.Input(
                    "reroll_retries", default=0, min=0, max=4, optional=True,
                    tooltip="In-loop exposure-snap guard. After each carried chunk samples, its "
                            "fresh rows are compared with the previous chunk's fresh rows (latent "
                            "luma proxy, spatial high-frequency share, per-channel means). If any "
                            "step exceeds `reroll_thresholds`, the chunk is re-sampled with a "
                            "different seed (base+index+1000*attempt), up to this many retries, "
                            "keeping the attempt with the smallest step. 0 = off (attempt 0 is "
                            "bit-identical to the unguarded render). Each retry costs one chunk."),
                io.String.Input(
                    "reroll_thresholds", default="luma=0.14,hf=3.0,cmean=0.25", optional=True,
                    tooltip="Step thresholds for the re-roll guard (units of "
                            "scripts/h3_latent_drift_probe.py's STEP columns): luma = |delta| of "
                            "the latent->RGB luma proxy (normal 0.02-0.08, measured exposure snap "
                            "0.17-0.20); hf = |delta| of the spatial HF share in points (normal "
                            "<= 2); cmean = mean |delta| of the 24 channel means (normal <= 0.2, "
                            "snap 0.33). Those numbers are for 10 s chunks / 39 f overlap; with "
                            "3 s chunks (53 % overlap) normal hops move much more -- calibrate per "
                            "regime from scripts/h3_latent_drift_probe.py's STEP columns (3 s: "
                            "luma=0.25,hf=10,cmean=0.45). A metric can be disabled with a huge value. "
                            "With `reference_latent` wired, two optional keys score the CUMULATIVE "
                            "deviation from chunk 0's relation to the source (a smooth walk never trips "
                            "the step score): luma_abs=0.15,cmean_abs=0.35 -- the guard then keeps the "
                            "attempt closest to target (best-of-N toward the source)."),
                # HIDDEN 2026-08-21. The prior-continuation path never behaved
                # correctly in practice, so the socket is withheld from the schema
                # rather than deleted: `execute` still accepts `prior_av_latent=None`
                # and the prepend/pad/phase implementation below is untouched.
                # Restore by uncommenting -- it was the LAST input, so putting it
                # back appends and cannot disturb widget order in saved graphs.
                # io.Latent.Input(
                # "prior_av_latent", optional=True,
                # tooltip="An already-rendered AV latent to continue from. It is copied "
                # "to the output verbatim and never sampled; `latent` then "
                # "describes only the NEW region, and the output is prior + new.\n\n"
                # "The schedule is planned over the COMBINED length, and every "
                # "window lying inside the prior is skipped. The first generated "
                # "chunk therefore overlaps the prior's tail and carries it like "
                # "any earlier chunk's tail, so the prior's length does not have "
                # "to line up with anything.\n\n"
                # "Prompts map to the GENERATED chunks: cond 0 is the first chunk "
                # "actually sampled, not the first window of the combined clip."),
                io.Int.Input(
                    "feather_latents", default=0, min=0, max=32, optional=True,
                    tooltip="`mask` carry only, and VIDEO only. A linear ramp on the mask "
                            "over N latents AFTER the carried region, easing back to full "
                            "generation instead of stepping at the seam. 0 disables.\n\n"
                            "Restored 2026-09-12 for re-test only; keep at 0 for renders. "
                            "Removed in 0.73.0 because a non-zero feather made the seam "
                            "NOISY (observed 2026-08-13). That observation stands, but "
                            "0.76.0 retracted the mechanism recorded for it, so the cause "
                            "is unknown. Each ramped row gets its own timestep "
                            "(rows_t = 1 - m*sigma); on the current core those survive at "
                            "full latent-frame resolution, unpooled and unbinarised."),
                # FORK 2026-10-23: context frames and a second stage per chunk (see
                # two_stage.py). Appended after every existing input so saved graphs keep
                # their widget values; every default reproduces the sampler as it was.
                io.Combo.Input(
                    "overlap_pin", options=TS.OVERLAP_PINS, default="full", optional=True,
                    tooltip="`mask` carry only. full = every carried row is pinned (as "
                            "before). tail_custom = only the last `context_frames` of the "
                            "carry stay pinned; the earlier carried rows are redrawn as this "
                            "chunk's own lead-in and then discarded -- the write-back starts at "
                            "the previous chunk's end, so frames already rendered never change. "
                            "The audio carry is unchanged."),
                io.Int.Input(
                    "context_frames", default=5, min=0, max=3600, optional=True,
                    tooltip="tail_custom: frames of the carry kept pinned, rounded UP to whole "
                            "latent rows (each 17 frames are rows of 1-4-4-4-4); with a 39-frame "
                            "overlap 5, 9, 13, 17, 22 and 39 are exact. 0 releases every carried "
                            "video row. At or above the overlap it equals full."),
                io.Int.Input(
                    "accepted_prefix_frames", default=0, min=0, max=100000, optional=True,
                    tooltip="Frames at the start of the master that are already final (a "
                            "resume's accepted prefix). Chunks inside it are skipped, its rows "
                            "are never written back, and tail_custom releases carried rows "
                            "inside it fully rather than to the master's resume pin."),
                io.Guider.Input(
                    "stage2_guider", optional=True,
                    tooltip="Stage 2's guider: its MODEL, CFG and negative. Its positive is "
                            "replaced by each chunk's prompt, like the main guider's. Unwired = "
                            "the main guider. Patches stage 2 needs (LoRA, PDD, SigmaShift, "
                            "control schedules, attention) must be on THIS model."),
                io.Sampler.Input(
                    "stage2_sampler", optional=True,
                    tooltip="Stage 2's sampler. Unwired = the main sampler."),
                io.Sigmas.Input(
                    "stage2_sigmas", optional=True,
                    tooltip="Wiring this turns stage 2 on: every chunk finishes with a second "
                            "pass over these sigmas (strictly descending, <= 1, ending at 0). "
                            "Stage 1's DENOISED estimate is re-noised at the first sigma, so any "
                            "schedule works -- e.g. stage 1 = the PDD sigmas split at step 7, "
                            "stage 2 = ManualSigmas 0.6316, 0.3158, 0. The main `sigmas` then "
                            "normally stop above 0."),
                io.Combo.Input(
                    "upscale_model", options=TS.upscale_choices(), default=TS.UPSCALE_NONE,
                    optional=True,
                    tooltip="Learned H3 latent upscaler (models/latent_upscale_models) run on "
                            "each chunk between the stages; needs stage 2. Upscaled chunks get "
                            "fresh video noise on the new grid; audio is never resized. none = "
                            "stage 2 on the same grid."),
                io.Float.Input(
                    "upscale_megapixels", default=1.0, min=0.05, max=16.0, step=0.05,
                    optional=True,
                    tooltip="Size of the stage-2 grid: 1024^2 pixels per MP, sides snapped to "
                            "32 px, aspect kept. Must be larger than the master's grid."),
                io.Combo.Input(
                    "upscale_precision", options=TS.UPSCALE_PRECISIONS, default="fp32",
                    optional=True,
                    tooltip="Upscaler compute dtype. fp32 measured fastest and best on the "
                            "H200; avoid bf16."),
                io.Boolean.Input(
                    "upscale_temporal_chunking", default=False, optional=True,
                    tooltip="Upscale in 32-latent temporal chunks with cross-fades (less VRAM)."),
                io.Boolean.Input(
                    "keep_upscaler_loaded", default=True, optional=True,
                    tooltip="Keep the upscaler on the GPU between chunks."),
                io.Combo.Input(
                    "stage1_carry", options=TS.STAGE1_CARRY, default="estimate", optional=True,
                    tooltip="With an upscaler: what stage 1 pins on its own grid. estimate = the "
                            "previous chunk's stage-1 estimate (no round trip). downscaled_final "
                            "= its finished stage-2 rows, area-downscaled."),
                io.Latent.Input(
                    "stage2_latent", optional=True,
                    tooltip="With an upscaler: the master on the stage-2 grid. Required when a "
                            "chunk carries rows this run does not render (resume, accepted "
                            "frames): the previous finished chunks come from here. Rows the "
                            "master pins (v2v keep regions) take its exact high-res source; "
                            "unwired, they are upscaled from stage 1. Its audio is not used."),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.Int.Output(display_name="chunks_rendered"),
                io.String.Output(display_name="report"),
                io.Latent.Output(
                    display_name="stage1_latent",
                    tooltip="The master on the stage-1 grid: each chunk's stage-1 estimate. "
                            "Equal to `latent` without an upscaler. Save it to resume a "
                            "two-grid run with stage1_carry = estimate."),
            ],
        )

    @classmethod
    def execute(cls, noise, guider, sampler, sigmas, cond_set, latent, chunk_frames,
                overlap_frames, carry, overlap_strength_video, overlap_strength_audio,
                sampling_start_step=0, sampling_end_step=1000,
                phase2_start_step=0, phase2_sampler=None, phase2_guider=None,
                keyframes=None, keyframe_indices="", vae=None,
                prior_av_latent=None, denoise_mask=None, denoise_mask_mode="max",
                audio_denoise_mask=None, pin_renorm_video=0.0,
                pin_renorm_mode="global", carry_precomp=0.0, carry_precomp_cap=0.05,
                carry_noise_level=0.0, carry_noise_until_sigma=0.96, reference_latent=None,
                reroll_retries=0,
                reroll_thresholds="luma=0.14,hf=3.0,cmean=0.25",
                feather_latents=0, overlap_pin="full", context_frames=5,
                accepted_prefix_frames=0, stage2_guider=None, stage2_sampler=None,
                stage2_sigmas=None, upscale_model=TS.UPSCALE_NONE, upscale_megapixels=1.0,
                upscale_precision="fp32", upscale_temporal_chunking=False,
                keep_upscaler_loaded=True, stage1_carry="estimate",
                stage2_latent=None) -> io.NodeOutput:
        conds = (cond_set or {}).get("conds") or []
        if not conds:
            raise ValueError("MMH3LoopingSampler: cond_set holds no conditioning.")

        if carry == "keyframe" and not _guides_available():
            raise RuntimeError(
                "MMH3LoopingSampler: carry='keyframe' needs any-index guides (upstream "
                "PR #15439), which is not applied. See docs/core-changes.md. Use "
                "carry='mask', which needs only #15375.")

        master_v, master_a = unpack_av(latent, "latent")

        # A prior is PREPENDED and the schedule planned over the combined clip, so the
        # prior needs no relationship to the window size: whichever window first
        # reaches past it becomes chunk 0, already overlapping it, and the ordinary
        # carry rule takes the prior's tail from there.
        prior_t = prior_at = 0
        if prior_av_latent is not None:
            prior_v, prior_a = unpack_av(prior_av_latent, "prior_av_latent")
            if prior_v.shape[1] != master_v.shape[1] or \
                    tuple(prior_v.shape[3:]) != tuple(master_v.shape[3:]):
                raise ValueError(
                    "MMH3LoopingSampler: prior_av_latent is %s but the target latent is "
                    "%s. They are concatenated on the time axis, so channels and frame "
                    "size must match."
                    % (tuple(prior_v.shape), tuple(master_v.shape)))
            if (prior_a is None) != (master_a is None):
                raise ValueError(
                    "MMH3LoopingSampler: one of prior_av_latent / latent carries audio "
                    "and the other does not. Both must be AV, or both video-only.")
            prior_t = int(prior_v.shape[VIDEO_T_DIM])
            keep_v = prior_v.to(master_v.dtype).clone()
            keep_a = None

            # GRID. A standalone clip is 5j+2 latents, so prior + new is 5k+4 -- not a
            # valid clip at all, and latents_to_frames() floors it, leaving the tail of
            # the new region outside every window and never sampled. Pad the NEW region
            # up instead: the prior is real footage that must not be invented or
            # discarded, and (prior_t + new_t) must be 2 mod 5, so the new side has to
            # be 0 mod 5. Costs at most 4 latents of extra generation.
            #
            # It also settles the phase: offset = prior_t - overlap = (5a+2) - (5m+2),
            # a multiple of 5, so every window still starts on phase 0 -- which H3's
            # FRAME_PER_TOKEN (1,4,4,4,4) indexing requires.
            pad_t = (LATENT_BASE - (prior_t + int(master_v.shape[VIDEO_T_DIM]))) \
                % LATENTS_PER_GROUP
            if pad_t:
                tail = master_v[:, :, -1:].repeat_interleave(pad_t, dim=VIDEO_T_DIM)
                master_v = torch.cat([master_v, torch.zeros_like(tail)], VIDEO_T_DIM)

            master_v = torch.cat([keep_v, master_v], VIDEO_T_DIM)
            if master_a is not None:
                prior_at = int(prior_a.shape[AUDIO_T_DIM])
                # The two axes are concatenated independently, so a prior whose audio
                # does not correspond to its own video silently shifts EVERYTHING after
                # it. This is easy to hit: encoding a loaded .mp4 counts audio from the
                # track's duration, and encoders routinely pad it past the last frame.
                want_at = frames_to_audio_t(latents_to_frames(prior_t))
                if prior_at != want_at:
                    drift = (prior_at - want_at) / float(AUDIO_LATENT_FPS)
                    raise ValueError(
                        "MMH3LoopingSampler: prior_av_latent has %d audio latents but "
                        "its %d video latents (%d frames) need %d -- %+.3fs of drift. "
                        "Everything after the prior would shift by that much. Trim the "
                        "prior's audio to its video (MMH3 Trim AV), or re-encode the "
                        "source so the two match."
                        % (prior_at, prior_t, latents_to_frames(prior_t), want_at, drift))
                keep_a = prior_a.to(master_a.dtype).clone()
                master_a = torch.cat([keep_a, master_a], AUDIO_T_DIM)
                # Audio is 40Hz against 24fps video, so it does not pad by the same
                # count. Size it from the combined VIDEO length, which is now grid
                # valid, rather than adding pad_t audio latents and hoping.
                want = frames_to_audio_t(
                    latents_to_frames(int(master_v.shape[VIDEO_T_DIM])))
                have = int(master_a.shape[AUDIO_T_DIM])
                if have < want:
                    master_a = torch.cat(
                        [master_a, torch.zeros_like(master_a[:, :, :, :1])
                         .repeat_interleave(want - have, dim=AUDIO_T_DIM)], AUDIO_T_DIM)
                elif have > want:
                    master_a = master_a[:, :, :, :want]

        total_t = int(master_v.shape[VIDEO_T_DIM])
        total_a = 0 if master_a is None else int(master_a.shape[AUDIO_T_DIM])
        total_f = latents_to_frames(total_t)

        # THE SAME schedule MMH3WindowPlan and MMH3SplitAudioToWindows compute, so
        # chunk N renders the audio window N's prompt was written against.
        #
        # With a prior, the schedule covers only the GENERATED span -- one carry plus
        # the new region -- and is then offset onto the combined clip. Planning over
        # the combined clip instead made the prior's length shift every window
        # boundary, so whether the new region fell in one chunk or two depended on how
        # the total happened to divide: chunk_frames 124 gave one chunk after a 10s
        # prior and two after a 20s one. Planning the generated span makes the prior's
        # length genuinely irrelevant, which is the whole point of it being arbitrary.
        # chunk_frames 0 asks for a single chunk over everything being generated. The
        # size is not something to work out by hand: it is the region PLUS the carry
        # PLUS whatever the grid padding came to, and getting it wrong by one grid step
        # silently costs a second chunk and therefore a second prompt.
        cf = int(chunk_frames)
        one_chunk = cf <= 0
        probe_cf = 3600 if one_chunk else cf

        plan_over_f = total_f
        if prior_t:
            _l, ov_probe, _pf, _pt, _w = _plan(
                latents_to_frames(total_t - prior_t), probe_cf,
                int(overlap_frames), "standard_static")
            plan_over_f = latents_to_frames(total_t - prior_t + ov_probe)
        if one_chunk:
            cf = plan_over_f

        length, overlap, plan_f, _plan_t, windows = _plan(
            plan_over_f, cf, int(overlap_frames), "standard_static")
        ov_frames = frame_at_latent(overlap)

        # Where the generated span begins in the combined clip: one carry before the
        # new region, so chunk 0 opens on the prior's tail.
        offset = max(0, prior_t - overlap) if prior_t else 0
        spans = [(frame_at_latent(min(w.index_list[0] + offset, total_t - 1)),
                  min(frame_at_latent(w.index_list[-1] + 1 + offset) - 1, total_f - 1))
                 for w in windows]
        n = len(windows)

        # the carry settings belong in the SUMMARY, not just the widgets: without
        # them two runs are indistinguishable in the log, and comparing a render
        # against an earlier one is exactly what diagnosing carry drift needs
        lines = ["%d chunk%s of %d latents (%d frames) over %d frames (%.2fs), "
                 "overlap %d latents (%d frames)"
                 % (n, "" if n == 1 else "s", length, latents_to_frames(length),
                    total_f, total_f / float(FPS), overlap, ov_frames),
                 "carry %s, strength video %.2f / audio %.2f, noise seed %s"
                 % (carry, float(overlap_strength_video), float(overlap_strength_audio),
                    getattr(noise, "seed", "?"))]
        if prior_t:
            lines.append("prior: %d latents (%d frames, %.2fs) kept verbatim; generating "
                         "from frame %d, carrying %d frames of it"
                         % (prior_t, frame_at_latent(prior_t),
                            frame_at_latent(prior_t) / float(FPS), spans[0][0],
                            frame_at_latent(prior_t - offset)))
        if len(conds) != n:
            lines.append("! %d prompt%s for %d chunks -- %s"
                         % (len(conds), "" if len(conds) == 1 else "s", n,
                            "the last repeats" if len(conds) < n
                            else "the extras are unused"))
            logging.warning("[MMH3LoopingSampler] %s", lines[-1][2:])

        # Schedule windowing is per chunk and invisible in the output otherwise --
        # a partially denoised master looks like a bad seed, not a setting.
        n_sig = max(0, len(sigmas) - 1)
        if int(sampling_start_step) > 0 or int(sampling_end_step) < n_sig:
            lines.append("schedule window: steps %d-%d of %d, per chunk"
                         % (int(sampling_start_step),
                            min(int(sampling_end_step), n_sig), n_sig))
            if int(sampling_end_step) < n_sig:
                lines[-1] += " -- output is PARTIALLY denoised"
        if phase2_sampler is not None and int(phase2_start_step) > 0:
            lines.append("phase 2 from step %d%s"
                         % (int(phase2_start_step),
                            "" if phase2_guider is None else ", with its own guider"))

        # ---- FORK 2026-10-23: context frames and stage 2 (two_stage.py) ------
        tail_mode = overlap_pin == "tail_custom"
        stage2_on = stage2_sigmas is not None
        upscale_on = (upscale_model or TS.UPSCALE_NONE) != TS.UPSCALE_NONE
        two_grid = stage2_on and upscale_on
        if upscale_on and not stage2_on:
            raise ValueError(
                "MMH3LoopingSampler: upscale_model runs between the stages, so it needs stage "
                "2 -- wire stage2_sigmas (and normally stage2_guider), or set upscale_model "
                "to none.")
        if stage2_on:
            TS.validate_sigmas(stage2_sigmas, "stage2_sigmas")
        if (tail_mode or stage2_on) and carry == "keyframe":
            raise ValueError(
                "MMH3LoopingSampler: overlap_pin tail_custom and stage 2 work on the masked "
                "carry; set carry to mask.")
        if (tail_mode or stage2_on) and (float(pin_renorm_video or 0) > 0
                                         or float(carry_precomp or 0) > 0
                                         or float(carry_noise_level or 0) > 0):
            raise ValueError(
                "MMH3LoopingSampler: pin_renorm, carry_precomp and carry_noise assume the "
                "whole carried head is pinned on one grid; set them to 0 to use "
                "overlap_pin tail_custom or stage 2.")
        if stage2_on and prior_av_latent is not None:
            raise ValueError("MMH3LoopingSampler: prior_av_latent does not support stage 2.")
        accepted_t = (frames_to_latents(int(accepted_prefix_frames))
                      if int(accepted_prefix_frames or 0) > 0 else 0)
        hi_size = None
        if two_grid:
            hi_size = TS.target_size(master_v, float(upscale_megapixels))
            lo_size = tuple(int(x) for x in master_v.shape[3:])
            if hi_size[0] < lo_size[0] or hi_size[1] < lo_size[1] or tuple(hi_size) == lo_size:
                raise ValueError(
                    "MMH3LoopingSampler: upscale_megapixels %.2f gives a %dx%d grid, not larger "
                    "than the master's %dx%d. Raise it, or set upscale_model to none."
                    % (float(upscale_megapixels), hi_size[1] * VAE_SPATIAL,
                       hi_size[0] * VAE_SPATIAL, lo_size[1] * VAE_SPATIAL,
                       lo_size[0] * VAE_SPATIAL))
        if tail_mode:
            lines.append("context frames: tail_custom, %d frame(s) of each carry pinned "
                         "(rounded up to whole latent rows); earlier carried rows redrawn "
                         "and discarded" % int(context_frames))
        if accepted_t:
            lines.append("accepted prefix: %d frames (%d latents) kept as given"
                         % (int(accepted_prefix_frames), accepted_t))
        if stage2_on:
            lines.append("stage 2: %s, %s guider, %s sampler"
                         % (TS.describe_sigmas(stage2_sigmas),
                            "own" if stage2_guider is not None else "main",
                            "own" if stage2_sampler is not None else "main"))
        if two_grid:
            lines.append("stage 2 grid: %dx%d -> %dx%d via %s (%.2f MP, %s); stage-1 carry = %s"
                         % (master_v.shape[4] * VAE_SPATIAL, master_v.shape[3] * VAE_SPATIAL,
                            hi_size[1] * VAE_SPATIAL, hi_size[0] * VAE_SPATIAL, upscale_model,
                            float(upscale_megapixels), upscale_precision, stage1_carry))

        # ---- keyframes, encoded ONCE, placed on the real timeline -----------
        guides_by_chunk = {}
        guides_hi = {}
        if keyframes is None:
            # Indices with no images attached are INERT, not fatal. A ladder reuses
            # one graph across passes and usually only the first pass carries
            # anchors, so a live keyframe_indices string with the image input
            # unplugged is the normal state of a refine pass. Parsing is skipped
            # entirely rather than parsed-then-discarded: with nothing to place,
            # an out-of-range index is not a mistake worth stopping for either.
            wanted = []
            if (keyframe_indices or "").strip():
                logging.info("[MMH3LoopingSampler] keyframe_indices is set (%r) but no "
                             "keyframes are attached; ignoring it",
                             (keyframe_indices or "").strip())
                lines.append("  keyframe_indices ignored: no keyframes attached")
        else:
            wanted = _parse_indices(keyframe_indices, total_f)
        if keyframes is not None and wanted:
            if vae is None:
                raise ValueError("MMH3LoopingSampler: keyframes need the H3 video vae.")
            # Zip to the shorter and say so, the way extra prompts are handled.
            # This used to raise. The argument for raising was that an unplaced
            # keyframe is a content change rather than an inert spare input, and that
            # is still true -- but it is stated in the log and in the report, and one
            # node answering the same class of mismatch two different ways was worse
            # than either answer. Out-of-range indices remain an ERROR: that is a
            # typo, not a count.
            n_img = int(keyframes.shape[0])
            if n_img != len(wanted):
                keep = min(n_img, len(wanted))
                extra = ("%d keyframe image(s)" % (n_img - keep) if n_img > keep
                         else "%d index/indices" % (len(wanted) - keep))
                logging.warning(
                    "[MMH3LoopingSampler] %d keyframe image(s) against %d "
                    "index/indices -- using the first %d, dropping the last %s. They "
                    "are ZIPPED, so image i is placed at index i.",
                    n_img, len(wanted), keep, extra)
                lines.append("  keyframes: %d image(s), %d index/indices -- kept %d, "
                             "dropped the last %s" % (n_img, len(wanted), keep, extra))
                wanted = wanted[:keep]
                keyframes = keyframes[:keep]
            if not _guides_available():
                raise RuntimeError(
                    "MMH3LoopingSampler: keyframes need any-index guides (PR #15439).")
            # keyframe rows share the TARGET grid, so a still at any other size has
            # to be fitted here or it fails deep in the model -- see _fit_keyframe
            tgt_h = int(master_v.shape[3]) * VAE_SPATIAL
            tgt_w = int(master_v.shape[4]) * VAE_SPATIAL
            for img_i, g in enumerate(wanted):
                ci, local = _owner(spans, ov_frames, g)
                px, note = _fit_keyframe(keyframes[img_i:img_i + 1], tgt_w, tgt_h,
                                         is_opener=(g == 0))
                if note:
                    logging.info("[MMH3LoopingSampler] keyframe frame %d resized %s",
                                 g, note)
                z = vae.encode(px)
                guides_by_chunk.setdefault(ci, []).append(
                    {"resolved_frame_index": int(local), "latent": z})
                if two_grid:
                    # stage 2 samples on the upscaled grid, and guide rows share it
                    px_hi, _n = _fit_keyframe(keyframes[img_i:img_i + 1],
                                              hi_size[1] * VAE_SPATIAL,
                                              hi_size[0] * VAE_SPATIAL, is_opener=(g == 0))
                    guides_hi.setdefault(ci, []).append(
                        {"resolved_frame_index": int(local), "latent": vae.encode(px_hi)})
                lines.append("  keyframe frame %d -> chunk %d local frame %d%s"
                             % (g, ci, local, "" if not note else ", resized " + note))

        # ---- fill the master, chunk by chunk --------------------------------
        out_v = master_v.clone()
        out_a = None if master_a is None else master_a.clone()
        in_mask_v, in_mask_a = _split_mask(latent)

        if in_mask_v is not None or in_mask_a is not None:
            lines.append("  input latent carried a noise_mask (video %s, audio %s)"
                         % ("yes" if in_mask_v is not None else "no",
                            "yes" if in_mask_a is not None else "no"))

        if denoise_mask is not None or audio_denoise_mask is not None:
            if not per_row_mask_is_continuous():
                raise RuntimeError(
                    "MMH3LoopingSampler: the mask inputs need per-row masking (#15375). "
                    "On this core a mask is accepted and IGNORED -- you would get a full "
                    "regeneration with nothing raised. Update ComfyUI, or unwire the "
                    "mask.")

        # The two halves are INDEPENDENT. `denoise_mask` reduces onto the video grid and
        # stops there; audio is masked only by `audio_denoise_mask`, or by whatever the
        # latent already carried.
        #
        # Deriving one from the other reads as a safety feature -- the two can then never
        # disagree about a frozen span -- and is wrong, because a mask only carries
        # temporal intent when it IS temporal. A subject matted out of every frame is
        # white somewhere at every timestep, so the spatial reduction returned "free"
        # everywhere and regenerated a track that `use_input_audio` had pinned.
        if denoise_mask is not None:
            mv = _mask_to_video_latent(
                denoise_mask.to(master_v.device), total_t,
                int(master_v.shape[3]), int(master_v.shape[4]), denoise_mask_mode)
            kept = float((mv < 0.5).float().mean()) * 100.0
            lines.append("  denoise_mask: %.1f%% of the video grid kept (pinned to the "
                         "input latent); the audio half is untouched" % kept)
            if float(master_v.abs().max()) < 1e-6:
                lines.append("  ! the input video latent is all zeros, so kept regions "
                             "pin BLACK -- encode the source into `latent`")
            in_mask_v = mv if in_mask_v is None else torch.minimum(
                in_mask_v.to(mv.dtype), mv)

        if audio_denoise_mask is not None:
            if master_a is None:
                lines.append("  ! audio_denoise_mask is wired but this latent has no "
                             "audio half -- ignored")
            else:
                am = _mask_to_video_latent(
                    audio_denoise_mask.to(master_v.device), total_t, 1, 1,
                    denoise_mask_mode)
                prof = _video_mask_to_audio(am, total_t, total_a, denoise_mask_mode)
                free = float((prof > 0.5).float().mean()) * 100.0
                lines.append("  audio_denoise_mask: %.1f%% of the track free to "
                             "regenerate" % free)
                ma = _audio_profile_to_mask(prof, master_a)
                in_mask_a = ma if in_mask_a is None else torch.minimum(
                    in_mask_a.to(ma.dtype), ma)

        # FORK 2026-10-23: the FINAL-grid master. One grid: the master itself, so every
        # existing path reads and writes exactly what it did. Two grids: stage 2's own
        # master -- stage2_latent's rows if wired, otherwise empty until chunks land --
        # and `out_v` becomes the stage-1 grid's master of estimates.
        fin_v, in_mask2_v, hi_valid = out_v, in_mask_v, None
        if two_grid:
            if stage2_latent is not None:
                s2_v = unpack_av(stage2_latent, "stage2_latent", allow_video_only=True)[0]
                if (int(s2_v.shape[1]) != int(master_v.shape[1])
                        or int(s2_v.shape[VIDEO_T_DIM]) != total_t
                        or tuple(int(x) for x in s2_v.shape[3:]) != tuple(hi_size)):
                    raise ValueError(
                        "MMH3LoopingSampler: stage2_latent is %s but stage 2 needs %d latents "
                        "at %dx%d (the master's length on the upscale_megapixels grid)."
                        % (tuple(s2_v.shape), total_t, hi_size[1] * VAE_SPATIAL,
                           hi_size[0] * VAE_SPATIAL))
                fin_v = s2_v.to(device=master_v.device, dtype=master_v.dtype).clone()
                hi_valid = torch.ones(total_t, dtype=torch.bool)
                lines.append("  stage2_latent: previous chunks and pinned source rows on "
                             "the stage-2 grid come from it")
            else:
                fin_v = torch.zeros(list(master_v.shape[:3]) + list(hi_size),
                                    dtype=master_v.dtype, device=master_v.device)
                hi_valid = torch.zeros(total_t, dtype=torch.bool)
            in_mask2_v = TS.resize_mask(in_mask_v, hi_size)

        # None unless something wired MMH3 Timeline Preview onto the model. The
        # sampler must behave identically when it is None, which is why the preview
        # is DISCOVERED here rather than owned: no wiring, no object, no branch that
        # can change a render.
        preview = begin_preview(getattr(guider, "model_patcher", None), len(windows))

        # LOCAL PATCH 2026-08-31 (pin_renorm, user-requested): per-channel reference
        # stats from window 0, captured lazily at the first carry -- by then window 0
        # holds rendered output (or, on resume, the pinned prior content). Context-only:
        # the renormalized head is what the model conditions on; the master gets the
        # original bytes back after write-back (same philosophy as the prior_t restore
        # below), so the carry ratchet's stats stop compounding into NEW rows without
        # ever editing already-rendered frames.
        renorm_ref = None
        pin_renorm_video = max(0.0, min(1.0, float(pin_renorm_video or 0.0)))
        # LOCAL PATCH 2026-09-07 (re-roll guard + context-only fractional carry)
        _rr_retries = max(0, int(reroll_retries or 0))
        _rr_th = _reroll_parse_thresholds(reroll_thresholds)
        _reroll_prev = None          # previous chunk's fresh-region stats (guard reference)
        if _rr_retries > 0:
            lines.append("  reroll guard: up to %d retries per chunk, thresholds %s"
                         % (_rr_retries, ", ".join("%s=%g" % kv for kv in _rr_th.items())))
        # LOCAL PATCH 2026-09-07 (anti-drift mitigations, user-approved after the root-cause verdict)
        _pc = max(0.0, min(1.0, float(carry_precomp or 0.0)))
        _pc_cap = max(0.0, float(carry_precomp_cap or 0.0))
        _pc_ref0 = None
        _ref_v = None
        if reference_latent is not None and _pc <= 0.0:
            # a reference without pre-compensation still serves the guard's drift scoring
            _ref_v = unpack_av(reference_latent, "reference_latent", allow_video_only=True)[0]
            if int(_ref_v.shape[2]) < total_t:
                raise ValueError(
                    "MMH3LoopingSampler: reference_latent has %d latents but the master has %d."
                    % (int(_ref_v.shape[2]), total_t))
        if _pc > 0.0 and reference_latent is None:
            # production flows gate the source encode (V2V INPAINT off -> None): skip, don't fail
            lines.append("  carry_precomp %.2f requested but no reference_latent is wired (source "
                         "encode gated off?) -- pre-compensation SKIPPED this run" % _pc)
            logging.warning("[MMH3LoopingSampler] %s", lines[-1].strip())
            _pc = 0.0
        if _pc > 0.0:
            _ref_v = unpack_av(reference_latent, "reference_latent", allow_video_only=True)[0]
            if int(_ref_v.shape[2]) < total_t:
                raise ValueError(
                    "MMH3LoopingSampler: reference_latent has %d latents but the master has %d -- "
                    "wire a reference covering the whole master (same alignment)."
                    % (int(_ref_v.shape[2]), total_t))
            lines.append("  carry_precomp %.2f (cap %.3f per channel per hop): carried head's "
                         "channel means held at chunk 0's offset from reference_latent, "
                         "context-only (bytes restored)" % (_pc, _pc_cap))
        if two_grid and _ref_v is not None and \
                tuple(int(x) for x in _ref_v.shape[3:]) != tuple(hi_size):
            raise ValueError(
                "MMH3LoopingSampler: reference_latent is %dx%d but the re-roll guard scores "
                "chunks on the stage-2 grid (%dx%d); encode the reference at that size."
                % (_ref_v.shape[4] * VAE_SPATIAL, _ref_v.shape[3] * VAE_SPATIAL,
                   hi_size[1] * VAE_SPATIAL, hi_size[0] * VAE_SPATIAL))
        _cn_level = max(0.0, min(1.0, float(carry_noise_level or 0.0)))
        _cn_sigma = float(carry_noise_until_sigma or 0.0)
        if _cn_level > 0.0:
            lines.append("  carry_noise: carried rows shown at mask %.2f while sigma >= %.3f, "
                         "then restored and pinned exactly" % (_cn_level, _cn_sigma))

        for i, w in enumerate(windows):
            idx = w.index_list
            v0, v1 = idx[0] + offset, min(idx[-1] + 1 + offset, total_t)
            a0 = _audio_index_at(v0, total_t, total_a)
            a1 = _audio_index_at(v1, total_t, total_a)

            # Carry whatever the PREVIOUS window actually reached, not the nominal
            # overlap. Core clamps the LAST window back so it ends on the clip end,
            # which makes it physically overlap its predecessor by far more than
            # `overlap` -- 62 latents against a nominal 7 on a 127s clip in 20s
            # chunks. Preserving only the nominal 7 left the other 55 to be
            # regenerated under THIS chunk's prompt and written over content the
            # previous chunk had already drawn: up to 12s of the second-to-last
            # section silently taking the last section's conditioning. Middle
            # windows are unaffected -- there actual == nominal.
            #
            # With a prior, chunk 0 is an EXTEND chunk: the prior ends at prior_t, so
            # that is its predecessor's end and the same rule takes the prior's tail.
            prev_end = (windows[i - 1].index_list[-1] + 1 + offset) if i else prior_t
            carried = min(max(0, prev_end - v0), v1 - v0) if (i or prior_t) else 0
            if carried > overlap:
                lines.append("  chunk %d: clamped tail -- carries %d latents (%d "
                             "frames) instead of %d, so the previous chunk's content "
                             "survives" % (i, carried, frame_at_latent(carried),
                                           overlap))
                logging.info("[MMH3LoopingSampler] %s", lines[-1].strip())

            if accepted_t and v1 <= accepted_t:
                lines.append("  chunk %d: inside the accepted prefix -- SKIPPED" % i)
                logging.info("[MMH3LoopingSampler] chunk %d/%d accepted -- skipped", i + 1, n)
                continue
            carried_a = (_audio_index_at(v0 + carried, total_t, total_a) - a0) if carried else 0
            # tail_custom: carried rows [0, free) are released; [free, carried) stay pinned
            free = (carried - TS.tail_context_rows(v0, v0 + carried, int(context_frames))
                    if (tail_mode and carried) else 0)

            sub_v = out_v[:, :, v0:v1].clone()
            sub_a = None if out_a is None else out_a[:, :, :, a0:a1].clone()
            chunk = pack_av(latent, sub_v, sub_a, noise_mask=None)

            chunk_cond = _strip_guide_keys(conds[min(i, len(conds) - 1)],
                                           "chunk %d prompt" % i)
            chunk_guides = list(guides_by_chunk.get(i, []))

            if carried and carry == "keyframe":
                kf = {"resolved_frame_index": 0,
                      "latent": sub_v[:, :, :carried].contiguous()}
                if sub_a is not None:
                    ca = _audio_index_at(v0 + carried, total_t, total_a) - a0
                    if ca > 0:
                        kf["audio_latent"] = sub_a[:, :, :, :ca].contiguous()
                chunk_guides.insert(0, kf)
            elif carried:
                chunk["noise_mask"] = _carry_mask(
                    sub_v, sub_a, carried,
                    _audio_index_at(v0 + carried, total_t, total_a) - a0,
                    float(overlap_strength_video), float(overlap_strength_audio),
                    in_mask_v, in_mask_a, v0, v1, a0, a1, int(feather_latents))
                if free:
                    chunk["noise_mask"] = TS.release_head(
                        chunk["noise_mask"], free,
                        TS.head_floor(in_mask_v, sub_v, v0, free, accepted_t))
            elif in_mask_v is not None or in_mask_a is not None:
                chunk["noise_mask"] = _sliced_mask(sub_v, sub_a, in_mask_v, in_mask_a,
                                                   v0, v1, a0, a1)

            # A window whose mask keeps EVERY row (video and audio) is a no-op by
            # construction: all rows run at the cond timestep and the write-back
            # reproduces the input (verified 2026-08-24 at codec-noise MAE). That is
            # exactly a resume run's pinned window (H3ContinueMaster head) or an
            # inpaint chunk the mask never touches -- skip the sampling instead of
            # paying a full chunk for it. Per-chunk noise is seed+index, so skipping
            # cannot shift any other chunk's noise.
            _nm = chunk.get("noise_mask")
            if _nm is not None:
                _parts = list(_nm.unbind()) if isinstance(_nm, NestedTensor) else [_nm]
                if all(float(p.max()) <= 1e-6 for p in _parts):
                    lines.append("  chunk %d: fully pinned -- SKIPPED (no rows to "
                                 "denoise; content carried as-is)" % i)
                    logging.info("[MMH3LoopingSampler] chunk %d/%d fully pinned -- "
                                 "skipped", i + 1, n)
                    continue

            if two_grid and carried:
                # Stage 2 pins the previous chunk's FINISHED rows on its own grid. Rows this
                # run never rendered (a resume's pinned window, accepted frames) can only
                # come from stage2_latent; upscaling a stage-1 estimate in their place would
                # silently swap finished frames for a guess.
                gap = (~hi_valid[v0 + free:v0 + carried]).nonzero()
                if gap.numel():
                    raise ValueError(
                        "MMH3LoopingSampler: chunk %d carries latents %d-%d, which this run "
                        "did not render, and nothing holds them on the stage-2 grid. Wire the "
                        "previous run's final master (stage-2 grid) to stage2_latent."
                        % (i, v0 + free + int(gap[0]), v0 + carried - 1))
                if stage1_carry == "downscaled_final":
                    # chunk["samples"] wraps sub_v, so this reaches stage 1
                    sub_v[:, :, :carried] = TS.downscale_video(
                        fin_v[:, :, v0:v0 + carried], tuple(sub_v.shape[3:])).to(sub_v.dtype)

            # LOCAL PATCH 2026-08-31 (pin_renorm): after the skip check so skipped
            # windows stay untouched; chunk["samples"] aliases sub_v (pack_av wraps
            # references), so the in-place edit reaches the sampler. Mask carry only --
            # keyframe carry took its anchor tensor before this point and measured
            # drift-free anyway.
            _pr_orig_head = None
            if pin_renorm_video > 0.0 and carried and carry != "keyframe":
                per_band = pin_renorm_mode == "per_band"
                if renorm_ref is None:
                    w0 = windows[0].index_list
                    r0 = w0[0] + offset
                    r1 = min(w0[-1] + 1 + offset, total_t)
                    ref_span = out_v[:, :, r0:r1]
                    renorm_ref = (_pin_renorm_band_stats(ref_span) if per_band
                                  else _pin_renorm_stats(ref_span))
                    lines.append("  pin_renorm: reference = window 0 (latents %d-%d), "
                                 "strength %.2f, mode %s, context-only"
                                 % (r0, r1, pin_renorm_video, pin_renorm_mode))
                _pr_orig_head = out_v[:, :, v0:v0 + carried].clone()
                if per_band:
                    ratios = _pin_renorm_head_band(sub_v, carried, renorm_ref,
                                                   pin_renorm_video)
                    lines.append("  chunk %d: pin_renorm band sigma-ratios "
                                 "hi %.4f / mid %.4f / lo %.4f vs chunk 1 "
                                 "(>1 = more energy than the anchor)"
                                 % (i, ratios[0], ratios[1], ratios[2]))
                else:
                    ratio = _pin_renorm_head(sub_v, carried, renorm_ref,
                                             pin_renorm_video)
                    lines.append("  chunk %d: pin_renorm head sigma-ratio %.4f vs "
                                 "chunk 1 (>1 = sharper/wider than the anchor)"
                                 % (i, ratio))
                logging.info("[MMH3LoopingSampler] %s", lines[-1].strip())

            # LOCAL PATCH 2026-09-07 (anti-drift mitigations): (1) per-hop pre-compensation of
            # the carried head's channel means against the reference (context-only; the
            # write-back restores the bytes via _pr_orig_head); (2) sigma-scheduled context
            # noise handed to _run_sampling as (level, until_sigma, carried).
            _cn = None
            if carried and carry != "keyframe":
                if _ref_v is not None and _pc_ref0 is None:
                    w0 = windows[0].index_list
                    r0 = w0[0] + offset
                    r1 = min(w0[-1] + 1 + offset, total_t)
                    _pc_ref0 = (fin_v[:, :, r0:r1].float().mean(dim=(0, 2, 3, 4))
                                - _ref_v[:, :, r0:r1].float().mean(dim=(0, 2, 3, 4)).to(out_v.device))
                    lines.append("  reference offset from window 0 (latents %d-%d), |offset| mean %.3f"
                                 % (r0, r1, float(_pc_ref0.abs().mean())))
                if _pc > 0.0:
                    if _pc_ref0 is None:
                        w0 = windows[0].index_list
                        r0 = w0[0] + offset
                        r1 = min(w0[-1] + 1 + offset, total_t)
                        _pc_ref0 = (out_v[:, :, r0:r1].float().mean(dim=(0, 2, 3, 4))
                                    - _ref_v[:, :, r0:r1].float().mean(dim=(0, 2, 3, 4)).to(out_v.device))
                        lines.append("  carry_precomp: reference offset from window 0 (latents %d-%d), "
                                     "|offset| mean %.3f" % (r0, r1, float(_pc_ref0.abs().mean())))
                    ctx_off = (sub_v[:, :, :carried].float().mean(dim=(0, 2, 3, 4))
                               - _ref_v[:, :, v0:v0 + carried].float().mean(dim=(0, 2, 3, 4)).to(sub_v.device))
                    drift = ctx_off - _pc_ref0.to(ctx_off)
                    shift = (-drift).clamp(-_pc_cap, _pc_cap) * _pc
                    if _pr_orig_head is None:
                        _pr_orig_head = out_v[:, :, v0:v0 + carried].clone()
                    sub_v[:, :, :carried] += shift.view(1, -1, 1, 1, 1).to(sub_v.dtype)
                    lines.append("  chunk %d: carry_precomp drift-vs-reference MAE %.3f (max %.3f) "
                                 "-> shift MAE %.3f" % (i, float(drift.abs().mean()),
                                                        float(drift.abs().max()), float(shift.abs().mean())))
                    logging.info("[MMH3LoopingSampler] %s", lines[-1].strip())
                if _cn_level > 0.0:
                    _cn = (_cn_level, _cn_sigma, carried)

            base_cond = chunk_cond
            if chunk_guides:
                if _has_refs(chunk_cond) and not _guide_origin_correct():
                    raise RuntimeError(
                        "MMH3LoopingSampler: chunk %d carries a reference AND a "
                        "keyframe, but this ComfyUI anchors guides on text_len instead "
                        "of the target origin. See docs/core-changes.md." % i)
                chunk_cond = node_helpers.conditioning_set_values(
                    chunk_cond, {"minimax_keyframes": chunk_guides})
            # stage 2 on another grid takes the same keyframes encoded for that grid
            cond2 = chunk_cond
            if two_grid:
                cond2 = (node_helpers.conditioning_set_values(
                    base_cond, {"minimax_keyframes": guides_hi[i]}) if guides_hi.get(i)
                    else base_cond)

            # frame_at_latent, NOT latents_to_frames: v0 is an arbitrary window
            # bound, and latents_to_frames is only meaningful on the 5j+2 grid --
            # it answers -12 for index 1.
            g = _chunk_guider(guider, chunk_cond, frame0=frame_at_latent(v0))
            # The phase-2 guider gets THIS chunk's conditioning too. Only its
            # guidance settings are wanted; its own positive would hand the tail
            # of every chunk whatever prompt happens to be wired to it.
            g2 = None if phase2_guider is None else _chunk_guider(
                phase2_guider, chunk_cond, frame0=frame_at_latent(v0))
            # Stage 2 likewise: its guidance settings and model, THIS chunk's prompt.
            gs2 = None if not stage2_on else _chunk_guider(
                stage2_guider if stage2_guider is not None else guider, cond2,
                frame0=frame_at_latent(v0))
            if tail_mode and carried:
                # Released rows are not all-zero mask rows, so H3ChunkPreviewTap cannot
                # infer the pinned context from the mask; publish the span instead.
                # (_chunk_guider rebinds both option levels, so this stays per chunk.)
                for _gg in (g, g2, gs2):
                    if _gg is not None:
                        _gg.model_options["transformer_options"]["h3_tap_context_rows"] = (
                            int(free), int(carried))
            if preview is not None:
                # Before, not after: the per-step callback fires during the sample
                # and needs to know which chunk it is watching.
                preview.set_chunk(i, v0, v1)
            # LOCAL PATCH 2026-09-07 (re-roll guard): a guarded chunk may be sampled more
            # than once; the reference is the previous chunk's fresh rows (else the carried
            # head bytes). With the guard active every attempt's guider carries
            # transformer_options["mmh3_reroll_attempt"], which tells H3ChunkPreviewTap to
            # defer its per-chunk preview/verdict until the kept attempt is emitted below.
            _rr_ref = None
            if _rr_retries > 0 and carried and carry != "keyframe":
                _rr_ref = _reroll_prev if _reroll_prev is not None else _fresh_stats(
                    _pr_orig_head if _pr_orig_head is not None
                    else fin_v[:, :, v0:v0 + carried])

            def _finish(nz, gs, s1):
                """Stage 2 of this chunk, from stage 1's denoised estimate `s1`."""
                x_v, x_a = unpack_av(s1, "chunk %d stage 1" % i)
                if two_grid:
                    lat_v = TS.upscale_video(
                        x_v, upscale_model, float(upscale_megapixels), upscale_precision,
                        upscale_temporal_chunking, keep_upscaler_loaded).to(x_v.dtype)
                    if stage2_latent is not None and in_mask2_v is not None:
                        # rows the master pins take stage2_latent's exact source
                        m = in_mask2_v[:, :, v0:v1].to(device=lat_v.device, dtype=torch.float32)
                        src = fin_v[:, :, v0:v1].to(device=lat_v.device, dtype=torch.float32)
                        lat_v = (src + (lat_v.float() - src) * m).to(x_v.dtype)
                else:
                    lat_v = x_v.clone()
                if carried:
                    # the pinned carry is the previous chunk's FINISHED rows
                    lat_v[:, :, free:carried] = fin_v[:, :, v0 + free:v0 + carried].to(
                        device=lat_v.device, dtype=lat_v.dtype)
                lat_a = None if x_a is None else x_a.clone()
                if lat_a is not None and carried_a > 0:
                    # anchor the audio carry on the original rows, not stage 1's
                    # partially re-estimated copy (strength < 1 would compound)
                    lat_a[:, :, :, :carried_a] = out_a[:, :, :, a0:a0 + carried_a].to(
                        device=lat_a.device, dtype=lat_a.dtype)
                if two_grid:
                    if carried:
                        nm2 = _carry_mask(lat_v, lat_a, carried, carried_a,
                                          float(overlap_strength_video),
                                          float(overlap_strength_audio), in_mask2_v, in_mask_a,
                                          v0, v1, a0, a1, int(feather_latents))
                        if free:
                            nm2 = TS.release_head(nm2, free, TS.head_floor(
                                in_mask2_v, lat_v, v0, free, accepted_t))
                    elif in_mask2_v is not None or in_mask_a is not None:
                        nm2 = _sliced_mask(lat_v, lat_a, in_mask2_v, in_mask_a, v0, v1, a0, a1)
                    else:
                        nm2 = None
                    noise2 = TS.Stage2Noise(nz, chunk, TS.stage2_seed(getattr(nz, "seed", 0)))
                else:
                    nm2 = chunk.get("noise_mask")
                    noise2 = nz      # same grid: the phase-2 handoff, same noise
                lat = pack_av(chunk, lat_v, lat_a)
                lat.pop("noise_mask", None)
                if nm2 is not None:
                    lat["noise_mask"] = nm2
                _, out2 = SamplerCustomAdvanced().sample(
                    noise2, gs, stage2_sampler if stage2_sampler is not None else sampler,
                    stage2_sigmas, lat)
                return out2

            stage1_by_attempt = {}

            def _run_once(nz, attempt, _g=g, _g2=g2, _gs=gs2):
                ga, g2a, gsa = _g, _g2, _gs
                if _rr_retries > 0:
                    ga = _mark_attempt(_g, attempt)
                    g2a = None if _g2 is None else _mark_attempt(_g2, attempt)
                    gsa = None if _gs is None else _mark_attempt(_gs, attempt)
                s1 = _run_sampling(nz, ga, sampler, sigmas, chunk,
                                   sampling_start_step, sampling_end_step,
                                   phase2_sampler, g2a, phase2_start_step,
                                   carry_noise=_cn)
                if not stage2_on:
                    return s1
                stage1_by_attempt[attempt] = s1
                return _finish(nz, gsa, s1)

            _rr_drift = None
            if (_rr_ref is not None and _ref_v is not None and _pc_ref0 is not None
                    and (_rr_th.get("luma_abs", 0.0) > 0.0 or _rr_th.get("cmean_abs", 0.0) > 0.0)):
                _rr_drift = (_fresh_stats(_ref_v[:, :, v0 + carried:v1]), _pc_ref0)
            done, _rr_kept, _rr_stats = _sample_with_reroll(
                _run_once, noise, i, _rr_retries if _rr_ref is not None else 0,
                _rr_ref, _rr_th, carried, lines, drift=_rr_drift)
            if _rr_retries > 0:
                # with stage 2 the finished chunk comes out of the stage-2 guider: emit
                # through its tap, or the main guider's if only that one is tapped
                _tg = g
                if gs2 is not None and callable(getattr(gs2, "h3_tap_emit", None)):
                    _tg = gs2
                _emit = getattr(_tg, "h3_tap_emit", None)
                if callable(_emit):
                    try:
                        _emit(done["samples"], stage2_sigmas if stage2_on else sigmas)
                    except Exception as e:  # a preview must never kill the run
                        logging.warning("[MMH3LoopingSampler] chunk %d: tap emit failed (%s: %s)",
                                        i, type(e).__name__, e)

            dv, da = unpack_av(done, "chunk %d output" % i)
            # tail_custom: the redrawn carry was context only -- the previous chunk keeps
            # its rows. Accepted rows are never rewritten. Otherwise the whole window, as
            # before (pinned head rows come back byte-identical at strength 1).
            w0 = v0 + carried if tail_mode else v0
            if accepted_t:
                w0 = max(w0, min(accepted_t, v1))
            fin_v[:, :, w0:v1] = dv[:, :, w0 - v0:].to(fin_v.dtype)
            if two_grid:
                s1v = unpack_av(stage1_by_attempt[_rr_kept], "chunk %d stage 1" % i)[0]
                out_v[:, :, w0:v1] = s1v[:, :, w0 - v0:].to(out_v.dtype)
                hi_valid[w0:v1] = True
            if out_a is not None and da is not None:
                aw0 = max(a0, _audio_index_at(accepted_t, total_t, total_a)) if accepted_t else a0
                out_a[:, :, :, aw0:a1] = da[:, :, :, aw0 - a0:].to(out_a.dtype)
            # LOCAL PATCH 2026-08-31 (pin_renorm): the renormalized head has done its
            # job as context; restore the original bytes so already-rendered overlap
            # frames are never altered (mirrors the prior_t restore below).
            if _pr_orig_head is not None:
                out_v[:, :, v0:v0 + carried] = _pr_orig_head
            if _rr_retries > 0:
                _reroll_prev = (_rr_stats if _rr_stats is not None
                                else _fresh_stats(fin_v[:, :, v0 + carried:v1]))

            lines.append("  chunk %d: prompt %d, frames %d-%d, %d carried%s"
                         % (i, min(i, len(conds) - 1), spans[i][0], spans[i][1],
                            frame_at_latent(carried) if carried else 0,
                            ", %d keyframe(s)" % (len(chunk_guides) - (1 if carried
                                                  and carry == "keyframe" else 0))
                            if guides_by_chunk.get(i) else ""))
            if (tail_mode and carried) or stage2_on:
                lines[-1] += "".join([
                    (", context %d of %d carried latents pinned (%d frames), %d redrawn"
                     % (carried - free, carried,
                        frame_at_latent(v0 + carried) - frame_at_latent(v0 + free), free))
                    if tail_mode and carried else "",
                    ", stage 2 done" if stage2_on else ""])
            if preview is not None:
                preview.chunk(i, dv, v0, v1)
            logging.info("[MMH3LoopingSampler] chunk %d/%d done", i + 1, n)

        # The first generated chunk OVERLAPS the prior -- that is what the carry is --
        # so its slice covers the prior's tail and the write-back lands on it. The mask
        # protects that tail only at strength 1.0, and overlap_strength_audio defaults
        # to 0.9, so the source's last fraction of a second would come back altered.
        # The carry has already done its job as context by now; restore the prior so
        # the output is the input plus the addition, byte for byte.
        if prior_t:
            out_v[:, :, :prior_t] = keep_v
            if out_a is not None and keep_a is not None:
                out_a[:, :, :, :prior_at] = keep_a

        out = pack_av(latent, fin_v, out_a, noise_mask=None)
        stage1_out = out
        if two_grid:
            # the input's noise_mask is on the stage-1 grid; it stays with stage1_latent
            out.pop("noise_mask", None)
            stage1_out = pack_av(latent, out_v, out_a, noise_mask=None)
        lines.append("master: %d latents (%d frames, %.2fs) -- the input length, exactly"
                     % (total_t, total_f, total_f / float(FPS)))
        report = "\n".join(lines)
        logging.info("[MMH3LoopingSampler] " + lines[0])
        return io.NodeOutput(out, n, report, stage1_out)


def _split_mask(latent):
    """The latent's own noise mask per modality, or (None, None).

    A supplied audio track pinned upstream arrives this way, and slicing it per
    chunk is the whole reason chunks read their mask rather than building one.
    """
    m = latent.get("noise_mask")
    if m is None:
        return None, None
    if isinstance(m, NestedTensor):
        parts = m.unbind()
        return parts[0], parts[-1]
    if getattr(m, "ndim", 0) == 5:
        return m, None
    return None, m


def _ones_like_mask(v, a):
    vm = torch.ones([v.shape[0], 1] + list(v.shape[2:]), dtype=torch.float32,
                    device=v.device)
    am = None if a is None else torch.ones(
        [a.shape[0], 1, a.shape[2], a.shape[3]], dtype=torch.float32, device=a.device)
    return vm, am


# The DiT reads the mask per 2x2 LATENT patch, so 2 latent cells -- 32 pixels at the
# VAE's /16 -- is the finest feature a mask can express. Snapping each patch to one
# value keeps an edge from landing mid-patch, where the token would otherwise carry a
# partial pooled strength and get its own timestep (rows_t = 1 - m*sigma).
MASK_TOKEN_PATCH = 2

_MASK_REDUCE = {
    "max": lambda g, d: g.amax(dim=d),
    "min": lambda g, d: g.amin(dim=d),
    "mean": lambda g, d: g.mean(dim=d),
    "last": lambda g, d: g.select(d, g.shape[d] - 1),
}


def _token_snap(x, method):
    """One value per 2x2 latent patch, replicate-padded on odd edges."""
    t, h, w = x.shape
    p = MASK_TOKEN_PATCH
    x = torch.nn.functional.pad(x[:, None], (0, -w % p, 0, -h % p), mode="replicate")
    if method == "min":
        x = -torch.nn.functional.max_pool2d(-x, p)
    elif method == "mean":
        x = torch.nn.functional.avg_pool2d(x, p)
    else:
        x = torch.nn.functional.max_pool2d(x, p)
    x = x.repeat_interleave(p, dim=-2).repeat_interleave(p, dim=-1)
    return x[:, 0, :h, :w]


def _mask_to_video_latent(mask, total_t, lat_h, lat_w, mode):
    """MASK batch -> [1,1,total_t,lat_h,lat_w] on the latent grid.

    Spatially pooled rather than interpolated: bilinear averages, which manufactures
    a fractional value along every edge, and on H3 each such cell then denoises at
    its own timestep. Temporally grouped on the VAE's real cycle -- latent k covers
    frames [frame_at_latent(k), frame_at_latent(k+1)) -- so an edge lands where the
    encoder actually put it. A uniform split misplaces it: the first latent of every
    17-frame group covers ONE frame and the other four cover four each.
    """
    m = mask
    if m.ndim == 4:
        m = m[..., 0] if m.shape[-1] == 1 else m.mean(dim=-1)
    if m.ndim == 2:
        m = m[None]
    m = m.to(torch.float32)

    reduce = _MASK_REDUCE.get(mode, _MASK_REDUCE["max"])
    x = m[:, None]
    if mode == "min":
        x = -torch.nn.functional.adaptive_max_pool2d(-x, (lat_h, lat_w))
    elif mode == "mean":
        x = torch.nn.functional.adaptive_avg_pool2d(x, (lat_h, lat_w))
    else:
        x = torch.nn.functional.adaptive_max_pool2d(x, (lat_h, lat_w))
    x = _token_snap(x[:, 0], mode)

    n = int(x.shape[0])
    if n == 1:
        out = x.expand(total_t, -1, -1)
    elif n == total_t:
        out = x
    else:
        px = frame_at_latent(total_t)
        if n != px:
            idx = torch.linspace(0, n - 1, px).round().long().clamp(0, n - 1)
            x = x[idx]
        rows = []
        for k in range(total_t):
            a, b = frame_at_latent(k), frame_at_latent(k + 1)
            a, b = min(a, x.shape[0] - 1), min(max(b, a + 1), x.shape[0])
            rows.append(reduce(x[a:b], 0))
        out = torch.stack(rows)
    return out[None, None].contiguous()


def _video_mask_to_audio(mv, total_t, total_a, mode):
    """Put a mask already on the video latent grid onto the audio grid.

    Video latent k owns audio [_audio_index_at(k), _audio_index_at(k+1)); each span
    takes the spatial reduction of that latent's mask. Boundaries are converted
    independently because audio_t is not additive.

    Called ONLY on `audio_denoise_mask`, which arrives pooled to 1x1, so the spatial
    reduction is a no-op and this is purely the time mapping. Deliberately NOT applied
    to `denoise_mask`: a spatial mask carries no temporal intent, and reducing a
    subject matte this way returns "free" at every timestep.
    """
    reduce = _MASK_REDUCE.get(mode, _MASK_REDUCE["max"])
    per_t = reduce(reduce(mv[0, 0].reshape(total_t, -1), 1)[:, None], 1)
    prof = torch.ones(total_a, dtype=torch.float32, device=mv.device)
    for k in range(total_t):
        a = _audio_index_at(k, total_t, total_a)
        b = _audio_index_at(k + 1, total_t, total_a) if k + 1 < total_t else total_a
        if b > a:
            prof[a:b] = per_t[k]
    return prof


def _audio_profile_to_mask(prof, sub_a_like):
    """[T40] profile -> [B,1,2,T40] on the audio latent's own axes (temporal = dim 3)."""
    return prof.view(1, 1, 1, -1).expand(
        sub_a_like.shape[0], 1, sub_a_like.shape[2], -1).contiguous()


def _sliced_mask(sub_v, sub_a, in_v, in_a, v0, v1, a0, a1):
    """This chunk's slice of whatever mask the master carried."""
    vm, am = _ones_like_mask(sub_v, sub_a)
    if in_v is not None:
        vm = in_v[:, :, v0:v1].to(dtype=vm.dtype, device=vm.device).clone()
    if am is not None and in_a is not None:
        am = in_a[:, :, :, a0:a1].to(dtype=am.dtype, device=am.device).clone()
    return NestedTensor([vm, am]) if am is not None else vm


# LOCAL PATCH 2026-08-31 (pin_renorm, user-requested) ------------------------
def _pin_renorm_stats(v):
    """Per-channel float32 mean/std of a video latent span [B,C,T,H,W]."""
    x = v.to(torch.float32)
    return x.mean(dim=(0, 2, 3, 4)), x.std(dim=(0, 2, 3, 4))


def _pin_renorm_head(sub_v, carried, ref_stats, strength):
    """Match the carried head's per-channel latent stats to the reference.

    In-place on sub_v's first `carried` rows: x' = (x-mu)/sd * ref_sd + ref_mu,
    blended by `strength`. Channels with degenerate spread are left alone.
    Returns the mean pre-renorm sigma-ratio (head_sd / ref_sd) over the
    non-degenerate channels -- the latent-space drift meter for the report.
    """
    ref_mean, ref_std = ref_stats
    head = sub_v[:, :, :carried].to(torch.float32)
    mu = head.mean(dim=(0, 2, 3, 4), keepdim=True)
    sd = head.std(dim=(0, 2, 3, 4), keepdim=True)
    rm = ref_mean.view(1, -1, 1, 1, 1).to(head.device)
    rs = ref_std.view(1, -1, 1, 1, 1).to(head.device)
    ok = (sd > 1e-6) & (rs > 1e-6)
    target = torch.where(ok, (head - mu) / sd.clamp_min(1e-6) * rs + rm, head)
    blended = head + float(strength) * (target - head)
    sub_v[:, :, :carried] = blended.to(sub_v.dtype)
    ratios = (sd / rs.clamp_min(1e-6)).reshape(-1)[ok.reshape(-1)]
    return float(ratios.mean().item()) if ratios.numel() else 1.0


# LOCAL PATCH 2026-09-03 (band-aware pin_renorm, user-requested) -------------
# Global renorm caps the carried head's TOTAL per-channel variance but cannot
# stop drift from redistributing energy between spatial frequency bands inside
# that budget -- measured 2026-09-03 as mid-band mottle (skin blotch) roughly
# doubling over ~10 chunks while global sigma and high-frequency lapvar stayed
# flat. Per-band mode splits the head into three spatial bands per latent frame
# (gaussian blurs at sigma 1 and 3 latent px: high = x - G1, mid = G1 - G3,
# low = G3) and matches each band's per-channel sigma to chunk 1's band
# independently (the low band also matches the mean; high/mid are ~zero-mean by
# construction). Bands sum exactly back to the input, so strength blending and
# the byte-original write-back restore behave exactly like global mode.

_PIN_BAND_SIGMAS = (1.0, 3.0)


def _pin_gauss_blur(flat, sigma):
    """Separable replicate-pad gaussian blur on [N,C,H,W] float32."""
    import torch.nn.functional as F
    c = flat.shape[1]
    k = max(3, int(sigma * 4) | 1)
    g = torch.arange(k, dtype=torch.float32, device=flat.device) - k // 2
    g = torch.exp(-g * g / (2.0 * sigma * sigma))
    g = g / g.sum()
    kx = g.view(1, 1, 1, k).expand(c, 1, 1, k)
    ky = g.view(1, 1, k, 1).expand(c, 1, k, 1)
    p = k // 2
    out = F.conv2d(F.pad(flat, (p, p, 0, 0), mode="replicate"), kx, groups=c)
    return F.conv2d(F.pad(out, (0, 0, p, p), mode="replicate"), ky, groups=c)


def _pin_band_split(x):
    """[B,C,T,H,W] float32 -> (high, mid, low) bands, summing exactly to x."""
    b, c, t, h, w = x.shape
    flat = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    g1 = _pin_gauss_blur(flat, _PIN_BAND_SIGMAS[0])
    g3 = _pin_gauss_blur(flat, _PIN_BAND_SIGMAS[1])
    def back(y):
        return y.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
    return back(flat - g1), back(g1 - g3), back(g3)


def _pin_renorm_band_stats(v):
    """Per-band, per-channel float32 (mean, std) of a video latent span."""
    bands = _pin_band_split(v.to(torch.float32))
    return [(b.mean(dim=(0, 2, 3, 4)), b.std(dim=(0, 2, 3, 4))) for b in bands]


def _pin_renorm_head_band(sub_v, carried, ref_band_stats, strength,
                          iterations=3):
    """Per-band variant of _pin_renorm_head: each band's per-channel sigma is
    matched to its chunk-1 band (low band also matches the mean), bands are
    recomposed, and the whole correction is blended by `strength`. In-place on
    sub_v's first `carried` rows. Returns (hi, mid, lo) mean pre-renorm
    sigma-ratios over non-degenerate channels -- the per-band drift meter.

    Gaussian bands overlap (not orthogonal projections), so a single
    fix-and-recompose under-corrects: re-splitting the result leaks energy
    between bands. The correction is therefore iterated to a fixed point
    (3 rounds ≈ band sigmas within a few percent) before the single
    strength-blend against the original head."""
    head = sub_v[:, :, :carried].to(torch.float32)
    ratios = None
    cur = head
    for _ in range(max(1, int(iterations))):
        bands = _pin_band_split(cur)
        fixed_sum = torch.zeros_like(cur)
        round_ratios = []
        for bi, (band, (rm, rs)) in enumerate(zip(bands, ref_band_stats)):
            mu = band.mean(dim=(0, 2, 3, 4), keepdim=True)
            sd = band.std(dim=(0, 2, 3, 4), keepdim=True)
            rmv = rm.view(1, -1, 1, 1, 1).to(band.device)
            rsv = rs.view(1, -1, 1, 1, 1).to(band.device)
            ok = (sd > 1e-6) & (rsv > 1e-6)
            if bi == 2:  # low band carries the DC level
                fixed = (band - mu) / sd.clamp_min(1e-6) * rsv + rmv
            else:        # high/mid are ~zero-mean: pure sigma rescale
                fixed = band * (rsv / sd.clamp_min(1e-6))
            fixed_sum = fixed_sum + torch.where(ok, fixed, band)
            r = (sd / rsv.clamp_min(1e-6)).reshape(-1)[ok.reshape(-1)]
            round_ratios.append(float(r.mean().item()) if r.numel() else 1.0)
        if ratios is None:
            ratios = tuple(round_ratios)  # report the PRE-renorm drift
        cur = fixed_sum
    blended = head + float(strength) * (cur - head)
    sub_v[:, :, :carried] = blended.to(sub_v.dtype)
    return ratios
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# LOCAL PATCH 2026-09-07: in-loop re-roll guard (exposure-snap detector) helpers.
# Root cause measured on saved masters (scripts/h3_latent_drift_probe.py): the carried
# head is the model's own previous output shown as a clean reference; its statistics
# drift off the natural VAE-latent distribution chunk over chunk and, at some chunk, the
# high-sigma decisions (exposure/tone) snap back to the prior. Identical seeds snap at the
# same chunk; a different seed usually does not. The guard measures the same three latent
# statistics the probe prints and re-samples a snapped chunk with a different seed.
# ---------------------------------------------------------------------------
_REROLL_DEFAULTS = {"luma": 0.14, "hf": 3.0, "cmean": 0.25,
                    # cumulative deviation from the chunk-0 relation to `reference_latent`
                    # (only scored when a reference is wired); 0 = off
                    "luma_abs": 0.0, "cmean_abs": 0.0}
_RGB_FACTORS = None


def _rgb_factors():
    global _RGB_FACTORS
    if _RGB_FACTORS is None:
        from comfy.latent_formats import MiniMaxH3Video
        w = torch.tensor(MiniMaxH3Video.latent_rgb_factors, dtype=torch.float32)
        b = getattr(MiniMaxH3Video, "latent_rgb_factors_bias", None)
        b = torch.tensor(b, dtype=torch.float32) if b is not None else torch.zeros(3)
        _RGB_FACTORS = (w, b)
    return _RGB_FACTORS


def _reroll_parse_thresholds(text):
    th = dict(_REROLL_DEFAULTS)
    for part in str(text or "").replace(";", ",").split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip().lower()
        if k in th:
            try:
                th[k] = max(0.0 if k.endswith("_abs") else 1e-6, float(v))
            except ValueError:
                pass
    return th


def _fresh_stats(v):
    """Latent stats of a [B,C,t,H,W] region: luma proxy (latent->RGB, same einsum as the
    probe, no 0..1 remap), spatial HF share (residual after a replicate-padded 3x3 box
    blur, in % of the total std) and the per-channel means."""
    x = v.detach().float().cpu()
    if x.ndim != 5 or x.shape[2] == 0:
        return {"luma": 0.0, "hf": 0.0, "ch_mean": torch.zeros(x.shape[1] if x.ndim == 5 else 24)}
    w, b = _rgb_factors()
    c = min(int(x.shape[1]), int(w.shape[0]))
    rgb = torch.einsum("bcthw,ck->bkthw", x[:, :c], w[:c]) + b.view(1, 3, 1, 1, 1)
    luma = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]
    tot = float(x.std())
    xp = torch.nn.functional.pad(x, (1, 1, 1, 1, 0, 0), mode="replicate")
    blur = torch.nn.functional.avg_pool3d(xp, (1, 3, 3), stride=1)
    hf = 100.0 * float((x - blur).std()) / max(tot, 1e-6)
    return {"luma": float(luma.mean()), "hf": hf, "ch_mean": x.mean(dim=(0, 2, 3, 4))}


def _reroll_score(stats, ref, th, drift=None):
    """Step score vs the previous chunk (`ref`), plus -- when `drift` = (source_stats_of_fresh_rows,
    target_offset_ch_mean) is given and the *_abs thresholds are set -- the attempt's CUMULATIVE
    deviation from the chunk-0 relation to the reference (a walk that continues smoothly never
    trips the step score; this term lets the guard pick the attempt closest to target)."""
    d_luma = float(stats["luma"] - ref["luma"])
    d_hf = float(stats["hf"] - ref["hf"])
    d_cm = float((stats["ch_mean"] - ref["ch_mean"]).abs().mean())
    parts = {"luma": abs(d_luma) / th["luma"], "hf": abs(d_hf) / th["hf"], "cmean": d_cm / th["cmean"]}
    deltas = {"luma": d_luma, "hf": d_hf, "cmean": d_cm}
    if drift is not None:
        src_stats, target_off = drift
        off = stats["ch_mean"] - src_stats["ch_mean"]           # this attempt's offset from the source
        dev = off - target_off.to(off)                          # ... relative to chunk 0's offset
        w, b = _rgb_factors()
        luma_dev = float((dev.view(1, -1) @ w[:dev.numel()]).squeeze(0) @ torch.tensor([0.2126, 0.7152, 0.0722]))
        cm_dev = float(dev.abs().mean())
        deltas["luma_abs"] = luma_dev
        deltas["cmean_abs"] = cm_dev
        if th.get("luma_abs", 0.0) > 0.0:
            parts["luma_abs"] = abs(luma_dev) / th["luma_abs"]
        if th.get("cmean_abs", 0.0) > 0.0:
            parts["cmean_abs"] = cm_dev / th["cmean_abs"]
    return max(parts.values()), parts, deltas


def _mark_attempt(guider, attempt):
    """Per-attempt guider copy tagged with transformer_options['mmh3_reroll_attempt'] (both
    option levels rebound, like _chunk_guider), so taps can tell attempts apart."""
    new_g = copy.copy(guider)
    opts = dict(getattr(guider, "model_options", {}) or {})
    opts["transformer_options"] = dict(opts.get("transformer_options", {}))
    opts["transformer_options"]["mmh3_reroll_attempt"] = int(attempt)
    new_g.model_options = opts
    return new_g


def _sample_with_reroll(run_once, noise, i, retries, ref, th, carried, lines, drift=None):
    """Sample chunk i up to retries+1 times. Attempt a uses seed base+i+1000*a (attempt 0 ==
    the unguarded seed). Returns (kept_output, kept_attempt, kept_fresh_stats or None)."""
    best = None
    attempt = 0
    for attempt in range(int(retries) + 1):
        nz = _chunk_noise(noise, i + 1000 * attempt) if attempt else _chunk_noise(noise, i)
        done = run_once(nz, attempt)
        if ref is None or int(retries) <= 0:
            return done, 0, None
        dv, _ = unpack_av(done, "chunk %d attempt %d" % (i, attempt))
        st = _fresh_stats(dv[:, :, carried:])
        score, parts, d = _reroll_score(st, ref, th, drift)
        seed_txt = getattr(nz, "seed", "?")
        lines.append("  chunk %d: reroll attempt %d (seed %s) -> dluma %+.3f dhf %+.2f "
                     "cmean %.3f%s score %.2f%s"
                     % (i, attempt, seed_txt, d["luma"], d["hf"], d["cmean"],
                        (" | vs target: luma %+.3f cmean %.3f" % (d["luma_abs"], d["cmean_abs"]))
                        if "luma_abs" in d else "", score,
                        "" if score <= 1.0 else "  EXCEEDS (%s)" % ", ".join(
                            k for k, v in parts.items() if v > 1.0)))
        logging.info("[MMH3LoopingSampler] %s", lines[-1].strip())
        if best is None or score < best[0]:
            best = (score, done, attempt, st)
        if score <= 1.0:
            break
    if attempt > 0 or best[0] > 1.0:
        lines.append("  chunk %d: reroll kept attempt %d (score %.2f%s)"
                     % (i, best[2], best[0], "" if best[0] <= 1.0 else ", still over threshold"))
        logging.info("[MMH3LoopingSampler] %s", lines[-1].strip())
    return best[1], best[2], best[3]
# ---------------------------------------------------------------------------


def _carry_mask(sub_v, sub_a, carried, carried_a, strength_v, strength_a,
                in_v, in_a, v0, v1, a0, a1, feather=0):
    """Preserve the carried head, honouring any mask the master already had.

    Starts from the master's own mask for this span -- a pinned audio track has
    to survive -- then pins the carry on top.

    There is deliberately NO feather. A ramp of intermediate mask values makes the
    seam NOISY on a core with the rebased #15375: each ramped cell gets its own
    timestep (rows_t = 1 - m*sigma) while the sampler blends its content as
    the two are now reconciled by core: scale_latent_inpaint pre-compensates so
    every pixel lands at its token's pooled strength. The feather was removed in
    0.73.0 on the evidence, and that stands -- but the mechanism recorded for it
    did not survive re-reading (see CHANGELOG 0.76.0). Observed 2026-08-13; the
    `feather_latents` input was removed in 0.73.0 rather than left as a trap.
    """
    packed = _sliced_mask(sub_v, sub_a, in_v, in_a, v0, v1, a0, a1)
    if isinstance(packed, NestedTensor):
        vm, am = packed.unbind()
    else:
        vm, am = packed, None

    # keep-wins (clamp, not overwrite): the master's own mask may already PIN part
    # of the carry span harder than the carry strength does -- an inpaint keep
    # region, a use_input_audio track. Overwriting re-noised those pins by
    # (1 - strength) every carry head (local patch 2026-08-24; identical when the
    # master carried no mask, since the slice is then all ones).
    vm[:, :, :carried] = vm[:, :, :carried].clamp(max=1.0 - strength_v)
    if am is not None and carried_a > 0:
        am[:, :, :, :carried_a] = am[:, :, :, :carried_a].clamp(max=1.0 - strength_a)

    # RESTORED 2026-09-12 for re-test, exactly as removed in 0.73.0: a linear ramp over
    # the first `feather` NEW rows, easing back to full generation instead of stepping at
    # the seam. VIDEO only. Composes with MINIMUM so easing toward full denoise cannot
    # un-pin something deliberately kept. 0 = off = byte-identical to the hard step.
    # The 0.72.2 observation (a non-zero feather made the seam NOISY) stands unexplained:
    # 0.76.0 retracted its recorded mechanism. Leave at 0 outside the A/B.
    if feather > 0:
        end = min(carried + int(feather), vm.shape[2])
        steps = end - carried
        if steps > 0:
            ramp = torch.linspace(1.0 - strength_v, 1.0, steps + 1,
                                  device=vm.device)[1:]
            vm[:, :, carried:end] = torch.minimum(ramp.view(1, 1, steps, 1, 1),
                                                  vm[:, :, carried:end])
    return NestedTensor([vm, am]) if am is not None else vm


class MMH3KeyframePlanner(io.ComfyNode):
    """End-anchored keyframe indices for a chained run.

    TRAVEL SEMANTICS, from LTXAVTools' planner. The first keyframe (optional)
    opens the clip at frame 0; every later one sits at the END of its chunk, so
    each chunk generates TOWARD its destination image and the next continues
    from the arrived state through the ordinary carry. Start-anchoring instead
    would put the image in the NEXT chunk and invite a snap at every seam. The
    final chunk's end is the clip's end, emitted as -1.

    Takes the same three numbers the sampler does and derives the same schedule
    from `_plan`, so the two cannot disagree about where a chunk ends.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3KeyframePlanner",
            display_name="MiniMax H3 Keyframe Planner",
            category="MMH3Tools/calculators",
            description=(
                "End-anchored keyframe indices: frame 0 opens, each chunk travels to a "
                "keyframe at its end, the last ends on -1. Wire `indices` to the Looping "
                "Sampler's keyframe_indices and `count` tells you how many images its "
                "`keyframes` batch needs, in that order."
            ),
            inputs=[
                io.Int.Input("total_frames", default=192, min=5, max=3600, step=17,
                             tooltip="The whole clip, same as the sampler's latent."),
                io.Int.Input("chunk_frames", default=192, min=5, max=3600, step=17,
                             tooltip="Same value the sampler gets."),
                io.Int.Input("overlap_frames", default=22, min=0, max=3600, step=17,
                             tooltip="Same value the sampler gets. It changes where each "
                                     "chunk ends, so it changes every index."),
                io.Boolean.Input("include_start", default=True,
                                 tooltip="A keyframe at frame 0 -- the opening image."),
                io.Boolean.Input("include_end", default=True,
                                 tooltip="A keyframe at -1 -- the closing image."),
            ],
            outputs=[
                io.String.Output(display_name="indices"),
                io.Int.Output(display_name="count"),
                io.Int.Output(display_name="chunk_count"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, total_frames, chunk_frames, overlap_frames,
                include_start, include_end) -> io.NodeOutput:
        _length, _ov, total_f, _t, windows = _plan(
            int(total_frames), int(chunk_frames), int(overlap_frames),
            "standard_static")
        spans = _window_frame_spans(windows, total_f)

        # A chunk's travel destination is the last frame IT renders, not the last
        # frame it spans. The final chunk is clamped to end on the clip, so it
        # overlaps its predecessor by far more than the nominal overlap -- and a
        # frame at that predecessor's span end is then inside the last chunk's NEW
        # content, which claims it. Using span ends left the second-to-last chunk
        # with no destination and gave the last one two.
        ov_f = frame_at_latent(_ov)
        dests = []
        for i in range(len(spans) - 1):
            d = min(spans[i + 1][0] + ov_f - 1, spans[i][1])
            dests.append(max(d, spans[i][0]))

        entries = ([0] if include_start else []) + dests
        if include_end:
            entries.append(-1)

        idx = ", ".join(str(e) for e in entries)
        lines = ["%d chunk%s over %d frames (%.2fs), %d keyframe%s"
                 % (len(spans), "" if len(spans) == 1 else "s", total_f,
                    total_f / float(FPS), len(entries),
                    "" if len(entries) == 1 else "s")]
        for e in entries:
            g = total_f - 1 if e < 0 else e
            lines.append("  frame %-6s %6.2fs%s"
                         % (e, g / float(FPS), "   (end of clip)" if e < 0 else ""))
        if not entries:
            lines.append("  ! nothing to place -- one chunk with both ends off")
        report = "\n".join(lines)
        logging.info("[MMH3KeyframePlanner] " + lines[0])
        return io.NodeOutput(idx, len(entries), len(spans), report)
