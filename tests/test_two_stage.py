"""MMH3LoopingSampler context frames and stage 2, through core's real sampling path.

A tiny CPU stand-in for MiniMax-H3 (the model_base class with a toy forward) runs
under stock SamplerCustomAdvanced / Guider_Basic / Euler, so masks, per-row
timesteps, nested AV noise and the inpaint re-injection are core's own. A spy on
SamplerCustomAdvanced records every call; a stub upscaler stands in for the learned
one (nearest resize on the latent grid).

Plain script like the rest of the suite: PASS/FAIL per check, non-zero exit on any
failure. Run from the pack root: python tests/test_two_stage.py
"""

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options
comfy.options.enable_args_parsing()

from types import SimpleNamespace

import torch
import torch.nn.functional as F
import comfy.latent_formats
import comfy.model_base
import comfy.model_patcher
import comfy.model_sampling
import comfy.samplers
import nodes
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_custom_sampler import Guider_Basic, Noise_RandomNoise
from comfy_extras.nodes_minimax_h3 import EmptyMiniMaxH3LatentAV

from mmh3tools import nodes_looping_sampler as LS
from mmh3tools import two_stage as TS
from mmh3tools.common import frame_at_latent

torch.set_num_threads(2)

fails = []


def check(label, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + label + ("" if ok or not detail else "  " + str(detail)))
    if not ok:
        fails.append(label)


def raises(label, fn, needle):
    try:
        fn()
    except (ValueError, RuntimeError) as e:
        check(label, needle in str(e), e)
        return
    check(label, False, "no error raised")


# ---- tiny H3 -----------------------------------------------------------------
class Sampling(comfy.model_sampling.ModelSamplingAV, comfy.model_sampling.CONST):
    pass


CALLS = []          # every model forward: (tag, transformer_options snapshot)


class ToyH3(comfy.model_base.MiniMaxH3):
    def __init__(self, tag, gain=0.27):
        torch.nn.Module.__init__(self)
        self.diffusion_model = torch.nn.Linear(1, 1)
        self.diffusion_model.dtype = torch.float32
        self.diffusion_model.patch_size = (1, 2, 2)
        self.diffusion_model.preprocess_text_embeds = lambda x: x
        self.latent_format = comfy.latent_formats.MiniMaxH3AV()
        self.model_sampling = Sampling()
        self.model_sampling.set_parameters(shift=12, audio_shift=3)
        self.latent_shapes = None
        self.concat_keys = ()
        self.manual_cast_dtype = None
        self.model_config = SimpleNamespace(memory_usage_factor=1)
        self.tag, self.gain = tag, gain

    def memory_required(self, input_shape, cond_shapes={}):
        return 0

    def apply_model(self, x, t, c_crossattn=None, transformer_options={}, **kwargs):
        CALLS.append((self.tag, dict(transformer_options)))
        # context-dependent on purpose: a carried row that changes changes the output
        return self.gain * x.tanh() + .13 * x.mean() + .31 * t.reshape(-1, 1, 1) + .02 * c_crossattn.mean()


def patcher(tag, gain=0.27):
    p = comfy.model_patcher.ModelPatcher(ToyH3(tag, gain), torch.device("cpu"), torch.device("cpu"))
    p.model.current_patcher = p
    return p


def guider(p):
    g = Guider_Basic(p)
    g.set_conds([[torch.zeros((1, 2, 3)), {}]])
    return g


def prompt(i):
    return [[torch.full((1, 2, 3), float(i)), {}]]


SAMPLES = []        # every SamplerCustomAdvanced call


class SpySampler(LS.SamplerCustomAdvanced):
    def sample(self, noise, guider, sampler, sigmas, latent_image):
        SAMPLES.append({"tag": guider.model_patcher.model.tag, "sigmas": sigmas.clone(),
                        "latent": latent_image, "noise": noise,
                        "positive": (getattr(guider, "original_conds", {}) or {}).get("positive"),
                        "opts": dict((getattr(guider, "model_options", {}) or {}).get("transformer_options", {}))})
        return super().sample(noise, guider, sampler, sigmas, latent_image)


LS.SamplerCustomAdvanced = SpySampler


class StubUpscaler:
    """Stands in for MinimaxH3LatentUpscaler3D: nearest resize to the target grid."""
    calls = 0

    @classmethod
    def execute(cls, latent, model_name, mode, align, chunking, force_unload, device, precision):
        cls.calls += 1
        v = latent["samples"]
        size = TS.target_size(v, mode["megapixels"], align)
        return ({"samples": TS.resize_frames(v, size, "nearest-exact")},)


nodes.NODE_CLASS_MAPPINGS["MinimaxH3LatentUpscaler3D"] = StubUpscaler

FRAMES, CHUNK, OV = 141, 73, 39     # 42 latents: windows 0-21, 10-31, 20-41; carry 12
SIG = torch.tensor([1.0, .95, .85, .7, .5, .25, 0.0])
MP_HI = 64 * 64 / (1024.0 * 1024.0)  # 32x32 px master -> 64x64 px stage-2 grid


def master(frames=FRAMES, value=None):
    lat = EmptyMiniMaxH3LatentAV.execute(32, 32, frames)[0]
    if value is not None:
        v, a = lat["samples"].unbind()
        lat = dict(lat)
        lat["samples"] = NestedTensor([torch.full_like(v, value), torch.full_like(a, value)])
    return lat


def run(latent=None, sigmas=SIG, g=None, prompts=3, **kw):
    del CALLS[:], SAMPLES[:]
    kw.setdefault("overlap_strength_video", 1.0)
    kw.setdefault("overlap_strength_audio", 0.9)
    res = LS.MMH3LoopingSampler.execute(
        Noise_RandomNoise(123), g or guider(patcher("A")), comfy.samplers.sampler_object("euler"),
        sigmas, {"conds": [prompt(i) for i in range(prompts)]},
        latent if latent is not None else master(), CHUNK, OV, "mask", **kw).result
    return res


def vid(lat):
    return lat["samples"].unbind()[0]


def aud(lat):
    return lat["samples"].unbind()[1]


def vmask(lat):
    m = lat.get("noise_mask")
    if m is None:
        return None
    return m.unbind()[0] if isinstance(m, NestedTensor) else m


print("\n1. tail_custom: rows math and masks")
check("5 frames of a 39-frame carry = 2 rows", TS.tail_context_rows(10, 22, 5) == 2)
check("6 frames rounds up to 3 rows (9 frames)", TS.tail_context_rows(10, 22, 6) == 3)
check("39 frames = the whole 12-row carry", TS.tail_context_rows(10, 22, 39) == 12)
check("0 frames releases every row", TS.tail_context_rows(10, 22, 0) == 0)

full = run()
full_v = vid(full[0])
tail = run(overlap_pin="tail_custom", context_frames=5)
tail_v = vid(tail[0])
stage1 = [s for s in SAMPLES if s["tag"] == "A"]
m1 = vmask(stage1[1]["latent"])
check("window 1 mask: released rows 0-9 free", bool((m1[:, :, :10] == 1).all()))
check("window 1 mask: rows 10-11 pinned", bool((m1[:, :, 10:12] == 0).all()))
check("window 1 mask: fresh rows free", bool((m1[:, :, 12:] == 1).all()))
check("tap span published as (free, carried)", stage1[1]["opts"].get("h3_tap_context_rows") == (10, 12))
check("no tap span on the uncarried window 0", "h3_tap_context_rows" not in stage1[0]["opts"])
check("previous chunk's rows unchanged (0-21 = full run's window 0)",
      torch.equal(tail_v[:, :, :22], full_v[:, :, :22]))
check("released context changed the new rows", not torch.equal(tail_v[:, :, 22:], full_v[:, :, 22:]))
check("report states the effective context",
      "context 2 of 12 carried latents pinned (5 frames), 10 redrawn" in tail[2], tail[2])

same = run(overlap_pin="tail_custom", context_frames=39)
check("context >= overlap == full, byte for byte", torch.equal(vid(same[0]), full_v)
      and torch.equal(aud(same[0]), aud(full[0])))

print("\n2. v2v keeps and the accepted prefix")
lat = master(value=0.5)
keep = torch.ones([1, 1, 42, 2, 2])
keep[:, :, 12:14] = 0.0                 # a pinned v2v region inside window 1's released span
lat["noise_mask"] = NestedTensor([keep, torch.ones([1, 1, 2, aud(lat).shape[-1]])])
run(latent=lat, overlap_pin="tail_custom", context_frames=5)
m1 = vmask([s for s in SAMPLES if s["tag"] == "A"][1]["latent"])
check("released rows return to the master's keep (rows 2-3 stay 0)", bool((m1[:, :, 2:4] == 0).all()))
check("other released rows free", bool((m1[:, :, :2] == 1).all()) and bool((m1[:, :, 4:10] == 1).all()))

lat = master(value=0.5)
pin = torch.ones([1, 1, 42, 2, 2])
pin[:, :, :22] = 0.0                    # the accepted window pinned, as H3ContinueMaster does
lat["noise_mask"] = NestedTensor([pin, torch.ones([1, 1, 2, aud(lat).shape[-1]])])
res = run(latent=lat, overlap_pin="tail_custom", context_frames=5, accepted_prefix_frames=73)
starts = [frame_at_latent(int(s["opts"].get("mmh3_control_frame0", -1))) for s in SAMPLES]
check("window inside the accepted prefix skipped (2 windows sampled)", len(SAMPLES) == 2, len(SAMPLES))
m1 = vmask(SAMPLES[0]["latent"])
check("released rows inside the accepted prefix release fully", bool((m1[:, :, :10] == 1).all()))
check("accepted rows never rewritten", torch.equal(vid(res[0])[:, :, :22], vid(lat)[:, :, :22]))

print("\n3. stage 2 on the same grid == the existing phase-2 handoff")
pa = patcher("A")
p2 = run(g=guider(pa), sigmas=SIG, phase2_start_step=4,
         phase2_sampler=comfy.samplers.sampler_object("euler"), overlap_strength_audio=1.0)
s2 = run(g=guider(pa), sigmas=SIG[:5], stage2_sigmas=SIG[4:], overlap_strength_audio=1.0)
check("same output, video", torch.equal(vid(p2[0]), vid(s2[0])))
check("same output, audio", torch.equal(aud(p2[0]), aud(s2[0])))
check("stage1_latent == latent on one grid", s2[3] is s2[0])

print("\n4. a separate model for stage 2")
gb = guider(patcher("B", gain=0.41))
res = run(sigmas=SIG[:5], stage2_sigmas=SIG[4:], stage2_guider=gb)
tags = [s["tag"] for s in SAMPLES]
check("A,B per window, in order", tags == ["A", "B"] * 3, tags)
check("stage 2 forwards ran on model B only",
      all(t == "B" for t, o in CALLS if o["sample_sigmas"][0] <= .5 + 1e-6)
      and all(t == "A" for t, o in CALLS if o["sample_sigmas"][0] > .5 + 1e-6))
check("stage 2 got the custom sigmas", torch.equal(SAMPLES[1]["sigmas"], SIG[4:]))
check("both stages of a chunk see its start frame",
      [s["opts"].get("mmh3_control_frame0") for s in SAMPLES] == [0, 0, 34, 34, 68, 68],
      [s["opts"].get("mmh3_control_frame0") for s in SAMPLES])
check("report names the stage", "stage 2: 2 steps from sigma 0.5000, own guider" in res[2], res[2])

print("\n5. upscaler between the stages (stub)")
StubUpscaler.calls = 0
res = run(sigmas=SIG[:5], stage2_sigmas=torch.tensor([.6, .3, 0.0]), stage2_guider=gb,
          upscale_model="stub.safetensors", upscale_megapixels=MP_HI)
out, n_ch, rep, s1 = res
check("final latent on the 4x4 grid", tuple(vid(out).shape[-2:]) == (4, 4), tuple(vid(out).shape))
check("stage1_latent on the 2x2 grid", tuple(vid(s1).shape[-2:]) == (2, 2), tuple(vid(s1).shape))
check("one upscale per chunk", StubUpscaler.calls == 3, StubUpscaler.calls)
check("final latent carries no stage-1-grid mask", "noise_mask" not in out)
st1 = [s for s in SAMPLES if s["tag"] == "A"]
st2 = [s for s in SAMPLES if s["tag"] == "B"]
# stage-2 input = upscaled stage-1 estimate on fresh rows, previous FINAL on pinned rows
x0 = vid(s1)[:, :, 10:32]               # window 1's stage-1 estimate rows survive in s1 from 22 on
lat2 = vid(st2[1]["latent"])
check("stage-2 fresh rows = upscaled stage-1 estimate",
      torch.equal(lat2[:, :, 12:], TS.resize_frames(x0, (4, 4), "nearest-exact")[:, :, 12:]))
check("stage-2 pinned rows = previous chunk's finished rows", torch.equal(lat2[:, :, :12], vid(out)[:, :, 10:22]))
check("stage-2 audio = stage-1 estimate (never resized)",
      aud(st2[1]["latent"]).shape == aud(st1[1]["latent"]).shape)
nz = st2[1]["noise"]
check("fresh stage-2 video noise seeded from the stage-1 seed", isinstance(nz, TS.Stage2Noise)
      and nz.seed == 123 + 1 + TS.STAGE2_SEED_OFFSET, getattr(nz, "seed", None))
check("stage-2 seed wraps below 2**64", TS.stage2_seed((1 << 64) - 1) == TS.STAGE2_SEED_OFFSET - 1)
gen = nz.generate_noise(st2[1]["latent"]).unbind()
first = st1[1]["noise"].generate_noise(st1[1]["latent"]).unbind()
check("stage-2 audio noise == stage-1 audio noise", torch.equal(gen[1], first[1]))
check("stage-2 video noise drawn on the target grid", tuple(gen[0].shape[-2:]) == (4, 4))
check("stage-1 carry = previous stage-1 estimate (estimate mode)",
      torch.equal(vid(st1[1]["latent"])[:, :, :12], vid(s1)[:, :, 10:22]))
m2 = vmask(st2[1]["latent"])
check("stage-2 mask on the target grid", tuple(m2.shape[-2:]) == (4, 4) and bool((m2[:, :, :12] == 0).all()))

res = run(sigmas=SIG[:5], stage2_sigmas=torch.tensor([.6, .3, 0.0]), stage2_guider=gb,
          upscale_model="stub.safetensors", upscale_megapixels=MP_HI, stage1_carry="downscaled_final")
st1 = [s for s in SAMPLES if s["tag"] == "A"]
check("downscaled_final: stage-1 carry = area-downscaled finished rows",
      torch.equal(vid(st1[1]["latent"])[:, :, :12], TS.downscale_video(vid(res[0])[:, :, 10:22], (2, 2))))

lat = master(value=0.5)
keep = torch.ones([1, 1, 42, 2, 2])
keep[:, :, :, :, :1] = 0.0              # left column pinned to the source
lat["noise_mask"] = NestedTensor([keep, torch.ones([1, 1, 2, aud(lat).shape[-1]])])
res = run(latent=lat, sigmas=SIG[:5], stage2_sigmas=torch.tensor([.6, .3, 0.0]), stage2_guider=gb,
          upscale_model="stub.safetensors", upscale_megapixels=MP_HI)
m2 = vmask([s for s in SAMPLES if s["tag"] == "B"][0]["latent"])
check("master pins resized nearest onto the target grid",
      bool((m2[:, :, :, :, :2] == 0).all()) and bool((m2[:, :, :, :, 2:] == 1).all()))
check("pinned source rows come back on the new grid (upscaled source)",
      torch.allclose(vid(res[0])[:, :, :, :, :2], torch.full_like(vid(res[0])[:, :, :, :, :2], 0.5)))

print("\n6. two-grid resume needs the stage-2 master")
lat = master(value=0.5)
pin = torch.ones([1, 1, 42, 2, 2])
pin[:, :, :22] = 0.0                    # window 0 fully pinned (H3ContinueMaster style)
apin = torch.ones([1, 1, 2, aud(lat).shape[-1]])
apin[..., :int(LS._audio_index_at(22, 42, aud(lat).shape[-1]))] = 0.0
lat["noise_mask"] = NestedTensor([pin, apin])
kw = dict(latent=lat, sigmas=SIG[:5], stage2_sigmas=torch.tensor([.6, .3, 0.0]), stage2_guider=gb,
          upscale_model="stub.safetensors", upscale_megapixels=MP_HI)
raises("carry from an unrendered chunk without stage2_latent -> error",
       lambda: run(**kw), "Wire the previous run's final master")
hi = master(value=0.0)
hv = torch.full([1, 24, 42, 4, 4], 0.7)
hi["samples"] = NestedTensor([hv, aud(hi)])
res = run(stage2_latent=hi, **kw)
st2 = [s for s in SAMPLES if s["tag"] == "B"]
check("pinned window skipped, two chunks rendered", len(st2) == 2, len(st2))
check("carried finished rows come from stage2_latent",
      torch.equal(vid(st2[0]["latent"])[:, :, :12], hv[:, :, 10:22]))
check("stage2_latent rows outside rendered chunks kept", torch.equal(vid(res[0])[:, :, :10], hv[:, :, :10]))

print("\n7. custom sigmas and refusals")
res = run(sigmas=SIG[:5], stage2_sigmas=torch.tensor([.3, .1, 0.0]))
check("stage 2 may start below the stage-1 handoff", torch.isfinite(vid(res[0])).all().item())
res = run(sigmas=SIG, stage2_sigmas=torch.tensor([.4, 0.0]))
check("stage 1 may finish (refine pass)", torch.isfinite(vid(res[0])).all().item())
raises("non-descending stage2_sigmas", lambda: run(stage2_sigmas=torch.tensor([.5, .6, 0.0])), "strictly descend")
raises("stage2_sigmas not ending at 0", lambda: run(stage2_sigmas=torch.tensor([.5, .2])), "must end at 0")
raises("stage2_sigmas above 1", lambda: run(stage2_sigmas=torch.tensor([1.5, 0.0])), "<= 1")
raises("upscaler without stage 2", lambda: run(upscale_model="stub.safetensors"), "needs stage")
raises("tail_custom with keyframe carry",
       lambda: LS.MMH3LoopingSampler.execute(
           Noise_RandomNoise(1), guider(patcher("A")), comfy.samplers.sampler_object("euler"), SIG,
           {"conds": [prompt(0)]}, master(), CHUNK, OV, "keyframe", 1.0, 1.0,
           overlap_pin="tail_custom"), "carry to mask")
raises("tail_custom with pin_renorm", lambda: run(overlap_pin="tail_custom", pin_renorm_video=0.5),
       "set them to 0")
raises("upscale grid not larger", lambda: run(sigmas=SIG[:5], stage2_sigmas=SIG[4:],
                                              upscale_model="stub.safetensors", upscale_megapixels=0.0009),
       "not larger")

print("\n8. re-roll redraws both stages")
res = run(sigmas=SIG[:5], stage2_sigmas=torch.tensor([.6, .3, 0.0]), stage2_guider=gb,
          upscale_model="stub.safetensors", upscale_megapixels=MP_HI, reroll_retries=1,
          reroll_thresholds="luma=1e-9,hf=1e-9,cmean=1e-9")
st1 = [s for s in SAMPLES if s["tag"] == "A"]
st2 = [s for s in SAMPLES if s["tag"] == "B"]
seeds1 = [s["noise"].seed for s in st1]
seeds2 = [s["noise"].seed for s in st2]
check("chunks 1-2 sampled twice in each stage", len(st1) == 5 and len(st2) == 5, (len(st1), len(st2)))
check("stage-1 seeds move per attempt", seeds1 == [123, 124, 1124, 125, 1125], seeds1)
check("stage-2 seeds follow the attempt",
      seeds2 == [s + TS.STAGE2_SEED_OFFSET for s in seeds1], seeds2)
check("kept attempt's stage-1 estimate lands in stage1_latent", "reroll kept attempt" in res[2])

print("\n9. defaults untouched")
a = run()
b = run(overlap_pin="full", context_frames=5, accepted_prefix_frames=0, stage2_guider=gb,
        stage2_sampler=None, stage2_sigmas=None, upscale_model="none", stage1_carry="estimate")
check("unused stage-2 inputs change nothing", torch.equal(vid(a[0]), vid(b[0])) and torch.equal(aud(a[0]), aud(b[0])))
check("one sample per chunk, main model only", [s["tag"] for s in SAMPLES] == ["A"] * 3)
check("no tap span without tail_custom", all("h3_tap_context_rows" not in s["opts"] for s in SAMPLES))

print("\n10. guide_video in both stages")


class GuideVAE:
    """Encodes a frame batch to its mean, so a guide's value names the frames it holds."""
    def __init__(self):
        self.seen = []

    def encode(self, image):
        self.seen.append(tuple(image.shape))
        from comfy_extras.nodes_minimax_h3 import video_latent_t
        t = 1 if len(image) == 1 else video_latent_t(len(image))
        return torch.full((1, 24, t, image.shape[1] // 16, image.shape[2] // 16), float(image.mean()))


def frame_ramp(count=FRAMES):
    return (torch.arange(count, dtype=torch.float32) / 255)[:, None, None, None].expand(-1, 32, 32, 3)


def kf_of(sample):
    # core's convert_cond stores each entry as one flat dict
    return [kf for d in sample["positive"] for kf in d.get("minimax_keyframes", [])]


gv = GuideVAE()
res = run(sigmas=SIG[:5], stage2_sigmas=torch.tensor([.6, .3, 0.0]), stage2_guider=gb,
          upscale_model="stub.safetensors", upscale_megapixels=MP_HI, guide_video=frame_ramp(), vae=gv)
st1 = [s for s in SAMPLES if s["tag"] == "A"]
st2 = [s for s in SAMPLES if s["tag"] == "B"]
spans = [(0, 73), (34, 107), (68, 141)]
check("stage 1: one guide per chunk on the 2x2 grid",
      all(len(kf_of(s)) == 1 and tuple(kf_of(s)[0]["latent"].shape[-3:]) == (22, 2, 2) for s in st1))
check("stage 2: the same span re-encoded on the 4x4 grid",
      all(len(kf_of(s)) == 1 and tuple(kf_of(s)[0]["latent"].shape[-3:]) == (22, 4, 4) for s in st2))
check("both stages' guides hold their chunk's own frames",
      all(abs(float(kf_of(s)[0]["latent"].mean()) - (f0 + f1 - 1) / 2 / 255) < 1e-5
          for st in (st1, st2) for s, (f0, f1) in zip(st, spans)))
check("the VAE saw 32 px and 64 px frames", {sh[1] for sh in gv.seen} == {32, 64}, sorted(set(gv.seen)))
gv = GuideVAE()
run(sigmas=SIG[:5], stage2_sigmas=SIG[4:], guide_video=frame_ramp(), vae=gv)
check("one grid: stage 2 reuses the stage-1 guide (no second encode)", len(gv.seen) == 3, len(gv.seen))
check("one grid: stage 2 carries the guide too",
      all(len(kf_of(s)) == 1 for s in SAMPLES if s["sigmas"][0] <= .5 + 1e-6))

print("\n11. stage2_latent never pins v2v keeps")
lat = master(value=0.5)
keep = torch.ones([1, 1, 42, 2, 2])
keep[:, :, :, :, :1] = 0.0
lat["noise_mask"] = NestedTensor([keep, torch.ones([1, 1, 2, aud(lat).shape[-1]])])
blank = master(value=0.0)
blank["samples"] = NestedTensor([torch.zeros([1, 24, 42, 4, 4]), aud(blank)])
res = run(latent=lat, sigmas=SIG[:5], stage2_sigmas=torch.tensor([.6, .3, 0.0]), stage2_guider=gb,
          upscale_model="stub.safetensors", upscale_megapixels=MP_HI, stage2_latent=blank)
check("keep regions come from the upscaled source, not a zero stage2_latent",
      torch.allclose(vid(res[0])[:, :, :, :, :2], torch.full_like(vid(res[0])[:, :, :, :, :2], 0.5)))

print("\nFAILURES: %s" % fails if fails else "\nall passed")
sys.exit(1 if fails else 0)
