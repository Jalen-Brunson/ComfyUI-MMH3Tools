import os, sys
sys.path.insert(0, r"C:\ComfyUI")
sys.path.insert(0, r"C:\ComfyUI\custom_nodes\ComfyUI-MMH3Tools")
import logging; logging.disable(logging.CRITICAL)
import torch
from comfy.nested_tensor import NestedTensor
from mmh3tools import nodes_looping_sampler as LS
from mmh3tools.common import (frames_to_latents, frames_to_audio_t,
                              latents_to_frames, VIDEO_T_DIM, AUDIO_T_DIM)

H = W = 4
SEEN = []


class FakeSampler:
    def sample(self, noise, guider, sampler, sigmas, latent):
        v, a = latent["samples"].unbind()
        SEEN.append(int(v.shape[VIDEO_T_DIM]))
        o = {"samples": NestedTensor([torch.full_like(v, float(len(SEEN))), a])}
        return o, o


LS.SamplerCustomAdvanced = FakeSampler


class G:
    def __init__(s): s.original_conds = {"positive": "P", "negative": "N"}
    def set_conds(s, p, n=None): s.original_conds["positive"] = p


class Noise:
    seed = 1000


def clip(frames, fill=0.0):
    t = frames_to_latents(frames)
    at = frames_to_audio_t(latents_to_frames(t))
    return ({"samples": NestedTensor([torch.full([1, 24, t, H, W], fill),
                                      torch.full([1, 32, 2, at], fill)])}, t, at)


cond = lambda tag: [[torch.zeros([1, 8]), {"tag": tag}]]
fails = []


def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label + "  got=%s want=%s" % (got, want))
    if not ok:
        fails.append(label)


def run(prior_frames, new_frames, chunk=481, ov=22):
    SEEN.clear()
    prior, pt, pat = clip(prior_frames, fill=9.0) if prior_frames else (None, 0, 0)
    new, nt, nat = clip(new_frames)
    out = LS.MMH3LoopingSampler.execute(
        Noise(), G(), "S", torch.linspace(1, 0, 13),
        {"conds": [cond("p%d" % i) for i in range(20)]},
        new, chunk, ov, "mask", 1.0, 1.0, 0, 1000, 0, None, None, None, "", None,
        prior).result[:3]
    return out, pt, nt


print("\n1. output = prior + new, padded up to the latent grid")
# prior (5a+2) + new (5b+2) is 5k+4, which is NOT a valid clip. The new side is
# padded to make the total 5j+2; without that the tail falls outside every window
# and is never sampled. At most 4 latents, never fewer than asked for.
for pf, nf in ((481, 1920), (192, 1920), (1000, 2400), (57, 960)):
    (o, n, rep), pt, nt = run(pf, nf)
    v, a = o["samples"].unbind()
    ot = int(v.shape[VIDEO_T_DIM])
    check("prior %4df + new %4df -> %d latents, on grid" % (pf, nf, ot), ot % 5, 2)
    check("   never short, padded by <5", pt + nt <= ot < pt + nt + 5, True)
    check("   audio sized to the video",
          int(a.shape[AUDIO_T_DIM]), frames_to_audio_t(latents_to_frames(ot)))
    check("   every generated latent was written",
          bool((v[0, 0, pt:, 0, 0] != 0.0).all()), True)

print("\n2. the prior survives verbatim -- ALL of it, carried tail included")
# The first generated chunk OVERLAPS the prior, so the write-back lands on its tail.
# An earlier version of this test checked only up to pt-7 and so never noticed that
# the tail was being altered -- at overlap_strength_audio 0.9 it certainly was.
(o, n, rep), pt, nt = run(481, 1920)
v, a = o["samples"].unbind()
pat = frames_to_audio_t(latents_to_frames(pt))
check("every prior video latent still 9.0", bool((v[0, 0, :pt, 0, 0] == 9.0).all()), True)
check("including the very last one", float(v[0, 0, pt - 1, 0, 0]), 9.0)
check("every prior audio latent still 9.0", bool((a[0, 0, 0, :pat] == 9.0).all()), True)
check("the region after it was generated", bool((v[0, 0, pt:, 0, 0] != 9.0).any()), True)

# and it must hold when the carry is NOT fully pinned -- 0.9 is the audio default
SEEN.clear()
prior2, pt2, _ = clip(481, fill=9.0)
new2, _, _ = clip(1920)
o2 = LS.MMH3LoopingSampler.execute(
    Noise(), G(), "S", torch.linspace(1, 0, 13), {"conds": [cond("p")]},
    new2, 481, 22, "mask", 0.5, 0.9, 0, 1000, 0, None, None, None, "", None,
    prior2).result[0]
v2, a2 = o2["samples"].unbind()
check("prior intact at strengths 0.5 / 0.9 too",
      bool((v2[0, 0, :pt2, 0, 0] == 9.0).all()), True)

print("\n3. chunk 0 carries the prior's tail")
check("report names the prior", "kept verbatim" in rep, True)
check("chunk 0 is not 0 carried", "chunk 0: prompt 0" in rep and ", 0 carried" not in
      [l for l in rep.splitlines() if "chunk 0:" in l][0], True)

print("\n4. prompts start at cond 0 on the first GENERATED chunk")
first = [l for l in rep.splitlines() if l.strip().startswith("chunk 0:")][0]
check("chunk 0 uses prompt 0", "prompt 0" in first, True)

print("\n5. no prior behaves exactly as before")
(o, n, rep), pt, nt = run(0, 3048)
check("7 chunks for a 127s clip", n, 7)
check("no prior line in the report", "kept verbatim" in rep, False)
check("chunk 0 carries nothing", ", 0 carried" in rep, True)

print("\n6. odd prior lengths do not break the grid")
for pf in (57, 124, 192, 311, 481, 900):
    (o, n, rep), pt, nt = run(pf, 1440)
    v, _ = o["samples"].unbind()
    ot = int(v.shape[VIDEO_T_DIM])
    check("prior %4df -> %d latents out, %d chunks, on grid" % (pf, ot, n), ot % 5, 2)
    check("   prior intact and region covered",
          bool((v[0, 0, :pt, 0, 0] == 9.0).all()) and bool((v[0, 0, pt:, 0, 0] != 0.0).all()),
          True)

print("\n7. mismatched shapes are refused")
new, _, _ = clip(1920)
bad = {"samples": NestedTensor([torch.zeros([1, 24, 32, 8, 8]),
                                torch.zeros([1, 32, 2, 40])])}
try:
    LS.MMH3LoopingSampler.execute(
        Noise(), G(), "S", torch.linspace(1, 0, 13), {"conds": [cond("p")]},
        new, 481, 22, "mask", 1.0, 1.0, 0, 1000, 0, None, None, None, "", None, bad)
    check("frame-size mismatch raises", False, True)
except ValueError as e:
    check("frame-size mismatch raises", "frame" in str(e), True)


print("\n8. chunk_frames=0 -- one chunk, therefore one prompt, always")
# The size is region + carry + grid padding, and being one grid step short silently
# costs a second chunk and a second prompt. The node works it out instead.
for pf in (72, 120, 240, 480, 1128):
    for nf in (39, 56, 107, 192, 226, 481):
        SEEN.clear()
        prior, pt, _ = clip(pf, fill=9.0)
        new, nt, _ = clip(nf)
        o, n, rep = LS.MMH3LoopingSampler.execute(
            Noise(), G(), "S", torch.linspace(1, 0, 13), {"conds": [cond("p")]},
            new, 0, 22, "mask", 1.0, 0.9, 0, 1000, 0, None, None, None, "", None,
            prior).result[:3]
        v, _ = o["samples"].unbind()
        ot = int(v.shape[VIDEO_T_DIM])
        good = (n == 1 and ot % 5 == 2
                and bool((v[0, 0, :pt, 0, 0] == 9.0).all())
                and bool((v[0, 0, pt:, 0, 0] != 0.0).all())
                and pt + nt <= ot < pt + nt + 5)
        if not good:
            check("prior %df + add %df" % (pf, nf), (n, ot), "1 chunk, covered, on grid")
check("every prior x addition pair gives exactly one chunk", True, True)

# and with no prior it means one chunk over the whole clip
SEEN.clear()
new, _, _ = clip(1920)
n0 = LS.MMH3LoopingSampler.execute(
    Noise(), G(), "S", torch.linspace(1, 0, 13), {"conds": [cond("p")]},
    new, 0, 22, "mask", 1.0, 0.9, 0, 1000, 0, None, None, None, "", None,
    None).result[1]
check("no prior, chunk_frames=0 -> one chunk", n0, 1)

print("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)
