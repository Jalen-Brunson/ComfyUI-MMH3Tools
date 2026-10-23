# Context frames and a second stage per chunk

`MMH3LoopingSampler` still renders the master chunk by chunk, each chunk finished before
the next starts. Two optional additions sit on the same loop. Every new input is appended
after `feather_latents`, so saved graphs keep their widget values. With the defaults the
output is byte-identical to the previous sampler; this was checked on nine configurations,
including masks, phase 2, a partial schedule, feather and the re-roll guard.

## Context frames (`overlap_pin = tail_custom`)

The masked carry normally pins every carried row. With `tail_custom`, only the last
`context_frames` of the carry stay pinned:

- The value is rounded **up** to whole H3 latent rows. Each 17 frames are rows of 1-4-4-4-4.
- With a 39-frame overlap, 5, 9, 13, 17, 22 and 39 are exact. 6 becomes 9.
- 0 releases every carried video row.
- Any value at or above the overlap behaves exactly like `full`.

```
next chunk:  | released carry | pinned | new |
             | redrawn, then  | held   |     |
             |   discarded    |        |     |
```

- **Released rows** go back to the master's own mask, so a v2v keep region stays kept.
  Rows inside `accepted_prefix_frames` are released fully.
- **Write-back** starts at the previous chunk's end, so frames that are already rendered
  never change.
- **Audio** carry is unchanged (`overlap_strength_audio`, 0.9 by default).
- **Report** lines show the effective rows per chunk, for example
  `context 2 of 12 carried latents pinned (5 frames), 10 redrawn`.
- **Preview tap:** the sampler publishes `transformer_options["h3_tap_context_rows"] = (released, carried)`.
  H3ChunkPreviewTap's context reset then grades only the pinned rows, decoded from the
  start of their 17-frame group.

## Stage 2 (`stage2_sigmas` wired)

Each chunk finishes with a second sampler pass:

| input | use |
|---|---|
| `stage2_sigmas` | Turns stage 2 on. Strictly descending, at most 1, ending at 0. |
| `stage2_guider` | Stage 2's MODEL, CFG and negative. Its positive is replaced per chunk. Unwired, the main guider is used. |
| `stage2_sampler` | Unwired, the main sampler is used. |

**Handoff.** Stage 1 runs the main `sigmas` and returns its **denoised** estimate. Stage 2
re-noises that estimate at `stage2_sigmas[0]`, which is the looping sampler's phase-2
handoff. Because it starts from a clean estimate, any schedule works: it can start below
stage 1's last sigma, or after a finished stage 1 (a refine pass). It does **not** continue
stage 1's solver state; this is deliberate, and it differs from the hybrid sampler's
joint finish.

**Carry and audio in stage 2:**

- Carry rows are the previous chunk's **finished** stage-2 rows.
- The audio carry is anchored on the original rows, not on stage 1's partly re-estimated copy.

**Patches.** Model patches stage 2 needs (LoRA, PDD, SigmaShift, control schedules,
attention) must also be applied to the stage-2 model.

**Re-roll guard.** It re-runs stage 1, the upscale and stage 2, and scores the finished
chunk. Stage 2's noise follows the attempt's seed, so a retry redraws both stages.

## Learned upscale between the stages (`upscale_model`)

This uses the MiniMax H3 latent upscaler node (`MinimaxH3LatentUpscaler3D`, files in
`models/latent_upscale_models`). It requires stage 2.

**Grid.** `upscale_megapixels` sets the stage-2 grid: 1024² pixels per MP, sides snapped
to 32 px, aspect kept. It must be larger than the master.

**Video handoff.** Each chunk's stage-1 estimate is upscaled, and stage 2 starts from
`σ0·ε + (1−σ0)·U(x0)` with **fresh** video noise drawn on the new grid. Reusing the stage-1
noise with a nearest resize leaves it spatially correlated, which the hybrid sampler
measured as a corrupted high-res finish.

**Audio** is never resized. It keeps stage 1's noise.

**Masks.** The master's pins are resized nearest-exact onto the new grid. Rows the master
pins (v2v keeps) come back from stage 1 as the source itself, so stage 2 receives them
upscaled. `stage2_latent` is never used for them: a resume master built without the
source holds zeros there, which would pin black.

**Guides.** `guide_video` (0.105.0) is encoded per chunk on each stage's grid. Stage 1
gets the chunk's span at the master's size. Stage 2 gets the same span encoded again at
the upscaled size, because a guide's rows have to share the grid being sampled. On one
grid, stage 2 reuses stage 1's guide.

**Stage-1 carry** (`stage1_carry`):

- `estimate`: the previous chunk's stage-1 estimate, with no round trip.
- `downscaled_final`: its finished rows, area-downscaled.

**Outputs.**

- `latent` is the master on the stage-2 grid.
- `stage1_latent` is the stage-1 grid's master of estimates. On one grid it is the same as `latent`.

**Resume and accepted frames on two grids:**

- Wire the previous run's final master, on the stage-2 grid and cut to this run's span,
  to `stage2_latent`. For example, run the saved high-res master through an
  H3 Continue Master with the same resume settings as the stage-1 master.
- A chunk that carries rows this run did not render stops with an error instead of
  upscaling a guess in their place.
- `latent` is the stage-1 grid master. To keep `stage1_carry = estimate`, resume it from
  a saved `stage1_latent`; `downscaled_final` needs only `stage2_latent`.

## Recipe: V2V Upscale on the looping sampler

The example graph is built by `/workspace/scripts/build_two_stage_looping_flow.py` and saved as
`MMH3 Two-Stage Looping - PDD upscale DMAD.json`. All three stages share the Ref2VA int8
trunk with SigmaShift 12/3.

| stage | model | sigmas | grid |
|---|---|---|---|
| Stage 1 | trunk + PDD Acc 8-step (`MiniMaxH3PDDAccApply`) | PDD sigmas, SplitSigmas step 7 (ends at 12/19) | 864×480 |
| Upscale | `h3_upscaler_sharpness_2000steps_v0.1_fp32` | 1 MP, fp32 | — |
| Stage 2 | trunk + DMAD 4-step LoRA 0.8 | ManualSigmas `0.631578922, 0.3158, 0` | 1376×768 |

Sampler settings: chunks 243 / overlap 39, `tail_custom` 5, audio carry 0.9.

## Costs

- Models A and B are swapped every chunk, and LoRA or PDD patches are re-applied each time.
- Two different full checkpoints will not both stay resident on an 80 GB card.
- The upscaler runs once per chunk.

## Not combined

- `carry = keyframe`.
- `pin_renorm`, `carry_precomp` and `carry_noise`. They assume the whole head is pinned on one grid.
- The hidden `prior_av_latent` path, with stage 2.

## Tests

`python tests/test_two_stage.py` runs 61 checks through core's real sampling path on a
tiny CPU H3:

- context rows, masks, discard and accepted rows;
- stage 2 matching the existing phase-2 handoff exactly on one grid;
- model B used only in stage 2;
- the upscale handoff, noise, masks and both carry modes;
- the resume error and `stage2_latent`;
- schedule validation and re-roll seeds;
- defaults unchanged.

Rendered quality is untested.
