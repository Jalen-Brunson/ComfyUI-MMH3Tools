# Hybrid sampling and accepted-prefix resume

`MMH3HybridWindowSampler` starts each window sequentially, then finishes the
generation using predictions from all overlapping windows. It is experimental:
long-run mottling and visible continuation boundaries remain possible.

![How the H3 hybrid sampler works](images/h3-hybrid-sampler.jpg)

## How the 6/2 example works

1. Run Euler steps 1–6 on the first window. Use its provisional clean prediction
   to anchor the overlap while running steps 1–6 on the next window. Repeat along
   the timeline. Keep the actual unfinished, float32 noisy state separately.
2. Resume that noisy state at step 7. Evaluate each window, blend its predictions
   with neighboring predictions in the overlap, then update one shared state.
   Repeat for step 8. Generated carry becomes editable; original source masks
   stay fixed. Control and reference conditioning remain active in both phases.
3. Decode the final latent. Prediction fusion happens during inference.

Windows are evaluated one at a time; “joint” describes the shared update.
The switch does not restart the generation or draw fresh noise. The first window
to cover a position owns its initial solver state and fixed noise.

Use plain Euler without churn or random inpaint noise, CFG 1, and a finite,
strictly descending sigma schedule ending at zero. `sequential_steps` can be any
integer from 1 through the number of scheduled steps. Setting it to the full
count gives a sequential baseline. The 6/2 split is an experiment, not a PDD
requirement; another speed LoRA still needs its own compatible schedule and tests.

`MMH3JointWindowSampler` runs the whole schedule with shared window updates. Both
nodes require matching window/overlap settings and one conditioning set per
window. Connect the model before other Context Windows nodes. H3-aligned frame
counts and window starts are required; keyframe guides are not supported here.

## Resumable V2V Character Replacement

This workflow replaces a masked character in an existing video while retaining
the source video's timing, motion, camera path, and unmasked content. It is built
for long replacements that need to be generated and reviewed a few overlapping
windows at a time. Accepted-prefix resume lets an artist approve the beginning of
the replacement, keep it, and extend the result without generating those accepted
chunks again.

The end-to-end flow is:

1. Load the source video, the character replacement mask, character references,
   and prompts. The mask determines which source regions may be regenerated and
   which remain fixed.
2. For a fresh run, choose how many windows to generate. The hybrid sampler starts
   each window sequentially to establish motion, then jointly finishes the
   overlapping windows so neighboring predictions can inform their shared frames.
3. Review the rendered result and choose how many complete opening chunks to
   accept. The save step records the full latent master together with its matching
   rendered video.
4. To continue, choose the number of accepted chunks and the number of new chunks
   to add. The workflow loads the latest completed master/video pair for the same
   project and prompt variant. An older registered pair can be selected explicitly.
5. The last accepted window is supplied as pinned context while the sampler
   generates the next source-aligned windows. Earlier accepted windows do not run
   through the sampler again.
6. The workflow restores the accepted latent prefix exactly, merges the newly
   generated latents into a new full master, keeps the accepted rendered frames,
   appends only the newly decoded frames, and registers the new master/video pair
   for a later continuation.

This is a resume between completed generations, not recovery of an interrupted
denoising step. The chunk grid, overlap, resolution, frame rate, and source
timeline must remain unchanged between runs.

### Resume controls and saved pairs

The hybrid sampler's optional `accepted_prefix_frames` fixes the beginning of
its input, including audio, through both phases. Fully accepted windows skip
sequential sampling but remain available as joint context. `start_window` keeps
per-window seed offsets on the original timeline. Defaults of zero mean a fresh
run. This resumes finished chunks, not an interrupted denoising step.

`MMH3LoadResumeRun` selects a completed master/video pair by project and prompt
variant. Leave `master` on **auto (latest completed for this project)**, or select
an older registered master; its paired video follows automatically. With
`resume_from=0`, no previous output is required. Auto refreshes its selection on
each execution and does not choose how many chunks to accept.

`MMH3SaveResumeVideo` wraps native `SaveVideo`. Wire the full master's saved path,
project, and variant to it. After the video saves successfully, it records the
exact pair under `output/latents/h3_resume_runs/`. The master must start at chunk
zero and describe the same frame count as the video. Missing files and
unregistered pairs are not selected automatically. Pre-existing outputs need an
explicit pair record; matching filenames alone are not sufficient.

**Resumable V2V Character Replacement** supplies timeline slicing and assembly
through `H3ContinueMaster`, `H3MergeMasters`, and `H3SaveAVLatent` from its separate
workflow helpers. The sampler and pair-selection nodes do not replace those
helpers. The saved latent prefix is exact; normal MP4 re-encoding does not
preserve the original compressed bytes or decoded pixels bit for bit.

## Validation

Run these scripts in a ComfyUI Python environment. For a checkout outside
`custom_nodes`, set `PYTHONPATH` to the ComfyUI directory:

```bash
python tests/test_hybrid_sampler.py
python tests/test_joint_sampler.py
python tests/test_resume_runs.py
```

The 20 CPU checks cover actual Euler/inpaint state continuation, window
conditioning and AV fusion, accepted-prefix preservation, global seed offsets,
and completed-pair save/load behavior. Stock-video runtime checks also verified
exact accepted latent values and automatic selection advancing after a new save.
Those checks establish mechanics, not long-generation image quality.
