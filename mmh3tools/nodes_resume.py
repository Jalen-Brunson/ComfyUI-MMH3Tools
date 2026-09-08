"""Associate completed videos with their exact masters for automatic resume."""

import hashlib
import json
import os
from pathlib import Path
import time

import torch

import folder_paths
from comfy.nested_tensor import NestedTensor
from comfy_api.latest import io
from comfy_extras.nodes_video import SaveVideo

from .common import latents_to_frames

AUTO = "auto (latest completed for this project)"


def _index_directory():
    return Path(folder_paths.get_output_directory()) / "latents" / "h3_resume_runs"


def completed_runs():
    directory = _index_directory()
    records = []
    if directory.exists():
        for path in directory.glob("*.json"):
            record = json.loads(path.read_text())
            if Path(record["master"]).is_file() and Path(record["video"]).is_file():
                records.append(record)
    return sorted(records, key=lambda r: r["completed_at_ns"], reverse=True)


def record_run(project, variant, master_path, video_path, frames, fps, width, height,
               completed_at_ns=None):
    master = Path(master_path).resolve()
    video = Path(video_path).resolve()
    if not master.is_file() or not video.is_file():
        raise FileNotFoundError("Both the master and video must finish saving before this run can be resumed.")
    # Read metadata/shapes without loading the tensor storage into RAM.
    saved = torch.load(master, map_location="cpu", weights_only=True, mmap=True)
    if int(saved.get("start_chunk", 0)) != 0:
        raise ValueError("Automatic prefix resume requires a full master starting at chunk 0.")
    master_frames = latents_to_frames(saved["video"].shape[2])
    if master_frames != int(frames):
        raise ValueError(f"Master/video length mismatch: {master_frames} versus {frames} frames.")
    record = {
        "project": str(project), "variant": str(variant or ""),
        "master": str(master), "video": str(video),
        "frames": int(frames), "fps": float(fps),
        "width": int(width), "height": int(height),
        "chunk_frames": int(saved.get("chunk_frames", 0)),
        "overlap_frames": int(saved.get("overlap_frames", 0)),
        "completed_at_ns": int(completed_at_ns if completed_at_ns is not None else time.time_ns()),
    }
    directory = _index_directory()
    directory.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(str(master).encode()).hexdigest()
    destination = directory / f"{key}.json"
    temporary = directory / f"{key}.json.tmp"
    temporary.write_text(json.dumps(record, indent=2))
    os.replace(temporary, destination)
    return record


def resolve_run(project, variant, master):
    records = [r for r in completed_runs()
               if r["project"] == project and r["variant"] == (variant or "")]
    if master == AUTO:
        if records:
            return records[0]
        raise ValueError("No completed run exists for this project and prompt variant. Set RESUME C to 0 for the first run.")
    selected = str(Path(master).resolve())
    for record in records:
        if record["master"] == selected:
            return record
    raise ValueError("The selected master has no completed video pair for this project and prompt variant. Choose auto or a matching saved run.")


class MMH3LoadResumeRun(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3LoadResumeRun", display_name="MMH3 Resume Run",
            category="MMH3Tools/latent",
            description="Load the latest completed master and its matching video for this project, or select an older master.",
            inputs=[
                io.String.Input("project", force_input=True),
                io.Int.Input("resume_from", default=0, min=0, force_input=True),
                io.Combo.Input("master", options=[AUTO] + [r["master"] for r in completed_runs()], default=AUTO,
                               tooltip="Auto selects the latest completed run for the wired project/variant. Refresh ComfyUI to list newly saved masters."),
                io.String.Input("variant", default="", optional=True, force_input=True),
            ],
            outputs=[io.Latent.Output("latent"), io.Int.Output("latent_t"),
                     io.Int.Output("frames"), io.Int.Output("start_chunk"),
                     io.String.Output("master_path"), io.String.Output("video_path"),
                     io.String.Output("report")],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")

    @classmethod
    def execute(cls, project, resume_from, master=AUTO, variant=""):
        if int(resume_from) == 0:
            return io.NodeOutput(None, 0, 0, 0, "", "", "Fresh run; no saved master or video loaded.")
        record = resolve_run(project, variant, master)
        saved = torch.load(record["master"], map_location="cpu", weights_only=True)
        video, audio = saved["video"].float(), saved["audio"].float()
        meta = {k: int(saved[k]) for k in ("start_chunk", "chunk_frames", "overlap_frames") if k in saved}
        latent = {"samples": NestedTensor([video, audio]), "h3_meta": meta}
        report = (f"Resume {project}" + (f" / {variant}" if variant else "") +
                  f": {record['frames']} saved frames; accepting {int(resume_from)} chunks.\n"
                  f"Master: {record['master']}\nVideo: {record['video']}")
        return io.NodeOutput(latent, int(video.shape[2]), record["frames"],
                             int(saved.get("start_chunk", 0)), record["master"], record["video"], report)


class MMH3SaveResumeVideo(SaveVideo):
    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "MMH3SaveResumeVideo"
        schema.display_name = "MMH3 Save Video and Resume Pair"
        schema.description = "Save the video normally, then remember its exact master for automatic resume."
        schema.category = "MMH3Tools/latent"
        schema.inputs.extend([
            io.String.Input("master_path", force_input=True),
            io.String.Input("project", force_input=True),
            io.String.Input("variant", default="", optional=True, force_input=True),
        ])
        schema.outputs.extend([io.String.Output("master_path"), io.String.Output("video_path"), io.String.Output("report")])
        return schema

    @classmethod
    def execute(cls, video, filename_prefix, format, master_path, project, codec=None, variant=""):
        saved = super().execute(video, filename_prefix, format, codec)
        result = saved.ui.as_dict()["images"][0]
        video_path = Path(folder_paths.get_output_directory()) / result["subfolder"] / result["filename"]
        width, height = video.get_dimensions()
        record = record_run(project, variant, master_path, video_path,
                            video.get_frame_count(), video.get_frame_rate(), width, height)
        report = f"Saved {record['frames']} frames. This master/video pair is available for auto resume."
        return io.NodeOutput(video, record["master"], record["video"], report, ui=saved.ui)
