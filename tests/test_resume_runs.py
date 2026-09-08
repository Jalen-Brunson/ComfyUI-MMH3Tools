"""Completed-pair selection, failure handling and native SaveVideo integration."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
                str(Path(__file__).resolve().parents[3])]
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options
comfy.options.enable_args_parsing()
import torch
from comfy_api.latest import io, ui
from comfy_extras.nodes_video import SaveVideo
from mmh3tools import nodes_resume as resume

torch.set_num_threads(2)


class ResumeRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.directory = patch.object(resume.folder_paths, "get_output_directory", return_value=self.temp.name)
        self.directory.start()

    def tearDown(self):
        self.directory.stop()
        self.temp.cleanup()

    def pair(self, name, project="project-a", variant="", stamp=1, record=True):
        master, video = self.root / (name + ".pt"), self.root / (name + ".mp4")
        torch.save({"video": torch.full((1,24,37,2,2), float(stamp)).half(),
                    "audio": torch.ones(1,32,2,207).half(), "start_chunk": 0,
                    "chunk_frames": 124, "overlap_frames": 39}, master)
        video.write_bytes(b"finished video stand-in")
        if record:
            resume.record_run(project, variant, master, video, 124, 24, 32, 32, stamp)
        return str(master), str(video)

    def test_auto_is_project_variant_scoped_and_tracks_new_runs(self):
        old = self.pair("a-old", stamp=10)
        self.pair("b-new", project="project-b", stamp=30)
        self.pair("a-variant", variant="alternate", stamp=40)
        self.assertEqual(resume.resolve_run("project-a", "", resume.AUTO)["master"], old[0])
        new = self.pair("a-new", stamp=20)
        self.assertEqual(resume.resolve_run("project-a", "", resume.AUTO)["master"], new[0])
        self.assertEqual(resume.resolve_run("project-a", "", old[0])["video"], old[1])
        with self.assertRaisesRegex(ValueError, "project and prompt variant"):
            resume.resolve_run("project-b", "", old[0])

    def test_incomplete_and_missing_pairs_are_not_auto_selected(self):
        old = self.pair("complete", stamp=1)
        self.pair("unregistered", stamp=20, record=False)
        missing = self.pair("deleted-video", stamp=30)
        Path(missing[1]).unlink()
        self.assertEqual(resume.resolve_run("project-a", "", resume.AUTO)["master"], old[0])
        with self.assertRaises(ValueError):
            resume.resolve_run("project-a", "", missing[0])
        with self.assertRaises(FileNotFoundError):
            resume.record_run("project-a", "", missing[0], missing[1], 124, 24, 32, 32)

    def test_fresh_needs_no_prior_and_load_preserves_exact_metadata(self):
        out = resume.MMH3LoadResumeRun.execute("new-project", 0)
        self.assertIsNone(out[0])
        self.assertEqual(out[4:6], ("", ""))
        master, video = self.pair("a")
        out = resume.MMH3LoadResumeRun.execute("project-a", 1)
        self.assertEqual(out[1:6], (37, 124, 0, master, video))
        self.assertEqual(out[0]["h3_meta"], {"start_chunk": 0, "chunk_frames": 124, "overlap_frames": 39})
        expected = torch.load(master, weights_only=True)
        for key, tensor in zip(("video", "audio"), out[0]["samples"].unbind()):
            torch.testing.assert_close(tensor, expected[key].float(), rtol=0, atol=0)

    def test_mismatched_lengths_and_partial_masters_are_rejected(self):
        master, video = self.pair("unpaired", record=False)
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            resume.record_run("project-a", "", master, video, 209, 24, 32, 32)
        data = torch.load(master, weights_only=True)
        data["start_chunk"] = 1
        torch.save(data, master)
        with self.assertRaisesRegex(ValueError, "full master"):
            resume.record_run("project-a", "", master, video, 124, 24, 32, 32)
        self.assertEqual(resume.completed_runs(), [])

    def test_save_registers_exact_native_output_only_after_success(self):
        master, path = self.pair("saved", record=False)
        video = SimpleNamespace(get_dimensions=lambda: (32,32), get_frame_count=lambda:124,
                                get_frame_rate=lambda:24)
        native = io.NodeOutput(video, ui=ui.PreviewVideo([ui.SavedResult(Path(path).name, "", io.FolderType.output)]))
        with patch.object(SaveVideo, "execute", return_value=native) as save:
            out = resume.MMH3SaveResumeVideo.execute(video, "prefix", "auto", master, "project-a")
        save.assert_called_once_with(video, "prefix", "auto", None)
        self.assertEqual(out[1:3], (master, path))
        self.assertIs(out.ui, native.ui)
        self.assertEqual(resume.resolve_run("project-a", "", resume.AUTO)["video"], path)
        newer, _ = self.pair("failed", record=False)
        with patch.object(SaveVideo, "execute", side_effect=OSError("encoder failed")):
            with self.assertRaises(OSError):
                resume.MMH3SaveResumeVideo.execute(video, "prefix", "auto", newer, "project-a")
        self.assertEqual(resume.resolve_run("project-a", "", resume.AUTO)["master"], master)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
