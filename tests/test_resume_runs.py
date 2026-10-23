"""Completed-pair selection, failure handling and native SaveVideo integration."""
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), "/workspace/ComfyUI"]
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

    def test_cached_metadata_notices_replaced_record_and_deleted_pair(self):
        old=self.pair('old',stamp=1)
        new=self.pair('new',stamp=2)
        resume.resolve_run('project-a','',resume.AUTO)
        with patch.object(Path,'read_text',side_effect=AssertionError('unchanged JSON re-read')):
            self.assertEqual(resume.resolve_run('project-a','',resume.AUTO)['master'],new[0])
        resume.record_run('project-b','',new[0],new[1],124,24,32,32,3)
        self.assertEqual(resume.resolve_run('project-a','',resume.AUTO)['master'],old[0])
        self.assertEqual(resume.resolve_run('project-b','',resume.AUTO)['master'],new[0])
        Path(new[1]).unlink()
        with self.assertRaises(ValueError):resume.resolve_run('project-b','',resume.AUTO)

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

    def test_completed_run_loader_preserves_paired_native_video(self):
        master, _ = self.pair("native-pair")
        data = torch.load(master, weights_only=True)
        data["h3_warmup"] = {"video": torch.full((1, 24, 37, 1, 1), 7.).half(),
                             "start_chunk": 0, "chunk_frames": 124, "overlap_frames": 39}
        torch.save(data, master)
        latent = resume.MMH3LoadResumeRun.execute("project-a", 1)[0]
        torch.testing.assert_close(latent["h3_warmup"]["video"], data["h3_warmup"]["video"], rtol=0, atol=0)
        self.assertEqual(latent["h3_warmup"]["start_chunk"], 0)
        torch.testing.assert_close(latent["samples"].unbind()[1], data["audio"].float(), rtol=0, atol=0)

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

    def fake_video(self):
        """A Video stand-in whose save_to writes real bytes at the path the node reserved."""
        def save_to(path, **kwargs):
            video.saved.append((path, kwargs))
            Path(path).write_bytes(b"encoded render " * 64)
        video = SimpleNamespace(get_dimensions=lambda: (32, 32), get_frame_count=lambda: 124,
                                get_frame_rate=lambda: 24, save_to=save_to, saved=[])
        return video

    def test_save_names_per_project_reserves_exclusively_and_registers_only_after_success(self):
        master, _ = self.pair("saved", record=False)
        video = self.fake_video()
        out = resume.MMH3SaveResumeVideo.execute(video, "final/video", "auto", master, "Project A: take 2?",
                                                 variant="flo")
        path = self.root / "final" / "Project A_ take 2" / "video_flo_00001_.mp4"
        self.assertEqual([p for p, _ in video.saved], [str(path)])
        self.assertEqual(video.saved[0][1]["format"], resume.Types.VideoContainer.MP4)
        self.assertEqual(out[1:3], (master, str(path)))
        shown = out.ui.as_dict()["images"][0]
        self.assertEqual((shown["filename"], shown["subfolder"]), ("video_flo_00001_.mp4", "final/Project A_ take 2"))
        record = resume.resolve_run("Project A: take 2?", "flo", resume.AUTO)
        self.assertEqual(record["video"], str(path))
        self.assertEqual((record["video_size"], record["pod"]), (path.stat().st_size, resume.socket.gethostname()))
        # a variant-less save of the same project counts in the same folder under its own name
        plain, _ = self.pair("saved-plain", record=False)
        resume.MMH3SaveResumeVideo.execute(video, "final/video", "auto", plain, "Project A: take 2?")
        self.assertTrue((path.parent / "video_00001_.mp4").is_file())
        self.assertEqual(resume.resolve_run("Project A: take 2?", "", resume.AUTO)["master"], plain)
        # a failed encode leaves neither a record nor a partial file under a valid name
        newer, _ = self.pair("failed", record=False)
        def boom(path, **kwargs):
            Path(path).write_bytes(b"partial")
            raise OSError("encoder failed")
        video.save_to = boom
        with self.assertRaises(OSError):
            resume.MMH3SaveResumeVideo.execute(video, "final/video", "auto", newer, "Project A: take 2?", variant="flo")
        self.assertEqual(resume.resolve_run("Project A: take 2?", "flo", resume.AUTO)["master"], master)
        self.assertFalse((path.parent / "video_flo_00002_.mp4").exists())

    def test_two_pods_saving_at_once_never_share_a_name(self):
        master, _ = self.pair("shared", record=False)
        folder = self.root / "final" / "p"
        folder.mkdir(parents=True)
        (folder / "video_00001_.mp4").touch()  # the other pod reserved 00001 and is still encoding
        video = self.fake_video()
        first = resume.MMH3SaveResumeVideo.execute(video, "final/video", "auto", master, "p")[2]
        self.assertEqual(Path(first).name, "video_00002_.mp4")
        self.assertEqual((folder / "video_00001_.mp4").stat().st_size, 0)
        # the other pod grabs the next number between our listing and our create: we step past it
        listing = resume.folder_paths.get_save_image_path
        def racing(*a, **k):
            result = listing(*a, **k)
            (folder / f"video_{result[2]:05}_.mp4").touch()
            return result
        with patch.object(resume.folder_paths, "get_save_image_path", side_effect=racing):
            second = resume.MMH3SaveResumeVideo.execute(video, "final/video", "auto", master, "p")[2]
        self.assertEqual(Path(second).name, "video_00004_.mp4")
        self.assertEqual((folder / "video_00003_.mp4").stat().st_size, 0)

    def test_project_slugs_cannot_leave_the_output_folder(self):
        self.assertEqual(resume.output_slug("Project A: take 2?"), "Project A_ take 2")
        for bad in ("../../etc", "a/b\\c", "%date:yyyy%", "..", "  .  "):
            slug = resume.output_slug(bad)
            self.assertNotIn("/", slug); self.assertNotIn("\\", slug); self.assertNotIn("..", slug)
            self.assertNotIn("%", slug)
        self.assertEqual(resume.run_filename_prefix("final_vids/video", "ProjectA", "take2"), "final_vids/ProjectA/video_take2")
        self.assertEqual(resume.run_filename_prefix("final_vids/video", "ProjectA", ""), "final_vids/ProjectA/video")
        self.assertEqual(resume.run_filename_prefix("video", "", ""), "video")

    def test_a_video_rewritten_by_another_run_is_refused_not_resumed(self):
        master, video = self.pair("a", stamp=1)
        self.assertEqual(resume.resolve_run("project-a", "", resume.AUTO)["master"], master)
        Path(video).write_bytes(b"the other project's much longer render")
        with self.assertRaisesRegex(ValueError, "rewritten after this run finished"):
            resume.resolve_run("project-a", "", resume.AUTO)
        with self.assertRaisesRegex(ValueError, "selected run"):
            resume.resolve_run("project-a", "", master)
        # a legacy record (no fingerprint) is checked by frame count instead
        record_file = next(resume._index_directory().glob("*.json"))
        record = resume.json.loads(record_file.read_text())
        for key in ("video_size", "video_mtime_ns", "pod"):
            record.pop(key)
        record_file.write_text(resume.json.dumps(record))
        with patch.object(resume, "probe_frame_count", return_value=8471):
            with self.assertRaisesRegex(ValueError, "holds 8471 frames while this run saved 124"):
                resume.resolve_run("project-a", "", resume.AUTO)
        with patch.object(resume, "probe_frame_count", return_value=124):
            self.assertEqual(resume.resolve_run("project-a", "", resume.AUTO)["master"], master)
        # a same-size copy with a new mtime (restored file) is accepted on a matching frame count
        resume.record_run("project-a", "", master, video, 124, 24, 32, 32, 5)
        os.utime(video, ns=(0, 0))
        with patch.object(resume, "probe_frame_count", return_value=124):
            self.assertEqual(resume.resolve_run("project-a", "", resume.AUTO)["master"], master)
        with patch.object(resume, "probe_frame_count", return_value=None):
            with self.assertRaisesRegex(ValueError, "unreadable number of frames"):
                resume.resolve_run("project-a", "", resume.AUTO)

    def graded_for(self, record, frames_written=b"graded stand-in"):
        """Write a _graded.mp4 sibling for a record's render (newer than it by construction)."""
        graded = Path(resume.graded_sibling(record["video"]))
        time.sleep(0.01)
        graded.write_bytes(frames_written)
        return graded

    def test_prefix_pixels_use_a_finished_graded_render_and_nothing_else(self):
        self.pair("graded-run")
        record = resume.resolve_run("project-a", "", resume.AUTO)
        raw = record["video"]

        # the finisher runs detached: right after a save there is no graded copy at all
        path, note = resume.resolve_prefix_video(record)
        self.assertEqual(path, raw)
        self.assertIn("no _graded.mp4 yet", note)

        graded = self.graded_for(record)
        with patch.object(resume, "RECENT_SECONDS", -1.0):  # settled without waiting
            with patch.object(resume, "probe_frame_count", return_value=124) as probe:
                path, note = resume.resolve_prefix_video(record)
            self.assertEqual(path, str(graded))
            self.assertIn("GRADED", note)
            probe.assert_called_once_with(str(graded))

            # opting out never probes anything
            with patch.object(resume, "probe_frame_count", side_effect=AssertionError) as probe:
                self.assertEqual(resume.resolve_prefix_video(record, prefer_graded=False)[0], raw)

            # a partial or stale grade would shorten the prefix and misalign MMH3JoinAV
            with patch.object(resume, "probe_frame_count", return_value=400):
                path, note = resume.resolve_prefix_video(record)
            self.assertEqual(path, raw)
            self.assertIn("400 frames", note)

            with patch.object(resume, "probe_frame_count", return_value=None):
                path, note = resume.resolve_prefix_video(record)
            self.assertEqual(path, raw)
            self.assertIn("unreadable", note)

            # a leftover from an earlier render at the same path
            stale = os.path.getmtime(raw) - 10
            os.utime(graded, (stale, stale))
            with patch.object(resume, "probe_frame_count", return_value=124):
                path, note = resume.resolve_prefix_video(record)
            self.assertEqual(path, raw)
            self.assertIn("older than the render", note)

    def test_a_graded_file_still_being_written_is_refused(self):
        self.pair("in-flight")
        record = resume.resolve_run("project-a", "", resume.AUTO)
        graded = self.graded_for(record)
        with patch.object(resume, "RECENT_SECONDS", 3600.0), \
             patch.object(resume, "SETTLE_SECONDS", 0.05), \
             patch.object(resume, "probe_frame_count", return_value=124):
            self.assertTrue(resume.settled(graded))          # size stable across the window
            grew = [graded.write_bytes(b"graded stand-in and more")]
            with patch.object(resume, "settled", return_value=False):
                path, note = resume.resolve_prefix_video(record)
        self.assertEqual(path, record["video"])
        self.assertIn("still being written", note)
        self.assertTrue(grew)

    def test_load_returns_graded_pixels_but_always_the_raw_master_latents(self):
        master, _ = self.pair("mixed")
        record = resume.resolve_run("project-a", "", resume.AUTO)
        graded = self.graded_for(record)
        with patch.object(resume, "RECENT_SECONDS", -1.0), \
             patch.object(resume, "probe_frame_count", return_value=124):
            out = resume.MMH3LoadResumeRun.execute("project-a", 1)
        self.assertEqual(out[4], master)                     # master_path: the raw .pt
        self.assertEqual(out[5], str(graded))                # video_path: the graded pixels
        self.assertIn("GRADED", out[6])
        expected = torch.load(master, weights_only=True)
        for key, tensor in zip(("video", "audio"), out[0]["samples"].unbind()):
            torch.testing.assert_close(tensor, expected[key].float(), rtol=0, atol=0)


class ResumePlanTests(unittest.TestCase):
    def test_fresh_and_chained_plans_on_both_window_grids(self):
        for chunk in (124, 243):
            stride = chunk - 39
            total = chunk + 9 * stride
            fresh = resume.MMH3ResumePlan.execute(total, chunk, 39, 0, 2)
            self.assertEqual(fresh[:9], (10, 0, 0., chunk + stride, 2, 0, 0, 0, 0))
            for accepted in (1, 2, 5, 9):
                out = resume.MMH3ResumePlan.execute(total, chunk, 39, accepted, 2)
                self.assertEqual(out[1], (accepted - 1) * stride)
                self.assertEqual(out[5:9], (accepted - 1, 1, accepted * stride + 39, chunk))
                self.assertEqual(out[4], 1 + min(2, 10 - accepted))
                self.assertEqual(out[1] + out[3], min(total, (accepted + 2) * stride + 39))

    def test_unlimited_cap_and_partial_final_window(self):
        out = resume.MMH3ResumePlan.execute(2895, 243, 39, 12, 0)
        self.assertEqual(out[:9], (14, 2244, 93.5, 651, 3, 11, 1, 2487, 243))
        out = resume.MMH3ResumePlan.execute(2878, 243, 39, 12, 0)
        self.assertEqual(out[3], 634)
        self.assertEqual(resume.MMH3ResumePlan.execute(2878, 243, 39, 0, 0)[3], 2878)

    def test_rejects_exhausted_project_and_invalid_stride(self):
        with self.assertRaisesRegex(ValueError, "continuation"):
            resume.MMH3ResumePlan.execute(447, 243, 39, 2, 1)
        with self.assertRaisesRegex(ValueError, "Overlap"):
            resume.MMH3ResumePlan.execute(447, 243, 243)


class ResumeOutputTests(unittest.TestCase):
    def test_fresh_output_does_not_pull_missing_prefix(self):
        images = torch.rand(243, 2, 3, 3)
        audio = {"sample_rate": 240, "waveform": torch.rand(1, 2, 2430)}
        self.assertEqual(resume.MMH3ResumeJoinAV.check_lazy_status(accepted_prefix_frames=0), [])
        out = resume.MMH3ResumeJoinAV.execute(images, 0, 0, audio)
        self.assertIs(out[0], images)
        self.assertIs(out[1], audio)

    def test_accepted_pixels_and_audio_are_exact_and_no_window_is_duplicated(self):
        for chunk in (124, 243):
            accepted = 2 * (chunk - 39) + 39
            prefix = torch.rand(accepted, 2, 3, 3)
            current = torch.rand(2 * chunk - 39, 2, 3, 3)
            pa = {"sample_rate": 240, "waveform": torch.rand(1, 2, accepted * 10)}
            ca = {"sample_rate": 240, "waveform": torch.rand(1, 2, len(current) * 10)}
            expected_pixels, expected_audio = prefix.clone(), pa["waveform"].clone()
            needed = resume.MMH3ResumeJoinAV.check_lazy_status(accepted_prefix_frames=accepted, audio=ca)
            self.assertEqual(needed, ["prefix_images", "prefix_audio"])
            out = resume.MMH3ResumeJoinAV.execute(current, accepted, chunk, ca, prefix, pa)
            self.assertEqual(len(out[0]), accepted + chunk - 39)
            torch.testing.assert_close(out[0][:accepted], expected_pixels, rtol=0, atol=0)
            torch.testing.assert_close(out[0][accepted:], current[chunk:], rtol=0, atol=0)
            torch.testing.assert_close(out[1]["waveform"][..., :accepted * 10], expected_audio, rtol=0, atol=0)
            torch.testing.assert_close(out[1]["waveform"][..., accepted * 10:], ca["waveform"][..., chunk * 10:], rtol=0, atol=0)

    def test_incomplete_prefix_and_one_sided_audio_fail_before_join(self):
        images = torch.rand(447, 2, 3, 3)
        with self.assertRaisesRegex(ValueError, "exactly 447"):
            resume.MMH3ResumeJoinAV.execute(images, 447, 243, prefix_images=images[:400])
        with self.assertRaisesRegex(ValueError, "both audio"):
            resume.MMH3ResumeJoinAV.execute(images, 447, 243, audio={}, prefix_images=images)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
