"""Continuation packet reuse, frame boundaries, and audio range parity."""
import sys
from pathlib import Path
from types import SimpleNamespace
from fractions import Fraction
import tempfile
import unittest
from unittest.mock import patch

sys.path[:0] = ['/workspace/ComfyUI', str(Path(__file__).resolve().parents[1]),
                '/workspace/ComfyUI/custom_nodes/comfyui-videohelpersuite']
sys.argv = [sys.argv[0], '--cpu']
import comfy.options
comfy.options.enable_args_parsing()
import server
server.PromptServer.instance = SimpleNamespace(prompt_queue=None)
import av
import torch
from videohelpersuite import utils as vhs
from comfy_api.latest import InputImpl, Types
from mmh3tools.nodes_media import MMH3ResumeVideo, MMH3AudioRange

torch.set_num_threads(4)


def packets(path, kind='video'):
    with av.open(str(path)) as container:
        return [bytes(packet) for packet in container.demux(**{kind: 0}) if packet.size]


def frames(path):
    with av.open(str(path)) as container:
        return [frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)]


class ResumeVideoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.images = torch.rand((39, 32, 48, 3), generator=torch.Generator().manual_seed(42))
        self.audio = {'sample_rate': 48000, 'waveform':torch.randn((1, 2, 78000), generator=torch.Generator().manual_seed(43)) * .03}

    def tearDown(self):
        self.temp.cleanup()

    def prefix(self, audio=False):
        path = self.root / 'prefix.mp4'
        InputImpl.VideoFromComponents(Types.VideoComponents(images=self.images, frame_rate=Fraction(24),
            audio=self.audio if audio else None)).save_to(str(path))
        return path

    def test_fresh_is_lazy_and_preserves_inputs(self):
        self.assertEqual(MMH3ResumeVideo.check_lazy_status(accepted_prefix_frames=0), [])
        output = MMH3ResumeVideo.execute(self.images, 0, 0, audio=self.audio)[0]
        self.assertIs(output.get_components().images, self.images)
        self.assertIs(output.get_components().audio, self.audio)

    def test_full_append_copies_prefix_packets_without_decode(self):
        prefix = self.prefix()
        expected = packets(prefix)
        with patch.object(InputImpl.VideoFromFile, 'get_components', side_effect=AssertionError('decoded prefix')):
            output = MMH3ResumeVideo.execute(self.images, 39, 22, prefix_video=str(prefix))[0]
            destination = self.root/'joined.mp4'
            with patch.object(InputImpl.VideoFromFile, '_save_transcoded', side_effect=AssertionError('transcoded compatible prefix')):
                output.save_to(str(destination))
        self.assertEqual(output.get_frame_count(), 56)
        self.assertEqual(len(frames(destination)), 56)
        self.assertEqual(packets(destination)[:39], expected)

    def test_earlier_prefix_is_frame_accurate(self):
        prefix = self.prefix()
        output = MMH3ResumeVideo.execute(self.images, 22, 22, prefix_video=str(prefix))[0]
        destination = self.root/'earlier.mp4'
        output.save_to(str(destination))
        self.assertEqual(len(frames(destination)), 39)
        self.assertEqual(output.get_frame_count(), 39)

    def test_audio_range_and_join_use_same_waveform_as_vhs(self):
        prefix = self.prefix(audio=True)
        audio = MMH3AudioRange.execute(str(prefix), 0., 39, 24.)[0]
        reference = vhs.get_audio(str(prefix), 0., 39/24)
        torch.testing.assert_close(audio['waveform'], reference['waveform'], rtol=0, atol=0)
        output = MMH3ResumeVideo.execute(self.images, 39, 22, prefix_video=str(prefix), audio=self.audio)[0]
        wanted = torch.cat((reference['waveform'], self.audio['waveform'][..., 44000:]), dim=-1)
        torch.testing.assert_close(output.complete_audio['waveform'], wanted, rtol=0, atol=0)
        destination = self.root/'audio.mp4'
        output.save_to(str(destination))
        self.assertEqual(len(frames(destination)), 56)
        self.assertEqual(packets(destination)[:39], packets(prefix))
        with av.open(str(destination)) as container:
            self.assertEqual(len(container.streams.audio), 1)
            self.assertEqual(container.streams.audio[0].rate, 48000)

    def test_chained_append_and_bit_depth_change_preserve_frame_counts(self):
        prefix = self.prefix()
        first = MMH3ResumeVideo.execute(self.images,39,22,prefix_video=str(prefix))[0]
        first_path = self.root/'first.mp4'
        first.save_to(str(first_path))
        second = MMH3ResumeVideo.execute(self.images,56,22,prefix_video=str(first_path))[0]
        second_path = self.root/'second.mp4'
        second.save_to(str(second_path))
        self.assertEqual(len(frames(second_path)),73)
        self.assertEqual(packets(second_path)[:56],packets(first_path))
        deeper = MMH3ResumeVideo.execute(self.images,39,22,prefix_video=str(prefix),bit_depth=10)[0]
        deep_path = self.root/'deep.mp4'
        deeper.save_to(str(deep_path))
        self.assertEqual(len(frames(deep_path)),56)
        self.assertEqual(InputImpl.VideoFromFile(str(deep_path)).get_bit_depth(),10)

    def test_invalid_prefix_does_not_silently_truncate(self):
        prefix = self.prefix()
        with self.assertRaisesRegex(ValueError, 'only 39'):
            MMH3ResumeVideo.execute(self.images, 40, 22, prefix_video=str(prefix))
        with self.assertRaisesRegex(ValueError, 'new frames'):
            MMH3ResumeVideo.execute(self.images, 39, 39, prefix_video=str(prefix))


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]], verbosity=2)
