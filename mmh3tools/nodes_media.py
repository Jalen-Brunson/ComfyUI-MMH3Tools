"""Load audio ranges and assemble encoded continuation videos."""
from fractions import Fraction
from io import BytesIO
from pathlib import Path
import sys
import time

import av
import torch
import folder_paths
from comfy_api.latest import io, InputImpl, Types


def vhs_module(name):
    suffix = '.videohelpersuite.' + name
    for key, module in tuple(sys.modules.items()):
        if key.endswith(suffix) or key == 'videohelpersuite.' + name:
            return module
    raise RuntimeError('Video Helper Suite must be loaded for H3 media input.')


def media_path(path):
    path = Path(path.strip().strip('"')).expanduser()
    if not path.is_absolute():
        path = Path(folder_paths.base_path) / path
    return path.resolve()


class MMH3AudioRange(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='MMH3AudioRange', display_name='H3 Load Audio Range', category='MMH3Tools/audio',
            inputs=[io.String.Input('path', force_input=True),
                    io.Float.Input('start_seconds', default=0., min=0.),
                    io.Int.Input('frames', default=0, min=0),
                    io.Float.Input('fps', default=24., min=.001)],
            outputs=[io.Audio.Output('audio')])

    @classmethod
    def fingerprint_inputs(cls, path, **kwargs):
        stat = media_path(path).stat()
        return stat.st_mtime_ns, stat.st_size

    @classmethod
    def execute(cls, path, start_seconds=0., frames=0, fps=24.):
        audio = vhs_module('utils').lazy_get_audio(str(media_path(path)), start_seconds, frames / fps)
        return io.NodeOutput(audio)


class MMH3ResumeVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='MMH3ResumeVideo', display_name='H3 Create Resume Video', category='MMH3Tools/latent',
            description='Encode the new frames and reuse compatible accepted video packets. Earlier-prefix edits and differing codecs are transcoded by the video system.',
            inputs=[io.Image.Input('images'), io.Int.Input('accepted_prefix_frames', force_input=True, min=0),
                    io.Int.Input('pinned_frames', force_input=True, min=0),
                    io.Float.Input('fps', default=24., min=.001),
                    io.Combo.Input('bit_depth', options=[8, 10], default=8),
                    io.Combo.Input('color_space', options=['sRGB', 'HDR', 'HDR PQ'], default='sRGB'),
                    io.Audio.Input('audio', optional=True),
                    io.String.Input('prefix_video', optional=True, lazy=True, force_input=True)],
            outputs=[io.Video.Output('video'), io.String.Output('report')])

    @classmethod
    def check_lazy_status(cls, accepted_prefix_frames=0, prefix_video=None, **kwargs):
        return ['prefix_video'] if accepted_prefix_frames and prefix_video is None else []

    @classmethod
    def execute(cls, images, accepted_prefix_frames, pinned_frames, fps=24., bit_depth=8,
                color_space='sRGB', audio=None, prefix_video=None):
        rate = Fraction(str(fps))
        if not accepted_prefix_frames:
            video = InputImpl.VideoFromComponents(Types.VideoComponents(images=images, audio=audio, frame_rate=rate),
                                                 bit_depth=bit_depth, color_space=color_space)
            return io.NodeOutput(video, f'Fresh output: {len(images)} frames.')
        if not prefix_video or not 0 < pinned_frames < len(images):
            raise ValueError('A continuation needs its accepted video and new frames after the pinned window.')
        prefix = InputImpl.VideoFromFile(str(media_path(prefix_video)))
        if prefix.get_dimensions() != (images.shape[2], images.shape[1]) or prefix.get_frame_rate() != rate:
            raise ValueError('Accepted video and continuation must have matching dimensions and frame rate.')
        total = prefix.get_frame_count()
        if accepted_prefix_frames > total:
            raise ValueError(f'The accepted video has only {total} frames.')
        if accepted_prefix_frames < total:
            prefix = prefix.as_trimmed(0., accepted_prefix_frames / fps, strict_duration=True)
        if prefix.get_bit_depth() != bit_depth or prefix.get_color_space() != color_space:
            converted = BytesIO()
            prefix.save_to(converted, format=Types.VideoContainer.MP4, bit_depth=bit_depth, color_space=color_space)
            prefix = InputImpl.VideoFromFile(converted)
        combined_audio = None
        if audio is None:
            with av.open(str(media_path(prefix_video))) as container:
                if container.streams.audio:
                    raise ValueError('A continuation needs both audio tracks, or neither.')
        if audio is not None:
            prior = vhs_module('utils').get_audio(str(media_path(prefix_video)), 0., accepted_prefix_frames / fps)
            if prior['sample_rate'] != audio['sample_rate']:
                raise ValueError('Accepted and new audio must have matching sample rates.')
            cut = round(pinned_frames / fps * audio['sample_rate'])
            combined_audio = dict(sample_rate=audio['sample_rate'],
                                  waveform=torch.cat((prior['waveform'], audio['waveform'][..., cut:]), dim=-1))
        suffix = InputImpl.VideoFromComponents(
            Types.VideoComponents(images=images[pinned_frames:].clone(), frame_rate=rate),
            bit_depth=bit_depth, color_space=color_space)
        started = time.perf_counter()
        encoded = BytesIO()
        suffix.save_to(encoded, format=Types.VideoContainer.MP4)
        video = InputImpl.VideoFromList([prefix, InputImpl.VideoFromFile(encoded)], complete_audio=combined_audio)
        report = (f'{accepted_prefix_frames} accepted + {len(images) - pinned_frames} new frames. '
                  f'New video encoded in {time.perf_counter() - started:.2f}s; compatible prefix video is copied at save.')
        return io.NodeOutput(video, report)
