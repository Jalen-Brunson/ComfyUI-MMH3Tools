"""Execution-local reference reuse, backed by disposable disk entries."""
import hashlib
import logging
import os
import pickle
from pathlib import Path
import tempfile

import torch
import folder_paths


def hash_value(h, value):
    if isinstance(value, torch.Tensor):
        h.update(b'tensor:')
        hash_value(h, (tuple(value.shape), str(value.dtype)))
        # Hash every byte; sampled hashes can reuse a changed reference silently.
        for part in value.detach().split(16):
            data = part.cpu().contiguous().view(torch.uint8).numpy()
            h.update(memoryview(data).cast('B'))
    elif isinstance(value, dict):
        h.update(b"dict:")
        hash_value(h, len(value))
        for key, item in value.items():
            hash_value(h, key)
            hash_value(h, item)
    elif isinstance(value, (list, tuple)):
        h.update(b"sequence:")
        hash_value(h, len(value))
        for item in value:
            hash_value(h, item)
    else:
        data = repr(value).encode()
        h.update(str(len(data)).encode() + b":" + data)


def model_key(vae):
    if vae is None:
        return None
    layout = tuple((name, getattr(vae.first_stage_model, name, None))
                   for name in ('tiling', 'tile_size', 'tile_overlap_min', 'clip_length', 'token_drop'))
    return (str(vae.patcher.patches_uuid), str(vae.vae_dtype), str(vae.vae_output_dtype()),
            str(vae.device), getattr(vae, "audio_sample_rate", None), layout)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to('cpu', copy=True)
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


class ReferenceCache:
    def __init__(self):
        self.directory = Path(folder_paths.get_temp_directory()) / 'mmh3_reference_cache'
        self.entries = {}
        self.hits = 0
        self.misses = 0

    def get(self, identity, build, vae, audio_vae):
        h = hashlib.sha256(b'mmh3-prepared-refs-v1')
        hash_value(h, identity)
        key = h.hexdigest()
        self.last_key = key
        if key in self.entries:
            self.hits += 1
            return self.entries[key]
        path = self.directory / (key + '.pt')
        if path.is_file():
            try:
                result = torch.load(path, map_location='cpu', weights_only=True)
            except (OSError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError):
                logging.warning('[MMH3ReferenceMultiPrompt] Rebuilding unreadable reference cache.')
            else:
                items, blocks = result
                for block in blocks:
                    if block.get('latent') is not None:
                        block['latent'] = block['latent'].to(vae.output_device)
                    if block.get('audio_latent') is not None:
                        block['audio_latent'] = block['audio_latent'].to(audio_vae.output_device)
                os.utime(path, None)
                self.entries[key] = result
                self.hits += 1
                return result
        result = build()
        self.directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=self.directory, suffix='.tmp', delete=False) as f:
            temporary = f.name
        try:
            torch.save(cpu_tree(result), temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        self.entries[key] = result
        self.misses += 1
        return result

    def finish(self):
        # Keep at most 4 GiB; entries are expendable and never resume masters.
        files = sorted(self.directory.glob('*.pt'), key=lambda p: p.stat().st_mtime_ns, reverse=True)
        size = 0
        for path in files:
            size += path.stat().st_size
            if size > 4 * 1024**3:
                path.unlink()
        logging.info('[MMH3ReferenceMultiPrompt] Prepared references: %d reused, %d built', self.hits, self.misses)
