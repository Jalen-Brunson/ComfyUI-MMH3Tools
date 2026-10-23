import sys
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import uuid

sys.path[:0] = ['/workspace/ComfyUI', str(Path(__file__).resolve().parents[1])]
sys.argv = [sys.argv[0], '--cpu']
import comfy.options
comfy.options.enable_args_parsing()
import torch
from mmh3tools import nodes_multiprompt as mp
from mmh3tools.reference_cache import ReferenceCache

torch.set_num_threads(4)

class Vae:
    vae_dtype = torch.float32
    output_device = device = 'cpu'
    def __init__(self):
        self.patcher = SimpleNamespace(patches_uuid=uuid.uuid4())
        self.calls = 0
        self.first_stage_model = SimpleNamespace(tiling=False)
    def vae_output_dtype(self):return torch.float32
    def encode(self, x):
        self.calls += 1
        return x.mean(dim=(1,2,3)).reshape(1,1,-1,1,1).clone()

class Clip:
    def __init__(self):
        self.patcher=SimpleNamespace(patches_uuid=uuid.uuid4())
        self.calls=0
    def tokenize(self,text,**kwargs):return text
    def encode_from_tokens_scheduled(self,tokens):
        self.calls+=1
        return [[torch.zeros(1,2,3), {'text':tokens}]]

class ReferenceCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = patch('folder_paths.get_temp_directory', return_value=self.temp.name)
        self.folder.start()
        self.vae=Vae()
        self.audio_vae=Vae()
        self.frames=torch.rand(73,64,64,3,generator=torch.Generator().manual_seed(1))
    def tearDown(self):self.folder.stop();self.temp.cleanup()
    def build(self, cache, span=(0,38), static=False):
        return mp._cached_refs(cache,self.vae,self.audio_vae,64,64,73,'match',self.frames[:1],
            {'ref_video_0':self.frames},None,None,ref_window=span,
            static_videos={'ref_video_0'} if static else set())
    def assert_tree(self,a,b):
        if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
        elif isinstance(a,dict):
            self.assertEqual(a.keys(),b.keys())
            for k in a:self.assert_tree(a[k],b[k])
        elif isinstance(a,(list,tuple)):
            self.assertEqual(len(a),len(b))
            for x,y in zip(a,b):self.assert_tree(x,y)
        else:self.assertEqual(a,b)
    def test_prompt_edits_skip_vae_and_reencode_only_changed_text(self):
        clip=Clip()
        args=dict(clip=clip,vae=self.vae,audio_vae=self.audio_vae,width=64,height=64,
                  length=73,ref_image_size='match',ref_images=self.frames[:1],unload_text_encoder=False)
        first=mp.MMH3ReferenceMultiPrompt.execute(**args,prompts='first | second')
        second=mp.MMH3ReferenceMultiPrompt.execute(**args,prompts='first | edited')
        self.assertEqual(self.vae.calls,1)
        self.assertEqual(clip.calls,3)
        self.assertIs(first[0]['conds'][0],second[0]['conds'][0])
        self.assertEqual(second[0]['conds'][1][0][1]['text'],'edited')
        clip.patcher.patches_uuid=uuid.uuid4()
        mp.MMH3ReferenceMultiPrompt.execute(**args,prompts='first | edited')
        self.assertEqual(clip.calls,5)
        self.assertEqual(self.vae.calls,1)

    def test_parity_disk_hit_and_no_persistent_tensor_store(self):
        expected=mp._build_refs(self.vae,self.audio_vae,64,64,73,'match',self.frames[:1],
            {'ref_video_0':self.frames},None,None,ref_window=(0,38),static_videos=set())
        self.assert_tree(expected,self.build(ReferenceCache()))
        calls=self.vae.calls
        self.assert_tree(expected,self.build(ReferenceCache()))
        self.assertEqual(calls,self.vae.calls)
    def test_static_reference_reused_in_next_window(self):
        cache=ReferenceCache()
        first=self.build(cache,static=True)
        calls=self.vae.calls
        second=self.build(cache,(34,72),static=True)
        self.assertEqual(self.vae.calls,calls)
        self.assert_tree(first,second)
    def test_window_and_single_pixel_changes_invalidate(self):
        cache=ReferenceCache();self.build(cache);initial=cache.fingerprint
        self.build(ReferenceCache(),(34,72))
        self.assertEqual(self.vae.calls,3) # common image + two video spans
        self.frames[20,0,1,0] += .1
        cache=ReferenceCache();self.build(cache)
        self.assertNotEqual(initial,cache.fingerprint)
        self.assertEqual(self.vae.calls,4)
    def test_model_change_invalidates_and_corruption_rebuilds(self):
        self.build(ReferenceCache())
        self.vae.patcher.patches_uuid=uuid.uuid4()
        self.build(ReferenceCache())
        self.assertEqual(self.vae.calls,4)
        for p in (Path(self.temp.name)/'mmh3_reference_cache').glob('*.pt'):p.write_bytes(b'corrupt')
        self.build(ReferenceCache())
        self.assertEqual(self.vae.calls,6)

if __name__=='__main__':unittest.main(argv=[sys.argv[0]],verbosity=2)
