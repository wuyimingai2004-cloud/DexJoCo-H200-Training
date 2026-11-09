"""CPU test bootstrap for the RSI contracts.

Loads the real ``wan.modules.model`` / ``models.*`` code without the CUDA-only
flash kernels or the heavy optional imports pulled by ``wan/__init__``
(T5 tokenizers, VAEs, diffusers). Only torch is required.

``flash_attention`` is replaced by a padding-aware SDPA reference with the same
varlen semantics (q rows past q_lens are zero, k columns past k_lens masked).
Import this module before any ``models.*`` / ``wan.*`` import.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "third_party/Thinker-WM/libs" / "motus-monitor"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def reference_flash_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.0,
                              softmax_scale=None, q_scale=None, causal=False,
                              window_size=(-1, -1), deterministic=False,
                              dtype=torch.bfloat16, version=None):
    b, lq, n, d = q.shape
    lk = k.size(1)
    out_dtype = q.dtype
    qf, kf, vf = (x.float().transpose(1, 2) for x in (q, k, v))
    if q_scale is not None:
        qf = qf * q_scale
    mask = torch.ones(b, 1, lq, lk, dtype=torch.bool, device=q.device)
    if k_lens is not None:
        cols = torch.arange(lk, device=q.device)[None, :] < k_lens.to(q.device)[:, None]
        mask = mask & cols[:, None, None, :]
    if causal:
        mask = mask & torch.ones(lq, lk, dtype=torch.bool, device=q.device).tril()[None, None]
    out = torch.nn.functional.scaled_dot_product_attention(
        qf, kf, vf, attn_mask=mask, scale=softmax_scale)
    out = out.transpose(1, 2)
    if q_lens is not None:
        rows = torch.arange(lq, device=q.device)[None, :] < q_lens.to(q.device)[:, None]
        out = out * rows[:, :, None, None]
    return out.to(out_dtype)


def _install():
    if "diffusers" not in sys.modules:
        d = types.ModuleType("diffusers")
        cu = types.ModuleType("diffusers.configuration_utils")
        dm = types.ModuleType("diffusers.models")
        mu = types.ModuleType("diffusers.models.modeling_utils")

        class ConfigMixin:  # noqa: D401 - test double
            pass

        def register_to_config(fn):
            return fn

        class ModelMixin(nn.Module):
            pass

        cu.ConfigMixin, cu.register_to_config, mu.ModelMixin = ConfigMixin, register_to_config, ModelMixin
        for name, mod in (("diffusers", d), ("diffusers.configuration_utils", cu),
                          ("diffusers.models", dm), ("diffusers.models.modeling_utils", mu)):
            sys.modules[name] = mod
    def _stub(name, **attrs):
        try:
            __import__(name)
        except ImportError:
            mod = types.ModuleType(name)
            for k, v in attrs.items():
                setattr(mod, k, v)
            sys.modules[name] = mod

    class _Unavailable:
        @classmethod
        def from_pretrained(cls, *a, **k):
            raise RuntimeError("transformers stub: real VLM unavailable in CPU tests")

    _stub("transformers", AutoProcessor=_Unavailable, Qwen3VLForConditionalGeneration=_Unavailable,
          AutoConfig=_Unavailable)
    _stub("qwen_vl_utils", process_vision_info=lambda *a, **k: (None, None))
    for pkg, path in (("thinker_wm.wan", ROOT / "third_party/Thinker-WM/wan"), ("thinker_wm.wan.modules", ROOT / "third_party/Thinker-WM/wan" / "modules"),
                      ("thinker_wm.wan.utils", ROOT / "third_party/Thinker-WM/wan" / "utils")):
        if pkg not in sys.modules:
            mod = types.ModuleType(pkg)
            mod.__path__ = [str(path)]
            sys.modules[pkg] = mod
    import thinker_wm.wan.modules.attention as attn
    attn.flash_attention = reference_flash_attention
    import thinker_wm.wan.modules.model as wm
    wm.flash_attention = reference_flash_attention
    if "thinker_wm.models.wan_model" not in sys.modules:
        from thinker_wm import models  # noqa: F401
        wmod = types.ModuleType("thinker_wm.models.wan_model")

        class WanVideoModel:  # VAE/T5 wrapper; never constructed in CPU tests
            def __init__(self, *a, **k):
                raise RuntimeError("WanVideoModel stub: use a test harness")

        wmod.WanVideoModel = WanVideoModel
        sys.modules["thinker_wm.models.wan_model"] = wmod


_install()


def tiny_wan(num_layers=6, dim=64, heads=4, in_dim=8, ffn=128):
    from thinker_wm.wan.modules.model import WanModel
    torch.manual_seed(0)
    return WanModel(model_type="ti2v", patch_size=(1, 1, 1), text_len=16, in_dim=in_dim,
                    dim=dim, ffn_dim=ffn, freq_dim=32, text_dim=dim, out_dim=in_dim,
                    num_heads=heads, num_layers=num_layers)


def tiny_world(enc=2, dec=2, und_layers=1, action_dim=8, t_downsample=4, ratio=4,
               enable_depth=False, enable_history=False, history_pool=1,
               history_logit_bias_init=-6.0, recovery_feature_dim=0, num_layers=6, seed=0,
               gradient_checkpointing=False, depth_uni_clone_layers=0):
    """Real UniWorldModel on a tiny Wan; und is fed through ``und_cache``."""
    from thinker_wm.models.world_model import UniWorldModel
    wan = tiny_wan(num_layers=num_layers)
    torch.manual_seed(seed)
    m = UniWorldModel(
        None, None, wan, action_dim=action_dim,
        video_encoder_layer_num=enc, video_decoder_layer_num=dec,
        action_encoder_layer_num=enc, action_decoder_layer_num=dec, und_layer_num=und_layers,
        action_t_downsample=t_downsample, video_action_freq_ratio=ratio,
        unified_action_dim=action_dim, enable_depth=enable_depth,
        enable_history=enable_history, history_spatial_pool=history_pool,
        history_logit_bias_init=history_logit_bias_init,
        enable_recovery_condition=recovery_feature_dim > 0,
        recovery_feature_dim=recovery_feature_dim,
        gradient_checkpointing=gradient_checkpointing, depth_uni_clone_layers=depth_uni_clone_layers)
    # zero-init adapters would make history/recovery invisible to the tests
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() >= 1 and torch.count_nonzero(p) == 0:
                p.normal_(std=0.05)
    return m.eval()


def und_cache(b, length=5, dim=64, lens=None, seed=1):
    g = torch.Generator().manual_seed(seed)
    tok = torch.randn(b, length, dim, generator=g)
    lens = torch.tensor(lens if lens is not None else [length] * b, dtype=torch.long)
    return tok, torch.tensor([length, 1, 1]).repeat(b, 1), lens
