"""Tiny randomly-initialized backbone stand-ins for micro debug training.

When ``MotusConfig.micro_train=True``, ``UBT_WM`` builds these stand-ins
instead of loading the real pretrained backbones:

- ``MicroVAE``         replaces ``wan.modules.vae2_2.Wan2_2_VAE``
- ``MicroVLM``         replaces ``Qwen3VLForConditionalGeneration``
- ``MicroVLMProcessor`` replaces the Qwen3-VL ``AutoProcessor``
- the Wan DiT itself is a real ``wan.modules.model.WanModel`` built from a
  tiny config (same architecture, much smaller dims/layers, random init).

(T5 was removed from the model entirely — 评审 #10 — so there is no T5
stand-in here anymore.)

The stand-ins only replicate the *interfaces* used by ``UniWorldModel`` /
``UndBranch`` / ``UBT_WM`` so that the full Motus training path
(``UBT_WM.training_step``) runs end-to-end and backpropagates. They carry no
semantic meaning and must never be used for real training or evaluation.
"""

import logging
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from thinker_wm.wan.modules.model import WanModel

logger = logging.getLogger(__name__)

# Tiny Wan DiT config mirroring the real Wan2.2-TI2V-5B config.json
# (dim=3072, ffn_dim=14336, num_heads=24, num_layers=30, in/out_dim=48,
# model_type='ti2v', patch_size=(1,2,2), text_len=512, freq_dim=256,
# text_dim=4096) with the sizes shrunk to a minimum.
#
# Constraint: VideoBranch/UniBranch require
# ``encode_layer_num + decode_layer_num < num_layers``, so with 1+1 branch
# layers the minimum is num_layers=3 (encode 1, uni 1, decode 1).
MICRO_WAN_CONFIG: Dict[str, Any] = {
    "model_type": "ti2v",
    "patch_size": (1, 2, 2),
    "text_len": 512,
    "in_dim": 48,
    "out_dim": 48,
    "freq_dim": 256,
    "text_dim": 4096,
    "dim": 192,
    "ffn_dim": 384,
    "num_heads": 2,
    "num_layers": 3,
}

_BYTE_VOCAB_SIZE = 258  # 0 = pad, 1..256 = utf-8 byte + 1, 257 = image token
_IMAGE_TOKEN_ID = 257
_NUM_IMAGE_TOKENS = 16  # 4x4 grid, matches MicroVLMProcessor image_grid_thw
_IMAGE_SIZE = 64


def _byte_tokenize(text: str, max_len: int) -> List[int]:
    ids = [b + 1 for b in text.encode("utf-8", errors="replace")[:max_len]]
    return ids or [0]


class MicroVAE(nn.Module):
    """Drop-in replacement for Wan2_2_VAE (encode/decode interface only).

    ``encode([B, 3, T, H, W])`` -> ``[B, z_dim, (T-1)/4 + 1, H/8, W/8]``
    ``decode(list of [z_dim, T', H', W'])`` -> list of ``[3, T, H, W]`` in [-1, 1]
    """

    def __init__(self, z_dim: int = 48):
        super().__init__()
        self.z_dim = z_dim
        self.enc_first = nn.Conv2d(3, z_dim, kernel_size=8, stride=8)
        self.enc_rest = nn.Conv3d(3, z_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))
        self.dec_first = nn.ConvTranspose2d(z_dim, 3, kernel_size=8, stride=8)
        self.dec_rest = nn.ConvTranspose3d(z_dim, 3, kernel_size=(4, 8, 8), stride=(4, 8, 8))

    def encode(self, videos: torch.Tensor) -> torch.Tensor:
        dtype_in = videos.dtype
        x = videos.to(self.enc_first.weight.dtype)
        first = self.enc_first(x[:, :, 0])  # [B, z, H/8, W/8]
        if x.shape[2] > 1:
            rest = self.enc_rest(x[:, :, 1:])  # [B, z, (T-1)/4, H/8, W/8]
            z = torch.cat([first.unsqueeze(2), rest], dim=2)
        else:
            # T=1（推理侧首帧条件编码）：无 rest 段，与 decode 的 T=1 分支对称
            z = first.unsqueeze(2)
        return z.to(dtype_in)

    def decode(self, zs) -> Optional[List[torch.Tensor]]:
        if not isinstance(zs, list):
            logging.info("zs should be a list")
            return None
        outputs = []
        for z in zs:
            z = z.unsqueeze(0).to(self.dec_first.weight.dtype)
            first = self.dec_first(z[:, :, 0]).unsqueeze(2)
            if z.shape[2] > 1:
                rest = self.dec_rest(z[:, :, 1:])
                frames = torch.cat([first, rest], dim=2)
            else:
                frames = first
            outputs.append(frames.float().clamp_(-1, 1).squeeze(0))
        return outputs


class MicroVLMProcessor:
    """Drop-in replacement for the Qwen3-VL AutoProcessor.

    Produces ``input_ids`` (16 image tokens + utf-8 byte ids), a resized
    ``pixel_values`` image and ``image_grid_thw`` in the same dict layout the
    real processor returns.
    """

    def apply_chat_template(self, messages, tokenize: bool = False, add_generation_prompt: bool = True):
        text = "".join(
            part.get("text", "")
            for message in messages
            for part in message.get("content", [])
            if part.get("type") == "text"
        )
        return f"<|im_start|>user\n<|image_pad|>{text}<|im_end|>\n<|im_start|>assistant\n"

    def __call__(self, text=None, images=None, videos=None, padding: bool = True, return_tensors: str = "pt"):
        texts = list(text) if text is not None else [""]
        input_ids = torch.stack([
            torch.tensor(
                [_IMAGE_TOKEN_ID] * _NUM_IMAGE_TOKENS + _byte_tokenize(t, 512),
                dtype=torch.long,
            )
            for t in texts
        ])
        attention_mask = torch.ones_like(input_ids)

        image_list = list(images) if images else []
        if image_list:
            arrays = [
                torch.from_numpy(
                    np.asarray(img.convert("RGB").resize((_IMAGE_SIZE, _IMAGE_SIZE)), dtype=np.float32) / 255.0
                ).permute(2, 0, 1)
                for img in image_list
            ]
            pixel_values = torch.stack(arrays)
            image_grid_thw = torch.tensor([[1, 4, 4]] * len(arrays), dtype=torch.long)
        else:
            pixel_values = torch.zeros(0, 3, _IMAGE_SIZE, _IMAGE_SIZE)
            image_grid_thw = torch.zeros(0, 3, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }


class _MicroVLMInner:
    """Replicates the ``vlm_model.model`` attribute surface used by UndBranch."""

    def __init__(self, outer: "MicroVLM"):
        self._outer = outer
        self.language_model = outer._language_model

    def get_placeholder_mask(self, input_ids, inputs_embeds=None, image_features=None):
        mask = input_ids == _IMAGE_TOKEN_ID
        return mask.unsqueeze(-1).expand(inputs_embeds.shape), None

    def get_rope_index(self, input_ids, image_grid_thw=None, video_grid_thw=None, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device)
        position_ids = positions.view(1, 1, seq_len).expand(3, batch_size, seq_len)
        rope_deltas = torch.zeros(batch_size, device=input_ids.device)
        return position_ids, rope_deltas


class _MicroVLMLanguageModel:
    """Callable language model returning an object with ``last_hidden_state``.

    Plain object (not an nn.Module): it owns no parameters and registering it
    as a submodule would create a reference cycle with MicroVLM.
    """

    def __init__(self, outer: "MicroVLM"):
        self._outer = outer

    def __call__(
        self,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        visual_pos_masks=None,
        deepstack_visual_embeds=None,
        **unused,
    ):
        outer = self._outer
        dtype_in = inputs_embeds.dtype
        h = inputs_embeds.to(outer.down.weight.dtype)
        residual = h
        h = outer.down(h)
        key_padding_mask = attention_mask == 0 if attention_mask is not None else None
        h = outer.block(h, src_key_padding_mask=key_padding_mask)
        h = outer.up(h) + residual
        return SimpleNamespace(last_hidden_state=h.to(dtype_in))


class MicroVLM(nn.Module):
    """Drop-in replacement for Qwen3VLForConditionalGeneration (interface only).

    ``vlm_dim`` must stay 2048 because ``UndBranch`` hardcodes
    ``nn.Linear(vlm_dim=2048, wan_dim)`` for its bridge layer.
    """

    def __init__(self, vlm_dim: int = 2048, hidden_dim: int = 256):
        super().__init__()
        self.vlm_dim = vlm_dim
        self.embed_tokens = nn.Embedding(_BYTE_VOCAB_SIZE, vlm_dim)
        self.visual_conv = nn.Conv2d(3, hidden_dim, kernel_size=16, stride=16)  # 64 -> 4x4 = 16 tokens
        self.visual_proj = nn.Linear(hidden_dim, vlm_dim)
        self.down = nn.Linear(vlm_dim, hidden_dim)
        self.block = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            norm_first=True,
        )
        self.up = nn.Linear(hidden_dim, vlm_dim)
        self._language_model = _MicroVLMLanguageModel(self)
        self.model = _MicroVLMInner(self)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def get_image_features(self, pixel_values, image_grid_thw) -> Tuple[List[torch.Tensor], None]:
        weight = self.visual_conv.weight
        feats = self.visual_conv(pixel_values.to(weight.dtype))  # [N, hidden, 4, 4]
        tokens = feats.flatten(2).transpose(1, 2)  # [N, 16, hidden]
        tokens = self.visual_proj(tokens)  # [N, 16, vlm_dim]
        return [tokens[i] for i in range(tokens.shape[0])], None


def build_micro_backbones(
    device: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
    wan_config: Optional[Dict[str, Any]] = None,
):
    """Build the tiny randomly-initialized backbones for micro debug training.

    Returns ``(vlm_preprocess, vlm_model, wan_video_model)``; called by
    ``UBT_WM.__init__`` when ``MotusConfig.micro_train=True``.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = dict(MICRO_WAN_CONFIG)
    if wan_config:
        config.update(wan_config)

    logger.info("Building micro WanModel (random init): %s", config)
    wan_model = WanModel(**config).to(device=device, dtype=dtype)

    vlm_preprocess = MicroVLMProcessor()
    vlm_model = MicroVLM().to(device)
    vae = MicroVAE(z_dim=config.get("in_dim", 48)).to(device)

    # UBT_WM only accesses ``.wan_model`` and ``.vae`` on the wrapper.
    wan_video_model = SimpleNamespace(wan_model=wan_model, vae=vae)
    return vlm_preprocess, vlm_model, wan_video_model
