from typing import Optional
from PIL.Image import Image
import logging
import torch
import torch.nn as nn

from thinker_wm.utils.vlm_utils import _process_vlm_inputs_batch
from thinker_wm.utils.vlm_utils import preprocess_vlm_messages
from thinker_wm.utils.vlm_utils import preprocess_vlm_text_only
from thinker_wm.utils.vlm_utils import MOTUS_VLM_REWRITE_FALLBACK
from thinker_wm.utils.vlm_utils import validate_vlm_rewrite
from thinker_wm.utils.vlm_utils import apply_field_dropout_to_rewrite
from thinker_wm.utils.vlm_rewrite_cache import VLMRewriteCache
from motus_monitor.timing import timed_section

logger = logging.getLogger(__name__)

class UndEncoder(nn.Module):
    def __init__(self, dim, ffn_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, ffn_dim),
                nn.GELU(approximate='tanh'),
                nn.Linear(ffn_dim, dim)
            )
            for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(dim)
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        for layer, norm in zip(self.layers, self.norms):
            x = x + (layer(norm(x)))
        return x

class UndBranch(nn.Module):
    """Understanding module - handles VLM with understanding queries and Understanding Expert."""

    def __init__(self, vlm_preprocess, vlm_model, dim, ffn_dim, num_layers, vlm_dim=2048,
                 vlm_text_rewrite: bool = False, vlm_rewrite_max_new_tokens: int = 128,
                 vlm_rewrite_cache_dir: str = "", vlm_field_dropout_prob: float = 0.0,
                 vlm_model_tag: str = ""):
        super().__init__()
        # VLM model reference
        self.vlm_preprocess = vlm_preprocess
        self.vlm_model = vlm_model
        # VLM 文本改写（20260819，见 encode）：条件先由 VLM 按消息中的压缩
        # 改写指令（纯文本、不输入首帧图；Task/Action + Scene 若有，按数据集
        # 家族分模板）贪心生成短文本再编码；False = 原始消息直接编码（旧行为）。
        self.vlm_text_rewrite = bool(vlm_text_rewrite)
        self.vlm_rewrite_max_new_tokens = int(vlm_rewrite_max_new_tokens)
        self._rewrite_logged = False
        # 改写结果磁盘缓存（20260818_v2，model.vlm_rewrite_cache_dir；空 = 禁用
        # 保持逐次 generate 旧行为）：key = sha256(model_tag + 消息文本)（消息
        # 内含家族模板指令，换模板/模型自动隔离命名空间），分片文件 + 原子写。
        self._rewrite_cache = (
            VLMRewriteCache(vlm_rewrite_cache_dir, vlm_model_tag or "vlm")
            if str(vlm_rewrite_cache_dir).strip()
            else None
        )
        # 最近一次 _rewrite_prompts 的缓存命中统计 (hits, misses)，供训练侧
        # 记入 step_record（vlm_rewrite_cache_hits/misses）。
        self._last_rewrite_cache_stats = (0, 0)
        # 输出侧字段 dropout 概率（20260818_v2，用户拍板）：rewrite 开启时
        # vlm_field_dropout_prob 改在改写输出文本上按字段行 drop（rewrite
        # 之后、编码之前；缓存之后作用不产生新 cache key）；仅训练态。
        self.vlm_field_dropout_prob = float(vlm_field_dropout_prob)
        # 无条件（空文本）VLM 特征常量（20260818_v2，用户拍板）：只经
        # tokenizer + embedding 层、不过语言模型；懒构建一次缓存。
        self._uncond_vlm_feature = None

        self.bridge = nn.Linear(vlm_dim, dim)
        
        # Understanding Expert reference
        self.und_encoder = UndEncoder(
            dim=dim,
            ffn_dim=ffn_dim,
            num_layers=num_layers
        )
        
    def extract_und_features(
        self,
        prompt_inputs: list[str],
        first_frame_pil: Optional[list[Image]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract understanding features from VLM last layer.

        first_frame_pil=None 时走纯文本路径（prompt_mode=text_only 的源：
        只用 tokenizer + 语言模型编码，无图像输入；batch 内单源，由数据
        管线 collate 保证不会图文混合）。

        Returns:
            Tuple of (und_token [B, L_padded, vlm_dim], real_lens [B])。
            real_lens 是逐样本真实（未 padding）长度，供下游把 und token
            作为 attention K/V 时 mask 掉 padding 位（padding 位的 hidden
            state 是无意义表示，见评审报告 #1）。
        """
        if first_frame_pil is None:
            vlm_inputs = [
                preprocess_vlm_text_only(v, self.vlm_preprocess)
                for v in prompt_inputs
            ]
        else:
            vlm_inputs = [preprocess_vlm_messages(v, f, self.vlm_preprocess) for v, f in zip(prompt_inputs, first_frame_pil)]
        vlm_inputs = _process_vlm_inputs_batch(vlm_inputs)

        if isinstance(vlm_inputs, list):
            B = len(vlm_inputs)
        else:
            B = vlm_inputs['input_ids'].shape[0]

        # Returns: inputs_embeds, attention_mask, visual_pos_masks, deepstack_image_embeds, position_ids
        inputs_embeds, attention_mask, visual_pos_masks, deepstack_image_embeds, position_ids = self._process_vlm_inputs_to_tokens(vlm_inputs, B)

        if attention_mask is not None:
            real_lens = attention_mask.sum(dim=1).to(torch.long)
        else:
            # 兜底：processor 未返回 attention_mask 时退化为全长（旧行为）
            real_lens = torch.full(
                (B,), inputs_embeds.shape[1], dtype=torch.long,
                device=inputs_embeds.device)

        # Forward through VLM with proper attention_mask and DeepStack features
        vlm_kwargs = {
            'inputs_embeds': inputs_embeds,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'past_key_values': None,
            'use_cache': False,
            'output_attentions': False,
            'output_hidden_states': False,
            'return_dict': True
        }

        # Add DeepStack parameters for Qwen3-VL
        if visual_pos_masks is not None:
            vlm_kwargs['visual_pos_masks'] = visual_pos_masks
        if deepstack_image_embeds is not None:
            vlm_kwargs['deepstack_visual_embeds'] = deepstack_image_embeds

        with torch.no_grad():
            vlm_output = self.vlm_model.model.language_model(**vlm_kwargs)

        # Extract last layer features directly without keeping every hidden state.
        und_token = getattr(vlm_output, 'last_hidden_state', None)
        if und_token is None:
            und_token = vlm_output[0]

        return und_token, real_lens
        
    def _process_vlm_inputs_to_tokens(self, vlm_inputs, B: int) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[list], torch.Tensor]:
        """Convert VLM inputs to tokens.

        Returns:
            Tuple of (inputs_embeds, attention_mask, visual_pos_masks, deepstack_image_embeds, position_ids)
        """
        # Handle both old format (List[Dict]) and new format (Dict[str, Tensor])
        device = self.und_encoder.layers[0][0].weight.device
        dtype = self.und_encoder.layers[0][0].weight.dtype
        if isinstance(vlm_inputs, list):
            # Old format: List[Dict] - do padding and batching
            input_ids_list = [vlm_input['input_ids'] for vlm_input in vlm_inputs]
            attention_mask_list = [vlm_input.get('attention_mask') for vlm_input in vlm_inputs]
            pixel_values_list = [vlm_input.get('pixel_values') for vlm_input in vlm_inputs]
            image_grid_thw_list = [vlm_input.get('image_grid_thw') for vlm_input in vlm_inputs]

            # Pad input_ids and attention_mask to same length
            max_seq_len = max(ids.shape[1] for ids in input_ids_list)
            padded_input_ids = []
            padded_attention_masks = []
            
            for ids, mask in zip(input_ids_list, attention_mask_list):
                if ids.shape[1] < max_seq_len:
                    padding_size = max_seq_len - ids.shape[1]
                    # Pad input_ids with zeros
                    id_padding = torch.zeros(ids.shape[0], padding_size, dtype=ids.dtype, device=ids.device)
                    padded_ids = torch.cat([ids, id_padding], dim=1)
                    # Pad attention_mask with zeros (padding tokens should be ignored)
                    mask_padding = torch.zeros(mask.shape[0], padding_size, dtype=mask.dtype, device=mask.device)
                    padded_mask = torch.cat([mask, mask_padding], dim=1)
                else:
                    padded_ids = ids
                    padded_mask = mask
                padded_input_ids.append(padded_ids)
                padded_attention_masks.append(padded_mask)

            # Batch process
            input_ids_batch = torch.cat(padded_input_ids, dim=0).to(device)
            attention_mask_batch = torch.cat(padded_attention_masks, dim=0).to(device)
            pixel_values_batch = torch.cat([pv.to(device) for pv in pixel_values_list], dim=0)
            image_grid_thw_batch = torch.cat([igt.to(device) for igt in image_grid_thw_list], dim=0)
        else:

            # New format: Dict[str, Tensor] - already batched and padded by collate_fn
            input_ids_batch = vlm_inputs['input_ids'].to(device)
            attention_mask_batch = vlm_inputs['attention_mask'].to(device)
            pixel_values = vlm_inputs.get('pixel_values')
            image_grid_thw = vlm_inputs.get('image_grid_thw')
            pixel_values_batch = (
                pixel_values.to(device) if pixel_values is not None else None
            )
            image_grid_thw_batch = (
                image_grid_thw.to(device) if image_grid_thw is not None else None
            )

        # Get input embeddings
        inputs_embeds = self.vlm_model.get_input_embeddings()(input_ids_batch)

        if pixel_values_batch is not None:
            # Process images - handle different return formats between Qwen2.5-VL and Qwen3-VL
            image_embeds, deepstack_image_embeds = self.vlm_model.get_image_features(pixel_values_batch, image_grid_thw_batch)

            image_embeds = torch.cat(image_embeds, dim=0).to(device, dtype)

            # Insert image embeddings
            image_mask, _ = self.vlm_model.model.get_placeholder_mask(
                input_ids_batch, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            visual_pos_masks = image_mask[..., 0]  # [B, seq_len] - visual positions only
        else:
            # 纯文本路径（text_only 源）：无图像 token，DeepStack/视觉掩码全空。
            visual_pos_masks = None
            deepstack_image_embeds = None

        # Compute position_ids (position_ids remains as original: [3, B, seq_len])
        # Qwen3-VL get_rope_index has different signature: (input_ids, image_grid_thw, video_grid_thw, attention_mask)
        position_ids, _rope_deltas = self.vlm_model.model.get_rope_index(
            input_ids=input_ids_batch,
            image_grid_thw=image_grid_thw_batch,
            video_grid_thw=None,  # No video in current implementation
            attention_mask=attention_mask_batch
        )

        return inputs_embeds, attention_mask_batch, visual_pos_masks, deepstack_image_embeds, position_ids
    
    @torch.no_grad()
    def _rewrite_prompts(self, prompt_inputs: list[str]) -> Optional[list[str]]:
        """批量 greedy 压缩改写（纯文本，不输入首帧图）。

        prompt_inputs 已是上游 build_vlm_rewrite_message 按数据集家族构建的
        压缩改写指令（输出 Task/Action + Scene 若有；egoscale 禁复述标注元
        数据）。逐样本 preprocess_vlm_text_only 后手工左补齐批量 generate，
        输出统一切掉 prompt 段取新 token。max_new_tokens 是输出长度硬上限——
        改写文本再走纯文本编码，und token 数 ≈ 输出 token + chat 模板开销，
        保证远小于 uni 分支 rope freqs 的 1024 行上限；这是模型按指令主动
        压缩，不是对原文的机械截断。

        空白消息（condition dropout 的 UNCONDITIONAL_VLM_MESSAGE=""）不参与
        生成、原样透传。逐样本 validate_vlm_rewrite 校验：缺 Task/Action 行
        或复读元数据行的退化输出替换为 MOTUS_VLM_REWRITE_FALLBACK 安全占位
        文本（短、合法、中性）——不回退长原文，避免重新触发 rope 上限。
        生成异常（如 micro 替身无 generate）返回 None，调用方整批回退原始
        消息编码。
        """
        tokenizer = self.vlm_preprocess.tokenizer
        device = self.und_encoder.layers[0][0].weight.device
        messages = [str(message) for message in prompt_inputs]
        texts: list[str] = list(messages)
        self._last_rewrite_cache_stats = (0, 0)  # 每次调用重置，避免陈旧统计
        gen_indices = [i for i, message in enumerate(messages) if message.strip()]
        if not gen_indices:
            return texts
        # 磁盘缓存（20260818_v2）：命中直接取文本、零 generate；仅 miss 走
        # 批量生成。field dropout 在输出侧（encode 内 rewrite 之后）作用，
        # 不进缓存 key，不影响命中率。
        miss_indices = gen_indices
        if self._rewrite_cache is not None:
            cached, hits, misses = self._rewrite_cache.get_many(
                [messages[i] for i in gen_indices])
            self._last_rewrite_cache_stats = (hits, misses)
            miss_indices = []
            for i, cached_text in zip(gen_indices, cached):
                if cached_text is None:
                    miss_indices.append(i)
                else:
                    texts[i] = cached_text
        if not miss_indices:
            return texts
        try:
            inputs_list = [
                preprocess_vlm_text_only(messages[i], self.vlm_preprocess)
                for i in miss_indices
            ]
            max_len = max(int(inputs["input_ids"].shape[1]) for inputs in inputs_list)
            pad_id = tokenizer.pad_token_id
            if pad_id is None:
                pad_id = tokenizer.eos_token_id
            input_ids_rows, mask_rows = [], []
            for inputs in inputs_list:
                ids = inputs["input_ids"][0]
                mask = inputs["attention_mask"][0]
                pad = max_len - int(ids.shape[0])
                input_ids_rows.append(torch.cat([
                    torch.full((pad,), int(pad_id), dtype=ids.dtype, device=ids.device),
                    ids,
                ]))
                mask_rows.append(torch.cat([
                    torch.zeros(pad, dtype=mask.dtype, device=mask.device), mask,
                ]))
            gen_inputs = {
                "input_ids": torch.stack(input_ids_rows).to(device),
                "attention_mask": torch.stack(mask_rows).to(device),
            }
            generated = self.vlm_model.generate(
                **gen_inputs,
                max_new_tokens=int(self.vlm_rewrite_max_new_tokens),
                do_sample=False,
                pad_token_id=int(pad_id),
            )
            prompt_len = int(gen_inputs["input_ids"].shape[1])
            cache_items = []
            for index, row in zip(miss_indices, generated):
                text = tokenizer.decode(row[prompt_len:], skip_special_tokens=True).strip()
                if validate_vlm_rewrite(text):
                    texts[index] = text
                    cache_items.append((messages[index], text, True))
                else:
                    logger.warning(
                        "VLM 改写输出不合法（空/缺 Task/Action 行/复读元数据），"
                        "该样本改用安全占位文本: %.150s", text,
                    )
                    texts[index] = MOTUS_VLM_REWRITE_FALLBACK
                    # 退化输出同样入缓存（greedy 确定性：同输入必同输出），
                    # 避免每个 epoch 重生成已知坏输入。
                    cache_items.append((messages[index], MOTUS_VLM_REWRITE_FALLBACK, False))
            if self._rewrite_cache is not None and cache_items:
                self._rewrite_cache.set_many(cache_items)
        except Exception as exc:
            logger.warning("VLM 改写批量生成失败，整批回退原始消息编码: %s", exc)
            return None
        return texts

    @torch.no_grad()
    def get_uncond_vlm_feature(self) -> tuple[torch.Tensor, torch.Tensor]:
        """无条件（空文本）VLM 特征常量：只经 tokenizer + embedding 层，
        不过语言模型（20260818_v2 用户拍板）。

        空文本约定 = UNCONDITIONAL_VLM_MESSAGE（""，定义于 models/ubt_wm.py；
        此处不 import 以避免循环依赖）。训练侧 modality dropout 掉 prompt
        与推理侧 CFG uncond 前向统一使用该常量（训练/推理一致性）；VLM
        冻结 → 懒构建一次即为真常量，使用时按 batch expand。
        返回 (hidden [1, L_e, vlm_dim], real_len [1])。
        """
        if self._uncond_vlm_feature is None:
            device = self.und_encoder.layers[0][0].weight.device
            inputs = preprocess_vlm_text_only("", self.vlm_preprocess)
            input_ids = inputs["input_ids"].to(device)
            hidden = self.vlm_model.get_input_embeddings()(input_ids)
            real_len = torch.tensor(
                [int(input_ids.shape[1])], dtype=torch.long, device=device)
            self._uncond_vlm_feature = (hidden, real_len)
        return self._uncond_vlm_feature

    def precompute_vlm_features(
        self,
        prompt_inputs: list[str],
        first_frame_pil: Optional[list[Image]],
        timing=None, timing_sync_cuda: bool = False, timing_level: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """und 编码的冻结段：rewrite（磁盘缓存）→ 输出侧 field dropout
        （训练态）→ VLM 特征提取，返回 (vlm_hidden [B, L, vlm_dim], real_lens
        [B])。可训练段（bridge+und_encoder）在 encode_from_vlm_hidden。

        空文本消息不过语言模型，直接取 get_uncond_vlm_feature 的 token 嵌入
        常量（20260818_v2 用户拍板）；混合 batch 内空文本行用常量填充、
        非空行批量 extract 后按序拼回（padding 位补 0，下游按 seq_lens
        mask，语义不变）。
        """
        # first_frame_pil=None：纯文本 prompt 路径（text_only 源），见
        # extract_und_features。
        # Level >= 2 splits the VLM feature extraction from the und encoder.
        fine = timing if (timing is not None and int(timing_level) >= 2) else None
        # VLM 文本改写（vlm_text_rewrite 开启且本批带首帧图——首帧图此处仅作
        # "机器人源"标记，改写本身纯文本不读图）：先让 VLM 按消息中的压缩改写
        # 指令把长输入生成 Task/Action（+Scene 若有）短文本，再对短文本走纯
        # 文本编码——und token 数由 max_new_tokens 硬上限保证（分段文本后原
        # 始消息可超 uni 分支 rope freqs 1024 行上限）；text_only 源
        # （pils=None）与生成异常时维持原路径不变。
        if self.vlm_text_rewrite and first_frame_pil is not None:
            with timed_section(fine, "wm_und_vlm_rewrite_seconds", timing_sync_cuda):
                rewritten = self._rewrite_prompts(prompt_inputs)
            if rewritten is not None:
                if not self._rewrite_logged:
                    self._rewrite_logged = True
                    logger.info(
                        "VLM 改写示例（首条）: %.150s => %.200s",
                        prompt_inputs[0], rewritten[0],
                    )
                prompt_inputs, first_frame_pil = rewritten, None
                # 输出侧字段 dropout（20260818_v2，用户拍板）：对改写输出按
                # 字段行（Task/Action/Scene）逐样本随机 drop（整行移除、全中
                # 保底留 1，utils.vlm_utils.apply_field_dropout_to_rewrite）。
                # 仅训练态；在缓存之后作用，不产生新 cache key。rewrite 关闭
                # 的旧配置仍走 train.py 消息重建处的输入侧路径，此处不适用。
                if self.training and self.vlm_field_dropout_prob > 0:
                    prompt_inputs = [
                        apply_field_dropout_to_rewrite(t, self.vlm_field_dropout_prob)
                        for t in prompt_inputs
                    ]
        texts = [str(t) for t in prompt_inputs]
        B = len(texts)
        empty_idx = [i for i, t in enumerate(texts) if not t.strip()]
        if not empty_idx:
            with timed_section(fine, "wm_und_vlm_seconds", timing_sync_cuda):
                return self.extract_und_features(texts, first_frame_pil)
        uncond_hidden, uncond_len = self.get_uncond_vlm_feature()
        if len(empty_idx) == B:
            return uncond_hidden.expand(B, -1, -1), uncond_len.repeat(B)
        empty_set = set(empty_idx)
        nonempty = [i for i in range(B) if i not in empty_set]
        sub_pils = (
            [first_frame_pil[i] for i in nonempty]
            if first_frame_pil is not None else None
        )
        with timed_section(fine, "wm_und_vlm_seconds", timing_sync_cuda):
            sub_hidden, sub_lens = self.extract_und_features(
                [texts[i] for i in nonempty], sub_pils)
        l_e = int(uncond_hidden.shape[1])
        l_max = max(int(sub_hidden.shape[1]), l_e)
        hidden = sub_hidden.new_zeros(B, l_max, sub_hidden.shape[2])
        real_lens = torch.empty(B, dtype=torch.long, device=sub_hidden.device)
        for row, i in enumerate(nonempty):
            l_i = int(sub_lens[row])
            hidden[i, :l_i] = sub_hidden[row, :l_i]
            real_lens[i] = sub_lens[row]
        for i in empty_idx:
            hidden[i, :l_e] = uncond_hidden[0].to(hidden.dtype)
            real_lens[i] = uncond_len[0]
        return hidden, real_lens

    def encode_from_vlm_hidden(
        self,
        vlm_hidden: torch.Tensor,
        real_lens: torch.Tensor,
        timing=None, timing_sync_cuda: bool = False, timing_level: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """und 编码的可训练段：VLM last-hidden → bridge + und_encoder →
        (und_token, grid_sizes, seq_lens)。

        供 prepare 级半缓存（train_steps_per_vae_encode>1 时 VLM 冻结段每
        batch 一次、随 prepared 复用）在 step 内调用；bridge/und_encoder
        可训练，梯度经本方法正常回传（VLM 段冻结、无梯度路径）。
        """
        fine = timing if (timing is not None and int(timing_level) >= 2) else None
        with timed_section(fine, "wm_und_encoder_seconds", timing_sync_cuda):
            und_tokens = self.bridge(vlm_hidden)
            und_tokens = self.und_encoder(und_tokens)

        bs = und_tokens.size(0)
        l_und = und_tokens.size(1)

        # seq_lens 语义为逐样本真实（未 padding）长度：und token 作为
        # attention K/V 时据此 mask 掉 padding 位（评审 #1）；und token
        # 作为 Q 侧时仍用全长（flash_attention 的 varlen 输出要求 q 全长）。
        seq_lens = real_lens

        # und 采用文本类 1D RoPE：grid = [L_und, 1, 1]（仅时间维位置）
        grid_sizes = torch.tensor([l_und, 1, 1]).repeat(bs, 1)

        return und_tokens, grid_sizes, seq_lens

    def encode(self, prompt_inputs: list[str], first_frame_pil: Optional[list[Image]],
               timing=None, timing_sync_cuda: bool = False, timing_level: int = 1) -> torch.Tensor:
        """完整 und 编码 = precompute_vlm_features（冻结段）+ encode_from_vlm_hidden
        （可训练段）。推理/评估路径仍用本方法；训练侧在 prepare 级半缓存下改为
        每 batch 一次 precompute + 每 step 一次 encode_from_vlm_hidden。"""
        vlm_hidden, real_lens = self.precompute_vlm_features(
            prompt_inputs, first_frame_pil,
            timing=timing, timing_sync_cuda=timing_sync_cuda, timing_level=timing_level)
        return self.encode_from_vlm_hidden(
            vlm_hidden, real_lens,
            timing=timing, timing_sync_cuda=timing_sync_cuda, timing_level=timing_level)

