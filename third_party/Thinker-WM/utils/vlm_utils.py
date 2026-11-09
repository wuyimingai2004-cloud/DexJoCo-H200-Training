# VLM Utilities
# Simple functions for VLM data preprocessing
#
# 统一 VLM task prompt（Motus）：
# - MOTUS_VLM_TASK_OUTPUT_FORMAT：Scene/Task/Action/Outcome 四字段固定顺序，
#   每字段严格一句、按固定句式模板生成；输出信息只能来自输入（首帧图像 +
#   任务文本），禁止自行推理/编造输入未包含的未来信息；某字段信息输入中
#   不存在时，该字段原位置输出 MOTUS_VLM_FIELD_PLACEHOLDER 占空文本。
# - build_vlm_task_message(task_text, drop_fields=...)：包装统一格式消息；
#   drop_fields 非空时被 drop 字段从格式指令中整字段移除（训练时字段
#   dropout 用，见 sample_vlm_field_dropout）。
# - sample_vlm_field_dropout(per_field_prob)：每字段独立伯努利 dropout，
#   保底留 1 字段；供训练框架逐样本调用。
# - build_vlm_rewrite_message(task_text, source_key)：vlm_text_rewrite 开启时
#   的压缩改写指令消息（纯文本、不输入首帧图；输出 Task/Action + Scene 若
#   有；按数据集家族分模板，egoscale 禁复述标注元数据）；输出由
#   validate_vlm_rewrite 校验，见文件末"改写"节。

from qwen_vl_utils import process_vision_info
from typing import Any, Collection, Dict, List, Optional, Tuple
import logging
import re
import torch

logger = logging.getLogger(__name__)


# robotwin 生产 catalog 文本的三视角样板文前缀（修复 #8，见
# tmp/problem_report_action_oscillation_20260810.md §3.3）：全部四个源
# （aloha/ur5/arx/piper）共享同一条 "The whole scene is in a realistic,
# industrial art style with three views: ... The <name> robot is currently
# performing the following task: <任务句>" 模板，且非 aloha 源的 robot 名
# 错写为 "The aloha robot"。长样板文 + 单一文本使 prompt 条件近似无信息；
# 剥离前缀只保留任务句（错写的 robot 名随前缀一起移除）。
_ROBOTWIN_VIEW_PREAMBLE_RE = re.compile(
    r"^\s*The whole scene is in a realistic, industrial art style with three views:.*?"
    r"currently performing the following task:\s*",
    re.IGNORECASE | re.DOTALL,
)


def strip_robotwin_view_preamble(text: Any) -> str:
    """剥离 robotwin 三视角样板文前缀，只留任务句；不匹配的文本原样返回（去首尾空白）。

    在 build_vlm_task_message 内统一调用——train（unified_vlm_task_prompt
    重建）、train/sample.py、periodic_eval、scripts/eval_video_ckpt.py 全部
    经该函数构建 VLM 消息，单点生效。
    """
    raw = str(text)
    stripped = _ROBOTWIN_VIEW_PREAMBLE_RE.sub("", raw, count=1).strip()
    return stripped if stripped else raw.strip()

def preprocess_vlm_messages(text_instruction: str, image_pil, processor):
    """
    Complete VLM preprocessing - create messages, process vision, and get final inputs.
    
    Args:
        text_instruction: Robot task instruction
        image_pil: PIL Image object
        processor: VLM processor (AutoProcessor)
        
    Returns:
        VLM inputs ready for model forward
    """
    # Create VLM messages format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_pil},
                {"type": "text", "text": text_instruction}
            ]
        }
    ]
    
    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Process vision info
    image_inputs, video_inputs = process_vision_info(messages)
    
    # Get final processor inputs
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    return inputs

def preprocess_vlm_text_only(text_instruction: str, processor):
    """纯文本 VLM 预处理（prompt_mode=text_only 的源，如 openvid/egoverse）。

    只用 VLM tokenizer + 语言模型编码任务文本：不带首帧图、不套四字段
    机器人模板。返回的 inputs 无 pixel_values/image_grid_thw。
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_instruction}
            ]
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return processor(text=[text], padding=True, return_tensors="pt")


def _process_vlm_inputs_batch(vlm_inputs: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Process and batch VLM inputs with padding."""
    # Extract components
    input_ids_list = [vlm_input['input_ids'] for vlm_input in vlm_inputs]
    pixel_values_list = [vlm_input.get('pixel_values') for vlm_input in vlm_inputs]
    image_grid_thw_list = [vlm_input.get('image_grid_thw') for vlm_input in vlm_inputs]
    attention_mask_list = [vlm_input.get('attention_mask') for vlm_input in vlm_inputs]
    
    # Pad input_ids to same length (simplified like model implementation)
    max_seq_len = max(ids.shape[1] for ids in input_ids_list)
    padded_input_ids = []
    padded_attention_masks = []
    
    for ids, mask in zip(input_ids_list, attention_mask_list):
        if ids.shape[1] < max_seq_len:
            padding_size = max_seq_len - ids.shape[1]
            # Pad input_ids
            padding = torch.zeros(ids.shape[0], padding_size, dtype=ids.dtype, device=ids.device)
            padded_ids = torch.cat([ids, padding], dim=1)
            # Pad attention_mask
            if mask is not None:
                mask_padding = torch.zeros(mask.shape[0], padding_size, dtype=mask.dtype, device=mask.device)
                padded_mask = torch.cat([mask, mask_padding], dim=1)
            else:
                padded_mask = None
        else:
            padded_ids = ids
            padded_mask = mask
            
        padded_input_ids.append(padded_ids)
        padded_attention_masks.append(padded_mask)
    
    # Batch everything
    return {
        'input_ids': torch.cat(padded_input_ids, dim=0),
        'pixel_values': torch.cat([pv for pv in pixel_values_list if pv is not None], dim=0) if pixel_values_list and any(pv is not None for pv in pixel_values_list) else None,
        'image_grid_thw': torch.cat([igt for igt in image_grid_thw_list if igt is not None], dim=0) if image_grid_thw_list and any(igt is not None for igt in image_grid_thw_list) else None,
        'attention_mask': torch.cat([mask for mask in padded_attention_masks if mask is not None], dim=0) if any(mask is not None for mask in padded_attention_masks) else None,
    }


def _process_language_embeddings_batch(language_embeddings: List[torch.Tensor], text_len: int = 512) -> torch.Tensor:
    """Process and batch language embeddings with padding."""
    padded_embeddings = []
    
    for emb in language_embeddings:
        if emb.shape[0] <= text_len:
            padded = torch.cat([emb, emb.new_zeros(text_len - emb.shape[0], emb.shape[1])])
        else:
            padded = emb[:text_len]
        padded_embeddings.append(padded)
    
    # Stack to [B, seq_len, dim]
    return torch.stack(padded_embeddings, dim=0)


# 统一 VLM task prompt（Motus）：数据集任务文本格式不一（robotwin 三视角
# preamble+任务句——preamble 由 strip_robotwin_view_preamble 在构建时剥离
# （修复 #8）、agibot_beta Scene+Task+segment 拼合、ewmbench 裸指令），
# 统一要求 VLM 以 Scene/Task/Action/Outcome 四字段（MOTUS_VLM_FIELD_NAMES
# 固定顺序）输出当前任务描述。严格格式约定：
#   - 每字段严格一句、按固定句式模板生成（见 _MOTUS_VLM_FIELD_FORMATS）；
#   - 输出信息只能来自输入（首帧图像 + 任务文本），禁止自行推理/编造输入
#     未包含的未来信息（逐字段 grounding）；
#   - 某字段信息输入中不存在时，该字段原位置输出占空文本
#     MOTUS_VLM_FIELD_PLACEHOLDER（字段标签与顺序不变）。
# 首帧图像由 preprocess_vlm_messages 作为 image content 插在文本之前，即下文
# "First frame" 所指。build_vlm_task_message 支持 drop_fields（训练时字段
# dropout，见 sample_vlm_field_dropout）：被 drop 字段从格式指令中整字段移除，
# 仅要求输出保留字段。
MOTUS_VLM_FIELD_NAMES = ("Scene", "Task", "Action", "Outcome")
MOTUS_VLM_FIELD_PLACEHOLDER = "Not specified in the input."

# 每字段的严格一句式模板（与 MOTUS_VLM_FIELD_NAMES 同序；被 drop 时整段移除）。
_MOTUS_VLM_FIELD_FORMATS = {
    "Scene": (
        "Scene: The robot is in/at <place>, with <key objects> on/around "
        "<surface or fixture>, as shown in the first frame."
    ),
    "Task": (
        "Task: The robot should/will <one action verb and its target>, "
        "e.g. 'move the mug from the shelf to the table'."
    ),
    "Action": (
        "Action: The robot will <one clause naming the arm(s) and the "
        "motion>, e.g. 'grasp the mug with its right arm and carry it to "
        "the table'."
    ),
    "Outcome": (
        "Outcome: The <key object> will be <one clause naming its final "
        "position and state>, e.g. 'on the table, upright and empty'."
    ),
}

MOTUS_VLM_TASK_OUTPUT_FORMAT = " ".join(
    _MOTUS_VLM_FIELD_FORMATS[field] for field in MOTUS_VLM_FIELD_NAMES
)


# prompt 模式（dataset.sources[*].prompt_mode）：
# - vlm_image_text（缺省）：首帧图 + 四字段模板包装文本，冻结 VLM 图文编码；
# - text_only：纯文本路径（openvid caption / egoverse 标注等非机器人任务
#   文本），只用 VLM tokenizer + 语言模型编码，不带首帧图、不套模板。
PROMPT_MODE_VLM_IMAGE_TEXT = "vlm_image_text"
PROMPT_MODE_TEXT_ONLY = "text_only"
_KNOWN_PROMPT_MODES = (PROMPT_MODE_VLM_IMAGE_TEXT, PROMPT_MODE_TEXT_ONLY)


def resolve_prompt_mode(config: Any, source_keys: Any) -> str:
    """解析当前 batch 的 prompt 模式（batch 内单源，由 motus_data collate 保证）。

    source_keys 取 batch 的 source_datasets（或调用方已知的 source name）首个
    元素。注意 batch 的 source_datasets 来自 catalog 元数据的 source 字段，是
    **家族名**（egoverse/libero/openvid/agibot_beta/robotwin），而
    config.dataset.sources 的 name/dataset 是 child 名（如
    egoverse_aria_bimanual）——因此除 name/dataset 精确匹配外，还按
    ``dataset == key 或 dataset 以 key+"_" 开头`` 做家族前缀匹配。同一家族
    多个 child 的 prompt_mode 必须一致，冲突直接报错；查不到一律回退
    vlm_image_text（历史行为）。
    """
    if not source_keys:
        return PROMPT_MODE_VLM_IMAGE_TEXT
    sources = getattr(getattr(config, "dataset", None), "sources", None)
    if not sources:
        return PROMPT_MODE_VLM_IMAGE_TEXT
    key = str(source_keys[0])
    matched_modes: set[str] = set()
    for source in sources:
        name = str(source.get("name") or "")
        dataset = str(source.get("dataset") or "")
        if key == name or key == dataset or dataset.startswith(key + "_"):
            mode = str(source.get("prompt_mode") or PROMPT_MODE_VLM_IMAGE_TEXT)
            if mode not in _KNOWN_PROMPT_MODES:
                raise ValueError(
                    f"source {name or dataset!r} has unknown prompt_mode={mode!r} "
                    f"(known: {list(_KNOWN_PROMPT_MODES)})"
                )
            matched_modes.add(mode)
    if len(matched_modes) > 1:
        raise ValueError(
            f"batch source {key!r} 匹配到多个 prompt_mode 不一致的 source："
            f"{sorted(matched_modes)}（同一家族的 prompt_mode 必须一致）"
        )
    if matched_modes:
        return matched_modes.pop()
    return PROMPT_MODE_VLM_IMAGE_TEXT


def _validate_drop_fields(drop_fields: Collection[str]) -> None:
    """校验 drop_fields ⊆ MOTUS_VLM_FIELD_NAMES 且不得覆盖全部四字段。"""
    invalid = set(drop_fields) - set(MOTUS_VLM_FIELD_NAMES)
    if invalid:
        raise ValueError(
            f"drop_fields contains invalid field(s): {sorted(invalid)}; "
            f"valid fields: {list(MOTUS_VLM_FIELD_NAMES)}"
        )
    if len(set(drop_fields)) >= len(MOTUS_VLM_FIELD_NAMES):
        raise ValueError(
            "drop_fields must not cover all four fields; "
            "at least one field must be kept"
        )


VLM_PROMPT_STYLE_FOUR_FIELD = "motus_four_field"
VLM_PROMPT_STYLE_ROBOTWIN_TLAYOUT = "robotwin_tlayout"
VLM_PROMPT_STYLE_DEXJOCO_TLAYOUT = "dexjoco_tlayout"
_KNOWN_VLM_PROMPT_STYLES = (
    VLM_PROMPT_STYLE_FOUR_FIELD,
    VLM_PROMPT_STYLE_ROBOTWIN_TLAYOUT,
    VLM_PROMPT_STYLE_DEXJOCO_TLAYOUT,
)


def resolve_vlm_prompt_style(config: Any) -> str:
    """``model.vlm_prompt_style``；缺省四字段壳（旧行为）。"""
    raw = getattr(getattr(config, "model", None), "vlm_prompt_style", None)
    style = str(raw or VLM_PROMPT_STYLE_FOUR_FIELD).strip()
    if style not in _KNOWN_VLM_PROMPT_STYLES:
        raise ValueError(
            f"unknown vlm_prompt_style={style!r}; "
            f"known: {list(_KNOWN_VLM_PROMPT_STYLES)}"
        )
    return style


def build_vlm_task_message(
    task_text: str,
    drop_fields: Collection[str] = frozenset(),
    style: str = VLM_PROMPT_STYLE_FOUR_FIELD,
) -> str:
    """把数据集原始任务文本包装为 VLM 条件消息。

    ``style="motus_four_field"``（默认）：混源 Motus 四字段壳。
    ``style="robotwin_tlayout"``：一句 T 形布局 + 任务句。``drop_fields``
    对此 style 无效（没有四字段可 drop）。

    drop_fields 非空时，被 drop 字段从格式指令中整字段移除（保留字段的
    标签与顺序不变），仅要求输出保留字段；默认空集 = 输出完整四字段，
    与历史调用（只传 task_text）逐字节兼容。drop_fields 必须是
    MOTUS_VLM_FIELD_NAMES 的子集且不得覆盖全部四个字段（否则 ValueError）。
    """
    # robotwin 三视角样板文剥离（修复 #8）：只留任务句进入格式指令。
    task_text = strip_robotwin_view_preamble(task_text)
    if style == VLM_PROMPT_STYLE_DEXJOCO_TLAYOUT:
        return (
            "The image is a T-layout: top is the external camera. "
            "For one arm the wrist view is bottom-center; for two arms, "
            "the right wrist is bottom-left and the left wrist is bottom-right.\n"
            f"Task: {str(task_text).strip()}"
        )
    if style == VLM_PROMPT_STYLE_ROBOTWIN_TLAYOUT:
        return (
            "The image is a T-layout: top is the head camera, "
            "bottom-left is the left wrist camera, "
            "bottom-right is the right wrist camera.\n"
            f"Task: {str(task_text).strip()}"
        )
    if style != VLM_PROMPT_STYLE_FOUR_FIELD:
        raise ValueError(
            f"unknown vlm prompt style={style!r}; "
            f"known: {list(_KNOWN_VLM_PROMPT_STYLES)}"
        )
    _validate_drop_fields(drop_fields)
    output_format = " ".join(
        _MOTUS_VLM_FIELD_FORMATS[field]
        for field in MOTUS_VLM_FIELD_NAMES
        if field not in drop_fields
    )
    return (
        "You are given the first frame of a robot episode and a task text. "
        "The task text may be a bare instruction, or may contain fields like "
        "'Scene Description' / 'Task' / 'Action'. Based on BOTH the image and "
        "the task text, output exactly in the following format (one sentence "
        f"per field): {output_format}\n"
        "Hard rules:\n"
        "- Ground every field ONLY in the input: the first frame image and "
        "the task text. Do not invent details that are not shown or stated.\n"
        "- Do NOT infer or fabricate future information that is not present "
        "in the input (e.g. do not assume the task will succeed).\n"
        f"- If a field is not specified in the input, write "
        f"'{MOTUS_VLM_FIELD_PLACEHOLDER}' in that field, keeping its label "
        "and position.\n"
        "Describe the Scene from the image instead of copying the task text; "
        "do not output anything else.\n"
        "First frame: <the first frame image provided with this message>.\n"
        f"Task text: {str(task_text).strip()}"
    )


def sample_vlm_field_dropout(
    per_field_prob: float,
    generator: Optional[torch.Generator] = None,
) -> frozenset:
    """按 per_field_prob 对四字段独立采样 dropout 集合（训练时逐样本调用）。

    每字段独立伯努利（概率 per_field_prob）；若四个字段全部被抽中，随机
    保留其中一个（保底留 1 字段，输出格式不会为空）。per_field_prob <= 0
    时恒返回空集（无 dropout）。
    """
    if per_field_prob <= 0.0:
        return frozenset()
    dropped = frozenset(
        field
        for field in MOTUS_VLM_FIELD_NAMES
        if torch.rand((), generator=generator).item() < per_field_prob
    )
    if len(dropped) == len(MOTUS_VLM_FIELD_NAMES):
        keep = MOTUS_VLM_FIELD_NAMES[
            int(torch.randint(len(MOTUS_VLM_FIELD_NAMES), (1,), generator=generator).item())
        ]
        dropped = frozenset(field for field in MOTUS_VLM_FIELD_NAMES if field != keep)
    return dropped


# ---------------------------------------------------------------------------
# VLM 文本改写消息（vlm_text_rewrite，20260819 v2，UndBranch._rewrite_prompts）
#
# 与 build_vlm_task_message（四字段条件消息，消息本身即编码对象）用途不同：
# 开启 vlm_text_rewrite 后，消息是给 VLM 的**压缩改写指令**——纯文本（不输
# 入首帧图）命令 VLM 把原始任务文本压缩为 Task/Action（+Scene，仅当输入含
# 场景信息）短文本，生成结果再走纯文本编码为 und token（token 数由
# vlm_rewrite_max_new_tokens 硬上限保证，分段文本改造后原始消息可超 uni
# 分支 rope freqs 1024 行上限）。
# 按数据集家族分模板：egoscale 原始输入含 Task name/Task label/Action
# label/Hand/Scene semantic/Camera motion 标注行——20260819 实测不处理时
# VLM 在字段开头后整段粘贴这些元数据、并在 token 上限处截断的退化模式；
# 经用户拍板，egoscale 输入进 prompt 前先整行剥离 Task label/Action
# label/Hand/Camera motion 元数据行（_strip_egoscale_metadata_lines），
# 模板只对保留的内容行（Task name/Task/Action/Scene semantic）下指令。

# 改写输出不合法（缺 Task/Action 行或复读元数据）时的安全占位文本：短、
# 格式合法、语义中性。不回退长原文——长原文可能再次超 rope 1024 上限。
MOTUS_VLM_REWRITE_FALLBACK = (
    "Task: Not specified in the input.\nAction: Not specified in the input."
)

_MOTUS_VLM_REWRITE_RULES = (
    "Hard rules:\n"
    "- Compress aggressively: keep only the task, the motion, and (if present) "
    "the scene. Drop metadata, labels, enumerations and repetitions.\n"
    "- Keep each field within 25 words.\n"
    "- NEVER fabricate: every field must contain ONLY information explicitly "
    "stated in the input text. If a detail is not stated in the input, leave "
    "it out — never guess. Do NOT predict or assume future outcomes.\n"
    "- Output the field lines only, nothing else."
)

_MOTUS_VLM_REWRITE_INSTRUCTION = (
    "Compress the following task text into a short standard description.\n"
    "Output format (one line per field):\n"
    "Task: <one concise sentence stating the task the robot should perform>\n"
    "Action: <one concise sentence describing the motion and the target "
    "object(s)>\n"
    "Scene: <one concise sentence naming the place and key objects; output "
    "this line ONLY when the input text contains scene/environment "
    "information, otherwise omit the Scene line entirely>\n"
    + _MOTUS_VLM_REWRITE_RULES
)

_MOTUS_VLM_REWRITE_INSTRUCTION_EGOSCALE = (
    "Compress the following egocentric activity annotation into a short "
    "standard description.\n"
    "The input lists a task overview ('Task name'/'Task') followed by timed "
    "segments, each with an 'Action' line (the motion source) and a 'Scene "
    "semantic' line (the scene source).\n"
    "Output format (one line per field):\n"
    "Task: <one concise sentence stating the task the person performs, from "
    "the 'Task'/'Task name' overview>\n"
    "Action: <one concise sentence compressing the segment 'Action' lines "
    "into the overall motion, naming the hand(s) and the target object(s)>\n"
    "Scene: <one concise sentence naming the place and key objects, grounded "
    "in 'Scene semantic' or the task overview; omit the Scene line entirely "
    "if no scene information is present>\n"
    + _MOTUS_VLM_REWRITE_RULES
)

# egoscale 输入行级过滤（20260819 用户拍板："Task label/Hand/Camera motion
# 根本没必要描述"）：这些标注元数据行进 prompt 前整行剥离——VLM 看不到也
# 就不会复读（v1 退化模式的治本）；保留 Task name/Task 概述与逐片段
# Action/Scene semantic 内容行。Action 行内的 "Left hand: ..." 分工描述是
# 内容、不是元数据行，不受影响（正则只匹配行首标签）。
_EGOSCALE_METADATA_LINE_RE = re.compile(
    r"^\s*(?:task label|action label|hand|camera motion)\s*:", re.IGNORECASE)


def _strip_egoscale_metadata_lines(text: str) -> str:
    """剥离 egoscale 文本的标注元数据行（见上正则），其余行原样保留。"""
    return "\n".join(
        line for line in str(text).splitlines()
        if not _EGOSCALE_METADATA_LINE_RE.match(line)
    )

# 家族名 → 专用改写指令（source_key 小写前缀匹配）；未命中用默认指令。
_MOTUS_VLM_REWRITE_INSTRUCTIONS_BY_FAMILY = {
    "egoscale": _MOTUS_VLM_REWRITE_INSTRUCTION_EGOSCALE,
}


def build_vlm_rewrite_message(task_text: str, source_key: Any = None) -> str:
    """把数据集原始任务文本包装为 VLM 压缩改写指令消息（vlm_text_rewrite）。

    source_key 取 batch 的 source_datasets[0]（家族名，如 egoscale/libero）
    或评估侧的源名（child 名，如 egoscale_first500h_shard00）——均按小写
    前缀匹配家族模板；未命中用默认模板。egoscale 家族的输入文本先经
    _strip_egoscale_metadata_lines 剥离标注元数据行（Task label/Action
    label/Hand/Camera motion）。输出字段固定为 Task/Action（+Scene 仅当
    输入含场景信息）；与 build_vlm_task_message 的四字段条件消息是两个
    用途，不要混用。
    """
    key = str(source_key or "").strip().lower()
    instruction = _MOTUS_VLM_REWRITE_INSTRUCTION
    family = None
    for name, family_instruction in _MOTUS_VLM_REWRITE_INSTRUCTIONS_BY_FAMILY.items():
        if key.startswith(name):
            instruction = family_instruction
            family = name
            break
    text = str(task_text).strip()
    if family == "egoscale":
        text = _strip_egoscale_metadata_lines(text)
    return f"{instruction}\nInput text:\n{text}"


# 改写输出合法性校验（UndBranch._rewrite_prompts 用）：必须有 Task:/Action:
# 行，且不得复读输入的标注元数据行（egoscale 退化模式，见上）。
_VLM_REWRITE_FORBIDDEN_LINE_PREFIXES = (
    "task name",
    "task label",
    "action label",
    "hand",
    "scene semantic",
    "camera motion",
)


def validate_vlm_rewrite(text: Any) -> bool:
    """改写输出是否合法：非空、含 Task:/Action: 行、无标注元数据复读。"""
    lines = [line.strip().lower() for line in str(text).splitlines() if line.strip()]
    if not any(line.startswith("task:") for line in lines):
        return False
    if not any(line.startswith("action:") for line in lines):
        return False
    return not any(
        line.startswith(prefix)
        for line in lines
        for prefix in _VLM_REWRITE_FORBIDDEN_LINE_PREFIXES
    )


# ---------------------------------------------------------------------------
# 输出侧字段 dropout（20260818_v2 起，用户拍板）：vlm_text_rewrite 开启时
# vlm_field_dropout_prob 改在改写输出文本上作用（rewrite 之后、编码之前），
# 不再改输入 prompt——缓存存完整改写输出，dropout 在缓存之后作用，同一消息
# 的 generate 仍只跑一次，dropout 不产生新 cache key、不增加 VLM 推理。

# 改写输出的字段行（Task:/Action:/Scene:，见 _MOTUS_VLM_REWRITE_INSTRUCTION）。
_MOTUS_REWRITE_OUTPUT_FIELD_RE = re.compile(r"^\s*(task|action|scene)\s*:", re.IGNORECASE)


def apply_field_dropout_to_rewrite(
    text: Any,
    per_field_prob: float,
    generator: Optional[torch.Generator] = None,
) -> str:
    """对 VLM 改写输出文本按字段行随机 dropout（整行移除）。

    逐字段行独立伯努利（概率 per_field_prob）；字段行全部被命中时随机保留
    其中一行（保底留 1 字段行，与输入侧 sample_vlm_field_dropout 同语义）；
    非字段行原样保留、相对顺序不变；无字段行的文本（text_only 源 caption
    等无结构文本）原样返回。per_field_prob <= 0 时恒原样返回。
    """
    text = str(text)
    if per_field_prob <= 0.0:
        return text
    lines = text.splitlines()
    field_idx = [
        i for i, line in enumerate(lines) if _MOTUS_REWRITE_OUTPUT_FIELD_RE.match(line)
    ]
    if not field_idx:
        return text
    dropped = {
        i for i in field_idx if torch.rand((), generator=generator).item() < per_field_prob
    }
    if len(dropped) == len(field_idx):
        keep = field_idx[
            int(torch.randint(len(field_idx), (1,), generator=generator).item())
        ]
        dropped.discard(keep)
    return "\n".join(line for i, line in enumerate(lines) if i not in dropped)
