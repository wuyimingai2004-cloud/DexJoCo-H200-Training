# VLM 文本改写结果的磁盘缓存（model.vlm_rewrite_cache_dir，20260818_v2 起）。
#
# 背景：vlm_text_rewrite 开启时 UndBranch._rewrite_prompts 每步对全 batch 跑
# 一次贪心 generate；同一 batch 经 train_steps_per_vae_encode 复用 N 步、同一
# 消息跨 epoch/运行重复出现（hash 的是最终消息文本——同 episode 不同窗口若
# 重叠片段集相同则消息相同，天然去重）。缓存把 generate 摊到"每个不同消息
# 一次"。
#
# 结构（高效读写、无锁、共享文件系统安全）：
#   {cache_dir}/{sha256[:2]}/{sha256[2:4]}/{sha256}.json
# key = sha256(f"{model_tag}\n{message}")——model_tag（VLM 权重目录名）与消息
# 内的家族模板指令共同构成命名空间，换模型/改模板旧条目自然失效（需人工清理
# 目录）。写 = 同目录临时文件 + os.replace 原子改名：多 rank/多进程并发写同
# key 时内容相同、last-wins，无需任何锁；读 = 直接 open，O(1)。
#
# 所有 I/O 异常兜底为 miss/跳过写（只 warning 一次），绝不影响训练主路径。

from typing import Any, Dict, List, Optional, Tuple
import json
import hashlib
import logging
import os
import threading
import uuid

logger = logging.getLogger(__name__)


class VLMRewriteCache:
    """按消息文本 hash 的改写结果分片文件缓存（进程间共享、无锁）。"""

    def __init__(self, cache_dir: str, model_tag: str):
        self.cache_dir = str(cache_dir)
        self.model_tag = str(model_tag)
        self._warned = False

    def key_for(self, message: str) -> str:
        return hashlib.sha256(
            f"{self.model_tag}\n{message}".encode("utf-8")
        ).hexdigest()

    def _path_for(self, key: str) -> str:
        return os.path.join(self.cache_dir, key[:2], key[2:4], f"{key}.json")

    def _warn_once(self, what: str, exc: Exception) -> None:
        if not self._warned:
            self._warned = True
            logger.warning("VLM 改写缓存 %s 失败（后续同类异常静默，缓存降级为旁路）: %s", what, exc)

    def get_many(self, messages: List[str]) -> Tuple[List[Optional[str]], int, int]:
        """批量查询。返回 (texts, hits, misses)：texts[i] 为缓存文本或 None。"""
        texts: List[Optional[str]] = [None] * len(messages)
        hits = 0
        for i, message in enumerate(messages):
            path = self._path_for(self.key_for(message))
            try:
                with open(path, "r", encoding="utf-8") as f:
                    record: Dict[str, Any] = json.load(f)
                text = record.get("text")
                if isinstance(text, str):
                    texts[i] = text
                    hits += 1
            except FileNotFoundError:
                pass
            except Exception as exc:  # 损坏 JSON / FS 异常 → 按 miss 处理
                self._warn_once("读取", exc)
        return texts, hits, len(messages) - hits

    def set_many(self, items: List[Tuple[str, str, bool]]) -> None:
        """批量写入 (message, text, validated)。任何单条失败不影响其他。"""
        for message, text, _validated in items:
            key = self.key_for(message)
            path = self._path_for(key)
            if os.path.exists(path):
                continue  # 同 key 内容确定（greedy），已存在则跳过
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                record = {
                    "model": self.model_tag,
                    "message": message,
                    "text": text,
                    "validated": bool(_validated),
                }
                # 同目录临时文件 + 原子改名：并发写安全（同 key 内容相同）。
                tmp = os.path.join(
                    os.path.dirname(path),
                    f".tmp-{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex}.json",
                )
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(record, f, ensure_ascii=False)
                os.replace(tmp, path)
            except Exception as exc:
                self._warn_once("写入", exc)
