"""Rank-local ready-batch queue for hiding DataLoader latency.

The queue wraps an already-created DataLoader. A background producer thread
calls ``next(dataloader_iter)``, optionally clones and transforms the fully
collated CPU batch, then stores it in a bounded queue. The training loop
consumes ready batches from this queue.

This does not change sample semantics, source sampling, or batch composition.
It only moves DataLoader iteration off the training thread and exposes queue
health metrics for timing analysis.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch


@dataclass
class _ProducerError:
    exc: BaseException
    traceback_text: str


class ReadyBatchQueue:
    """A bounded rank-local queue of fully collated DataLoader batches."""

    def __init__(
        self,
        dataloader,
        max_size: int = 2,
        get_timeout_seconds: float = 600.0,
        low_watermark: int = 0,
        name: str = "train",
        clone_tensors: bool = False,
        producer_transform: Optional[Callable[[Any], Any]] = None,
    ):
        if max_size <= 0:
            raise ValueError("ReadyBatchQueue max_size must be positive")
        if get_timeout_seconds <= 0:
            raise ValueError("ReadyBatchQueue get_timeout_seconds must be positive")

        self.dataloader = dataloader
        self.max_size = int(max_size)
        self.get_timeout_seconds = float(get_timeout_seconds)
        self.low_watermark = int(low_watermark)
        self.name = str(name)
        self.clone_tensors = bool(clone_tensors)
        self.producer_transform = producer_transform

        self._queue: queue.Queue = queue.Queue(maxsize=self.max_size)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[_ProducerError] = None
        self._iterator = None
        self._epoch = 0

        self._produced_count = 0
        self._consumed_count = 0
        self._empty_count = 0
        self._low_watermark_count = 0
        self._producer_restart_count = 0
        self._producer_put_wait_total_seconds = 0.0
        self._consumer_wait_total_seconds = 0.0
        self._latest_producer_batch_seconds = None
        self._latest_producer_dataloader_seconds = None
        self._latest_producer_clone_seconds = None
        self._latest_producer_transform_seconds = None
        self._latest_producer_put_wait_seconds = None
        self._producer_state = "init"
        self._producer_state_since = time.time()
        self._latest_producer_exception_type = None
        self._latest_producer_exception_message = None

    def _set_state(self, state: str) -> None:
        with self._lock:
            self._producer_state = str(state)
            self._producer_state_since = time.time()

    @staticmethod
    def _clone_tensor_tree(obj: Any) -> Any:
        """Detach CPU batch tensors from DataLoader worker shared-memory storage.

        PyTorch multiprocessing DataLoader can hand the main process tensors backed
        by /dev/shm. If the ready queue stores those tensors directly, a full queue
        can keep shared-memory segments alive longer than needed. This clone is
        opt-in because it costs CPU memory bandwidth, but it is useful for
        long-running stability experiments.
        """
        if torch.is_tensor(obj):
            return obj.detach().clone()
        if isinstance(obj, Mapping):
            return {key: ReadyBatchQueue._clone_tensor_tree(value) for key, value in obj.items()}
        if isinstance(obj, tuple):
            return tuple(ReadyBatchQueue._clone_tensor_tree(value) for value in obj)
        if isinstance(obj, list):
            return [ReadyBatchQueue._clone_tensor_tree(value) for value in obj]
        return obj

    def start(self) -> "ReadyBatchQueue":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._producer_loop,
            name=f"ReadyBatchQueueProducer-{self.name}",
            daemon=True,
        )
        self._thread.start()
        return self

    @property
    def started(self) -> bool:
        return self._thread is not None

    def close(self, timeout: float = 10.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def __enter__(self) -> "ReadyBatchQueue":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _set_epoch_if_supported(self, epoch: int) -> None:
        sampler = getattr(self.dataloader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        batch_sampler = getattr(self.dataloader, "batch_sampler", None)
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch)

    def _new_iterator(self):
        return iter(self.dataloader)

    def _set_error(self, exc: BaseException) -> None:
        with self._lock:
            if self._error is None:
                self._error = _ProducerError(exc=exc, traceback_text=traceback.format_exc())
                self._latest_producer_exception_type = type(exc).__name__
                self._latest_producer_exception_message = str(exc)
                self._producer_state = "error"
                self._producer_state_since = time.time()

    def _raise_if_error(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise RuntimeError(
                "ReadyBatchQueue producer failed:\n" + error.traceback_text
            ) from error.exc

    def _producer_loop(self) -> None:
        try:
            self._set_state("creating_iterator")
            self._iterator = self._new_iterator()
            while not self._stop_event.is_set():
                batch_start = time.perf_counter()
                try:
                    self._set_state("waiting_dataloader")
                    batch = next(self._iterator)
                    dataloader_seconds = time.perf_counter() - batch_start
                except StopIteration:
                    self._epoch += 1
                    self._set_epoch_if_supported(self._epoch)
                    self._set_state("creating_iterator")
                    self._iterator = self._new_iterator()
                    with self._lock:
                        self._producer_restart_count += 1
                    continue

                clone_start = time.perf_counter()
                if self.clone_tensors:
                    self._set_state("cloning_batch")
                    batch = self._clone_tensor_tree(batch)
                clone_seconds = time.perf_counter() - clone_start

                transform_start = time.perf_counter()
                if self.producer_transform is not None:
                    self._set_state("transforming_batch")
                    batch = self.producer_transform(batch)
                transform_seconds = time.perf_counter() - transform_start

                put_start = time.perf_counter()
                while not self._stop_event.is_set():
                    try:
                        self._set_state("waiting_queue_put")
                        self._queue.put(batch, timeout=0.5)
                        break
                    except queue.Full:
                        continue
                put_wait = time.perf_counter() - put_start
                if self._stop_event.is_set():
                    break

                with self._lock:
                    self._produced_count += 1
                    self._latest_producer_dataloader_seconds = float(dataloader_seconds)
                    self._latest_producer_clone_seconds = float(clone_seconds)
                    self._latest_producer_transform_seconds = float(transform_seconds)
                    self._latest_producer_batch_seconds = float(time.perf_counter() - batch_start)
                    self._latest_producer_put_wait_seconds = float(put_wait)
                    self._producer_put_wait_total_seconds += float(put_wait)
                    self._producer_state = "idle"
                    self._producer_state_since = time.time()
                batch = None
        except BaseException as exc:
            self._set_error(exc)

    def next(self) -> Tuple[Any, Dict[str, Any]]:
        self._raise_if_error()
        before_get = self._queue.qsize()
        was_empty = before_get <= 0
        get_start = time.perf_counter()
        deadline = get_start + self.get_timeout_seconds

        item = None
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                self._raise_if_error()
                raise TimeoutError(
                    f"ReadyBatchQueue timed out after {self.get_timeout_seconds:.1f}s "
                    f"waiting for a batch; snapshot={self.snapshot()}"
                )
            try:
                item = self._queue.get(timeout=min(0.5, remaining))
                break
            except queue.Empty:
                self._raise_if_error()

        wait_seconds = time.perf_counter() - get_start
        after_get = self._queue.qsize()

        with self._lock:
            self._consumed_count += 1
            if was_empty:
                self._empty_count += 1
            if after_get <= self.low_watermark:
                self._low_watermark_count += 1
            self._consumer_wait_total_seconds += float(wait_seconds)

            stats = {
                "ready_queue_enabled": True,
                "ready_queue_name": self.name,
                "ready_queue_max_size": self.max_size,
                "ready_queue_clone_tensors": bool(self.clone_tensors),
                "ready_queue_epoch": int(self._epoch),
                "ready_queue_size_before_get": int(before_get),
                "ready_queue_size_after_get": int(after_get),
                "ready_queue_was_empty": bool(was_empty),
                "ready_queue_get_wait_seconds": float(wait_seconds),
                "ready_queue_low_watermark": int(self.low_watermark),
                "ready_queue_produced_count": int(self._produced_count),
                "ready_queue_consumed_count": int(self._consumed_count),
                "ready_queue_empty_count": int(self._empty_count),
                "ready_queue_low_watermark_count": int(self._low_watermark_count),
                "ready_queue_producer_restart_count": int(self._producer_restart_count),
                "ready_queue_wait_total_seconds": float(self._consumer_wait_total_seconds),
                "ready_queue_producer_put_wait_total_seconds": float(
                    self._producer_put_wait_total_seconds
                ),
                "ready_queue_producer_batch_seconds": self._latest_producer_batch_seconds,
                "ready_queue_producer_dataloader_seconds": self._latest_producer_dataloader_seconds,
                "ready_queue_producer_clone_seconds": self._latest_producer_clone_seconds,
                "ready_queue_producer_transform_seconds": self._latest_producer_transform_seconds,
                "ready_queue_producer_put_wait_seconds": self._latest_producer_put_wait_seconds,
                "ready_queue_producer_state": self._producer_state,
                "ready_queue_producer_state_since": self._producer_state_since,
                "ready_queue_producer_state_age_seconds": float(time.time() - self._producer_state_since),
                "ready_queue_producer_exception_type": self._latest_producer_exception_type,
                "ready_queue_producer_exception_message": self._latest_producer_exception_message,
            }

        return item, stats

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "ready_queue_enabled": True,
                "ready_queue_name": self.name,
                "ready_queue_max_size": self.max_size,
                "ready_queue_clone_tensors": bool(self.clone_tensors),
                "ready_queue_current_size": int(self._queue.qsize()),
                "ready_queue_epoch": int(self._epoch),
                "ready_queue_produced_count": int(self._produced_count),
                "ready_queue_consumed_count": int(self._consumed_count),
                "ready_queue_empty_count": int(self._empty_count),
                "ready_queue_low_watermark_count": int(self._low_watermark_count),
                "ready_queue_producer_restart_count": int(self._producer_restart_count),
                "ready_queue_wait_total_seconds": float(self._consumer_wait_total_seconds),
                "ready_queue_producer_put_wait_total_seconds": float(
                    self._producer_put_wait_total_seconds
                ),
                "ready_queue_producer_batch_seconds": self._latest_producer_batch_seconds,
                "ready_queue_producer_dataloader_seconds": self._latest_producer_dataloader_seconds,
                "ready_queue_producer_clone_seconds": self._latest_producer_clone_seconds,
                "ready_queue_producer_transform_seconds": self._latest_producer_transform_seconds,
                "ready_queue_producer_put_wait_seconds": self._latest_producer_put_wait_seconds,
                "ready_queue_producer_state": self._producer_state,
                "ready_queue_producer_state_since": self._producer_state_since,
                "ready_queue_producer_state_age_seconds": float(time.time() - self._producer_state_since),
                "ready_queue_producer_exception_type": self._latest_producer_exception_type,
                "ready_queue_producer_exception_message": self._latest_producer_exception_message,
            }
