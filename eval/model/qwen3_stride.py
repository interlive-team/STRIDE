#
# Copyright (C) 2025 InterLive Team. All Rights Reserved.
#
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoProcessor
from transformers.cache_utils import DynamicCache

from stride.model import Qwen3VLForSTRIDE
from stride.utils.input_processing import (
    VideoSpec,
    resize_image,
    resolve_resolution,
)

from . import register_model
from .base import BaseTriggerModel


@dataclass
class TrackedSegment:
    start_sec: float
    end_sec: float
    is_complete: bool = False


@dataclass
class STRIDETriggerArguments:
    chunk_size: int = field(
        default=0,
        metadata={"help": "Frames per chunk. 0 = auto (trigger_window_past // 2)"},
    )
    max_window_size: int = field(
        default=256, metadata={"help": "Max frames in the mega-window"}
    )
    stride: int = field(
        default=128,
        metadata={"help": "Frames to drop when mega-window is full"},
    )
    unmasking_steps: int = field(
        default=8, metadata={"help": "Diffusion unmasking iterations"}
    )
    confidence_threshold: Optional[str] = field(
        default="0.75",
        metadata={
            "help": "Min confidence to retain previous slot prediction (None = disable)"
        },
    )
    margin_seconds: float = field(
        default=1.0,
        metadata={"help": "Time window before last frame that must be inactive"},
    )
    max_triggers: int = field(
        default=50, metadata={"help": "Max segments for multi-event"}
    )
    frame_spatial_patch: int = field(default=32)
    frame_max_tokens: int = field(default=256)

    def __post_init__(self):
        if isinstance(self.confidence_threshold, str):
            self.confidence_threshold = (
                None
                if self.confidence_threshold.lower() == "none"
                else float(self.confidence_threshold)
            )


@register_model("qwen3_stride")
class STRIDEQwen3VL(BaseTriggerModel, Qwen3VLForSTRIDE):
    TRIGGER_ARGUMENTS = STRIDETriggerArguments
    MOCK_FPS = 100
    SPECIAL_TOKENS = ["<activation>", "<active>", "<inactive>"]
    FRAME_TIME_PATCH = 2

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.trigger_args = STRIDETriggerArguments()
        self._frame_time_patch: int = self.FRAME_TIME_PATCH
        self._processor: Optional[AutoProcessor] = None
        self._resolution_cache: Dict[tuple, tuple] = {}
        self._query: str = ""
        self._is_multi_event: bool = False

    @classmethod
    def from_pretrained_with_processor(
        cls,
        model_path: str,
        device: str = "cuda",
        torch_dtype=torch.float16,
        **kwargs,
    ) -> "STRIDEQwen3VL":
        model = cls.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
            **kwargs,
        )
        processor = AutoProcessor.from_pretrained(model_path)
        vocab = processor.tokenizer.get_vocab()
        missing = [t for t in cls.SPECIAL_TOKENS if t not in vocab]
        if missing:
            raise ValueError(
                f"Pretrained tokenizer at {model_path} is missing special tokens "
                f"{missing}. The model must be trained with these tokens."
            )
        model._activation_token_id = processor.tokenizer.convert_tokens_to_ids(
            "<activation>"
        )
        model._active_token_id = processor.tokenizer.convert_tokens_to_ids("<active>")
        model._inactive_token_id = processor.tokenizer.convert_tokens_to_ids(
            "<inactive>"
        )
        model._processor = processor
        model = model.to(device).eval()
        return model

    def start_stream(self, query, is_multi_event=False):
        self._query = query
        self._is_multi_event = is_multi_event
        self._slot_token_ids = np.array(
            [
                self._inactive_token_id,
                self._active_token_id,
                self._activation_token_id,
            ]
        )
        self._slot_history: Dict[float, int] = {}
        self._active_segments: List[TrackedSegment] = []

    def compute_dst_resolution(self, src_resolution: tuple) -> tuple:
        if src_resolution in self._resolution_cache:
            return self._resolution_cache[src_resolution]

        spec = VideoSpec(
            path="",
            fps=1.0,
            start_seconds=0.0,
            end_seconds=1.0,
            src_resolution=src_resolution,
            num_frames=self._frame_time_patch,
        )
        dst_resolution, _ = resolve_resolution(
            spec,
            frame_time_patch=self._frame_time_patch,
            frame_spatial_patch=self.trigger_args.frame_spatial_patch,
            frame_max_tokens=self.trigger_args.frame_max_tokens,
        )
        result = tuple(dst_resolution)
        self._resolution_cache[src_resolution] = result
        return result

    def preprocess_frames(self, frames: np.ndarray) -> np.ndarray:
        src_resolution = (frames.shape[1], frames.shape[2])
        dst_resolution = self.compute_dst_resolution(src_resolution)
        if (
            frames.shape[1] == dst_resolution[0]
            and frames.shape[2] == dst_resolution[1]
        ):
            return frames
        return np.stack([resize_image(f, dst_resolution) for f in frames])

    def _build_prefix_cache(
        self,
        frames: np.ndarray,
        timestamps: np.ndarray,
    ) -> tuple:
        device = next(self.parameters()).device
        assert self._query
        inputs = self._processor.apply_chat_template(
            [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self._query},
                            {"type": "video", "video": frames},
                        ],
                    }
                ]
            ],
            add_generation_prompt=False,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            videos_kwargs={
                "do_sample_frames": False,
                "do_resize": False,
                "video_metadata": [
                    {
                        "fps": self.MOCK_FPS,
                        "frames_indices": (timestamps * self.MOCK_FPS).tolist(),
                        "total_num_frames": len(frames),
                    }
                ],
            },
        ).to(device)

        input_ids = inputs["input_ids"][0]
        ve_id = self._processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        prefix_lens = (input_ids == ve_id).nonzero(as_tuple=True)[0].cpu().numpy() + 1

        cache = DynamicCache()
        self.model(**inputs, past_key_values=cache, use_cache=True)
        cache.crop(prefix_lens[-1])
        return cache, prefix_lens

    def _denoise_step(
        self,
        cache: DynamicCache,
        slot_labels: np.ndarray,
    ) -> np.ndarray:
        device = next(self.parameters()).device
        total_slots = len(slot_labels)

        reps = 1 if getattr(self.config, "single_sequence", False) else 2
        window_ids = np.tile(self._slot_token_ids[slot_labels], reps)
        window_ids = torch.tensor(window_ids, device=device).unsqueeze(0)

        hidden_states = self.model(
            input_ids=window_ids, past_key_values=cache, use_cache=False
        )[0]

        act_hidden = hidden_states[0, -total_slots:]
        logits = self.lm_head(act_hidden)[
            :, [self._inactive_token_id, self._active_token_id]
        ]
        return torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()

    @staticmethod
    def _unmask(slot_labels, probs, k):
        masked_pos = np.where(slot_labels == 2)[0]
        assert 0 < k <= len(masked_pos)
        masked_probs = probs[masked_pos]
        confidences = np.maximum(masked_probs, 1 - masked_probs)
        c = min(2 * k, len(masked_pos))
        candidates = np.argpartition(confidences, -c)[-c:]
        selected = np.random.choice(candidates, k, replace=False)
        slot_labels[masked_pos[selected]] = (masked_probs[selected] > 0.5).astype(int)

    @torch.no_grad()
    def detect(
        self,
        frames: np.ndarray,
        timestamps: np.ndarray,
        trigger_history: Optional[List[Dict[str, float]]] = None,
    ) -> List[Dict[str, float]]:
        frames = self.preprocess_frames(frames)

        args = self.trigger_args
        FTP = self._frame_time_patch
        total = len(frames)
        chunk_size = args.chunk_size or (self.config.trigger_window_past // 2)

        stack_start = 0
        cursor = 0
        prev_mega_start = -1
        mega_cache: Optional[DynamicCache] = None
        mega_prefix_lens: Optional[np.ndarray] = None

        def _n_complete():
            return sum(1 for s in self._active_segments if s.is_complete)

        while cursor < total:
            cursor = min(cursor + chunk_size, total)

            # Window shift until within budget
            drop = ((args.stride + FTP - 1) // FTP) * FTP
            while cursor - stack_start > args.max_window_size:
                stack_start += drop

            # FTP-align frame count
            n = cursor - stack_start
            n -= n % FTP
            if n < FTP:
                continue

            # ── Mega-window cache (rebuild when stack_start changes) ──
            if stack_start != prev_mega_start:
                mega_end = min(stack_start + args.max_window_size, total)
                mega_n = mega_end - stack_start
                mega_n -= mega_n % FTP
                mega_frames = frames[stack_start : stack_start + mega_n]
                mega_ts = timestamps[stack_start : stack_start + mega_n]
                mega_norm_ts = mega_ts - float(mega_ts[0])

                mega_cache, mega_prefix_lens = self._build_prefix_cache(
                    mega_frames, mega_norm_ts
                )
                prev_mega_start = stack_start

            # ── Slice cache for current chunk ──
            n_groups = n // FTP
            crop_len = int(mega_prefix_lens[n_groups - 1])

            chunk_ts = timestamps[stack_start : stack_start + n]
            time_offset = float(chunk_ts[0])
            last_frame_time = float(chunk_ts[-1]) - time_offset

            # ── Diffusion with window shifting (re-run on new completions) ──
            last_abs_time = float(chunk_ts[-1])
            for _rediffuse in range(args.max_triggers + 1):
                chunk_cache_iter = copy.deepcopy(mega_cache)
                chunk_cache_iter.crop(crop_len)

                if _rediffuse != 0:
                    self._slot_history.clear()

                slot_labels, slot_starts_abs, slot_ends_abs, unseen = (
                    self._diffuse_shifted(
                        chunk_cache_iter,
                        crop_len,
                        last_frame_time,
                        time_offset,
                    )
                )

                spans = self._extract_spans(
                    slot_labels, slot_starts_abs, slot_ends_abs
                )
                seen_mask = ~unseen
                first_seen_start = (
                    float(slot_starts_abs[seen_mask][0]) if np.any(seen_mask) else 0.0
                )
                newly_completed = self._update_segments(
                    spans,
                    first_seen_start,
                    last_abs_time,
                )

                if not newly_completed:
                    break
                if not self._is_multi_event or _n_complete() >= args.max_triggers:
                    break

            if not self._is_multi_event and _n_complete() > 0:
                break
            if _n_complete() >= args.max_triggers:
                break

            # End-of-video: finalize last incomplete segment
            if cursor >= total and self._active_segments:
                if not self._active_segments[-1].is_complete:
                    self._active_segments[-1].is_complete = True

        # ── Collect results from _active_segments ──
        segments = [
            {
                "start_sec": max(0.0, seg.start_sec),
                "end_sec": max(0.0, seg.end_sec),
            }
            for seg in self._active_segments
            if seg.is_complete
        ]
        if not segments:
            segments.append(
                {
                    "start_sec": 0.0,
                    "end_sec": float(timestamps[-1]),
                    "fallback": True,
                }
            )

        return segments

    def _diffuse_shifted(
        self,
        cache: DynamicCache,
        crop_len: int,
        last_frame_time: float,
        time_offset: float,
    ):
        args = self.trigger_args
        window_past = self.config.trigger_window_past
        trigger_res = self.config.trigger_temporal_resolution
        total_slots = window_past + self.config.trigger_window_future

        # Slot grid (normalized time for denoise, absolute for history lookup)
        slot_starts_norm = (
            last_frame_time
            - window_past * trigger_res
            + np.arange(total_slots) * trigger_res
        )
        slot_ends_norm = slot_starts_norm + trigger_res
        slot_starts_abs = slot_starts_norm + time_offset
        slot_ends_abs = slot_ends_norm + time_offset

        slot_labels = np.full(total_slots, 2, dtype=np.int64)
        unseen = slot_ends_norm + 0.1 < 0

        # ── Step 3: Inactive fixation for completed segments ──
        inactive_time = max(
            (seg.end_sec for seg in self._active_segments if seg.is_complete),
            default=float("-inf"),
        )
        slot_labels[slot_ends_abs <= inactive_time] = 0
        slot_labels[unseen] = 0

        # ── Step 1–2: Baseline confidence + retain previous predictions ──
        _do_retain = self._slot_history and args.confidence_threshold is not None
        baseline_probs = None
        if _do_retain:
            baseline_cache = copy.deepcopy(cache)
            baseline_cache.crop(crop_len)
            baseline_probs = self._denoise_step(baseline_cache, slot_labels)
            del baseline_cache
            for i in range(total_slots):
                if unseen[i]:
                    continue
                # [ slot_starts_abs[i], slot_ends_abs[i] )
                prevs = [
                    v
                    for t, v in self._slot_history.items()
                    if slot_starts_abs[i] < t + trigger_res and t < slot_ends_abs[i]
                ]
                assert 0 <= len(prevs) <= 2
                if len(set(prevs)) == 1:  # if the case is consensus
                    if (
                        abs(baseline_probs[i] - prevs[0])
                        < 1 - args.confidence_threshold
                    ):
                        slot_labels[i] = prevs[0]

        # ── Step 4: Diffusion on remaining masked slots ──
        seen = ~unseen
        n_masked = int(np.count_nonzero((slot_labels == 2) & seen))
        n_seen = int(np.count_nonzero(~unseen))

        if n_masked > 0:
            # Schedule based on n_seen (matches training), clamp to remaining masked
            steps = args.unmasking_steps
            base, rem = divmod(n_seen, steps)
            schedule = [base + (1 if i < rem else 0) for i in range(steps)]
            for step_k in schedule:
                n_remaining = int(np.count_nonzero(slot_labels[-n_seen:] == 2))
                step_k = min(step_k, n_remaining)
                if step_k == 0:
                    continue
                cache.crop(crop_len)
                probs = self._denoise_step(cache, slot_labels)
                self._unmask(
                    slot_labels[-n_seen:],
                    probs[-n_seen:],
                    step_k,
                )

        slot_labels[unseen] = 0

        # ── Step 5: Save to history ──
        self._slot_history = {
            float(slot_starts_abs[i]): int(slot_labels[i])
            for i in range(total_slots)
            if not unseen[i]
        }

        return slot_labels, slot_starts_abs, slot_ends_abs, unseen

    @staticmethod
    def _extract_spans(
        slot_labels: np.ndarray,
        slot_starts: np.ndarray,
        slot_ends: np.ndarray,
    ) -> List[TrackedSegment]:
        active_idx = np.where(slot_labels == 1)[0]
        if len(active_idx) == 0:
            return []
        return [
            TrackedSegment(
                start_sec=float(slot_starts[active_idx[0]]),
                end_sec=float(slot_ends[active_idx[-1]]),
            )
        ]

    def _update_segments(
        self,
        spans: List[TrackedSegment],
        first_seen_start: float,
        last_abs_time: float,
    ) -> List[TrackedSegment]:
        """Update tracked segments with current spans.
        Returns list of newly completed segments (triggers)."""
        args = self.trigger_args
        margin_start = last_abs_time - args.margin_seconds

        newly_completed: List[TrackedSegment] = []

        # ── Step 1: Resolve previous incomplete segment ──
        if self._active_segments and not self._active_segments[-1].is_complete:
            prev_seg = self._active_segments[-1]
            if spans and spans[0].start_sec <= prev_seg.end_sec:
                # Overlap with prev_seg → merge (continuation)
                spans[0].start_sec = min(spans[0].start_sec, prev_seg.start_sec)
                self._active_segments.pop(-1)
            elif prev_seg.end_sec >= first_seen_start:
                # prev_seg is within the seen window but no overlapping span
                # → false positive, discard
                self._active_segments.pop(-1)
            else:
                # prev_seg is outside the current window → accept as complete
                prev_seg.is_complete = True
                newly_completed.append(prev_seg)

        # ── Step 2: Register remaining spans ──
        for span in spans:
            span.is_complete = span.end_sec < margin_start
            self._active_segments.append(span)
            if span.is_complete:
                newly_completed.append(span)

        return newly_completed
