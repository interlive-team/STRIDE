#
# Copyright (C) 2025 InterLive Team. All Rights Reserved.
#
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image


@dataclass
class VideoSpec:
    path: str
    fps: float
    start_seconds: float
    end_seconds: float
    src_resolution: Tuple[int, int]
    dst_resolution: Optional[Tuple[int, int]] = None
    num_frames: Optional[int] = None
    num_tokens: Optional[int] = None
    content: Optional[np.ndarray] = field(default=None, repr=False)
    content_time: Optional[np.ndarray] = None


@dataclass
class ImageSpec:
    path: str
    src_resolution: Tuple[int, int]
    dst_resolution: Optional[Tuple[int, int]] = None
    num_tokens: Optional[int] = None
    content: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class TextSpec:
    content: str
    output: bool
    num_tokens: Optional[int] = None


@dataclass
class ActivationEntry:
    start_seconds: float
    end_seconds: float
    mask: bool
    value: str


@dataclass
class ActivationSpec:
    entries: List[ActivationEntry]
    num_tokens: int = 0


InputSpec = Union[VideoSpec, ImageSpec, TextSpec, ActivationSpec]


def resize_image(image: np.ndarray, dst_resolution: Tuple[int, int]) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    dst_h, dst_w = dst_resolution

    if (src_w, src_h) == (dst_w, dst_h):
        return image

    if dst_w <= src_w and dst_h <= src_h:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR

    return cv2.resize(image, (dst_w, dst_h), interpolation=interpolation)


def prepare_spec(data: List[Dict[str, Any]]) -> List[InputSpec]:
    specs = []
    for item in data:
        if item["type"] == "video":
            specs.append(
                VideoSpec(
                    path=item["path"],
                    fps=item["fps"],
                    start_seconds=item["start_seconds"],
                    end_seconds=item["end_seconds"],
                    src_resolution=tuple(item["src_resolution"]),
                )
            )
        elif item["type"] == "image":
            specs.append(
                ImageSpec(
                    path=item["path"],
                    src_resolution=tuple(item["src_resolution"]),
                )
            )
        elif item["type"] == "text":
            specs.append(
                TextSpec(
                    content=item["content"],
                    output=item.get("output", False),
                )
            )
        elif item["type"] == "activation_output":
            entries = [
                ActivationEntry(
                    start_seconds=e["start_seconds"],
                    end_seconds=e["end_seconds"],
                    mask=e["mask"],
                    value=e["value"],
                )
                for e in item["entries"]
            ]
            specs.append(ActivationSpec(entries=entries))
    return specs


def load_content(spec: InputSpec):
    """Loads content from path into the spec's content field."""
    if isinstance(spec, ImageSpec):
        assert spec.path is not None
        assert spec.content is None
        image = np.array(Image.open(spec.path).convert("RGB"))
        assert tuple(image.shape[0:2]) == tuple(spec.src_resolution)
        spec.content = resize_image(image, spec.dst_resolution)
    elif isinstance(spec, VideoSpec):
        assert spec.path is not None
        assert spec.content is None
        assert isinstance(spec.num_frames, int)
        assert 0 <= spec.num_frames
        assert 0 <= spec.start_seconds <= spec.end_seconds
        vr = VideoReader(spec.path, ctx=cpu(0))
        fps = vr.get_avg_fps()
        start_frame = math.ceil(spec.start_seconds * fps)
        end_frame = min(math.ceil(spec.end_seconds * fps), len(vr))
        assert start_frame < end_frame
        indices = np.linspace(start_frame, end_frame - 1, spec.num_frames, dtype=int)
        spec.content_time = indices / fps
        frames = vr.get_batch(indices).asnumpy()
        assert tuple(frames.shape[1:3]) == tuple(spec.src_resolution)
        spec.content = np.stack([resize_image(f, spec.dst_resolution) for f in frames])
    return spec


def time_synchronize(specs: List[InputSpec]) -> List[InputSpec]:
    time_offset = 0.0
    last_ct0_abs = None  # content_time[0] before sync (absolute)
    last_video_offset = None

    for spec in specs:
        if isinstance(spec, VideoSpec):
            last_ct0_abs = float(spec.content_time[0])
            last_video_offset = time_offset
            spec.content_time = spec.content_time - last_ct0_abs + time_offset
            time_offset = float(spec.content_time[-1]) + 1.0 / spec.fps
        elif isinstance(spec, ActivationSpec):
            assert last_ct0_abs is not None, "ActivationSpec must follow a VideoSpec"
            for entry in spec.entries:
                entry.start_seconds = (
                    entry.start_seconds - last_ct0_abs + last_video_offset
                )
                entry.end_seconds = entry.end_seconds - last_ct0_abs + last_video_offset
    return specs


def distribute_frames(
    specs: List[VideoSpec],
    *,
    min_frames_per_clip: int,
    frame_multiple: int,
    max_total_frames: int,
    max_fps: float,
) -> List[int]:
    assert min_frames_per_clip % frame_multiple == 0
    low, high = 0.0, max_fps
    for _ in range(30):
        fps = (low + high) / 2
        frames = [
            max(
                min_frames_per_clip,
                round((spec.end_seconds - spec.start_seconds) * fps / frame_multiple)
                * frame_multiple,
            )
            for spec in specs
        ]
        if sum(frames) <= max_total_frames:
            low = fps
            result = frames
        else:
            high = fps
    return result


def resolve_resolution(
    spec: VideoSpec,
    *,
    frame_time_patch: int,
    frame_spatial_patch: int,
    frame_max_tokens: int,
) -> List[int]:
    ratio = spec.src_resolution[1] / spec.src_resolution[0]
    resolution = [int(r / frame_spatial_patch) for r in spec.src_resolution]
    best_resolution = None
    best_tokens = float("-inf")
    for h in range(1, resolution[0] + 1):
        for w in [int(h * ratio), int(h * ratio) + 1]:
            n_tokens = h * w
            if 0 < w <= resolution[1] and best_tokens < n_tokens <= frame_max_tokens:
                best_resolution = (h, w)
                best_tokens = n_tokens
    dst_resolution = [r * frame_spatial_patch for r in best_resolution]
    num_tokens = best_tokens * spec.num_frames // frame_time_patch
    return dst_resolution, num_tokens


def extract_bounded_spans(
    input_ids: torch.Tensor,
    start_seq: Tuple[int, ...],
    end_seq: Tuple[int, ...],
    start_offset: int,
    end_offset: int,
    fill_value: int = -100,
) -> torch.Tensor:
    assert input_ids.ndim == 2
    assert max(len(start_seq), len(end_seq)) <= input_ids.size(1)
    s_seq = torch.tensor(start_seq, device=input_ids.device)
    e_seq = torch.tensor(end_seq, device=input_ids.device)
    labels = torch.full_like(input_ids, fill_value)
    for input_id, label, smask, emask in zip(
        input_ids,
        labels,
        (input_ids.unfold(1, len(s_seq), 1) == s_seq.view(1, 1, -1)).all(dim=-1),
        (input_ids.unfold(1, len(e_seq), 1) == e_seq.view(1, 1, -1)).all(dim=-1),
    ):
        s_indices = smask.nonzero(as_tuple=True)[0].tolist()
        e_indices = emask.nonzero(as_tuple=True)[0].tolist()
        assert s_indices
        assert e_indices
        e_iterate = iter(e_indices)
        e_idx = next(e_iterate)
        for s_idx in s_indices:
            while e_idx < s_idx:
                e_idx = next(e_iterate)
            rng = slice(s_idx + start_offset, e_idx + end_offset)
            label[rng] = input_id[rng]
    return labels
