#
# Copyright (C) 2025 InterLive Team. All Rights Reserved.
#
from typing import List, Optional, Tuple

import cv2
import numpy as np
from decord import VideoReader, cpu


class VideoLoader:
    """Minimal video frame loader.

    Decodes the requested frame indices from a video and optionally resizes
    them. Decoding is single-threaded via decord (exact indexed access).
    """

    def run(
        self,
        video_path: str,
        frame_indices: List[int],
        target_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Decode ``frame_indices`` from ``video_path`` and optionally resize.

        :param video_path: path to the video file
        :param frame_indices: frame indices to extract (order/duplicates preserved)
        :param target_size: (height, width) to resize to, or None for original
        :returns: array of shape (len(frame_indices), H, W, C), dtype uint8
        """
        uniq = sorted(set(frame_indices))
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        batch = vr.get_batch(uniq).asnumpy()

        pos = {f: i for i, f in enumerate(uniq)}
        frames = batch[[pos[f] for f in frame_indices]]

        if target_size is not None:
            th, tw = target_size
            if (frames.shape[1], frames.shape[2]) != (th, tw):
                frames = np.stack(
                    [cv2.resize(f, (tw, th), interpolation=cv2.INTER_AREA) for f in frames]
                )
        return frames
