#
# Copyright (C) 2025 InterLive Team. All Rights Reserved.
#
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np


class BaseModel(ABC):
    """Root abstraction for eval models.

    Provides resolution computation and frame preprocessing shared
    by the trigger detection pipeline.
    """

    @classmethod
    @abstractmethod
    def from_pretrained_with_processor(
        cls,
        model_path: str,
        device: str = "cuda",
        **kwargs,
    ) -> "BaseModel":
        """Load model + processor from a pretrained checkpoint."""
        ...

    @abstractmethod
    def compute_dst_resolution(
        self, src_resolution: Tuple[int, int]
    ) -> Tuple[int, int]:
        """Compute target resolution for a given source resolution."""
        ...

    @abstractmethod
    def preprocess_frames(self, frames: np.ndarray) -> np.ndarray:
        """Resize frames to the model's target resolution.

        Args:
            frames: (N, H, W, 3) uint8.
        Returns:
            (N, H', W', 3) uint8.
        """
        ...


class BaseTriggerModel(BaseModel):
    """Abstract interface for streaming trigger detection models.

    Subclasses encapsulate all model-specific logic: preprocessing,
    inference, sliding window, thresholding, and fallback.

    Usage:
        model = SomeModel.from_pretrained_with_processor(path)
        model.configure(threshold=0.3)
        model.start_stream(query="What happened?", is_multi_event=True)
        segments = model.detect(frames, timestamps)
    """

    @abstractmethod
    def start_stream(
        self,
        query: str,
        is_multi_event: bool = False,
    ) -> None:
        """Signal the start of a new video stream for a query.

        Resets internal state: fallback tracker, running best probability,
        multi-event flag. Must be called before the first detect() call
        for each (video, question) pair.

        Args:
            query: The natural language question/prompt.
            is_multi_event: If True, detect multiple trigger segments.
        """
        ...

    @abstractmethod
    def detect(
        self,
        frames: np.ndarray,
        timestamps: np.ndarray,
        trigger_history: Optional[List[Dict[str, float]]] = None,
    ) -> List[Dict[str, float]]:
        """Run trigger detection on video frames.

        The model handles video preprocessing (resolution, resizing)
        internally. The caller provides raw decoded frames.

        Args:
            frames: Raw video frames, shape (N, H, W, 3), dtype uint8.
            timestamps: Corresponding timestamps in seconds, shape (N,).
            trigger_history: Previously detected segments from prior
                detect() calls in this stream.

        Returns:
            List of detected trigger segments:
            [{"start_sec": float, "end_sec": float}, ...]
        """
        ...
