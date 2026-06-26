#
# Copyright (C) 2025 InterLive Team. All Rights Reserved.
#
from .base import BaseModel, BaseTriggerModel

MODEL_REGISTRY: dict[str, type[BaseModel]] = {}


def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model_class(name):
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model type '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name]


from .qwen3_stride import STRIDEQwen3VL  # noqa: E402

__all__ = [
    "BaseModel",
    "BaseTriggerModel",
    "MODEL_REGISTRY",
    "register_model",
    "get_model_class",
    "STRIDEQwen3VL",
]
