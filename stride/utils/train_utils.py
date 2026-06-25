#
# Copyright (C) 2025 InterLive Team. All Rights Reserved.
#
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.distributed as dist
import tqdm
import yaml
from torch.utils.data import Dataset
from transformers import Trainer
from transformers.modeling_utils import unwrap_model
from transformers.trainer import is_sagemaker_mp_enabled

from stride.utils.data_processing import LengthGroupedSampler
from stride.utils.input_processing import (
    InputSpec,
    load_content,
    prepare_spec,
    time_synchronize,
)


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args)
    else:
        print(*args)


class BaseDataset(Dataset):
    """Dataset for supervised fine-tuning supporting multiple model types."""

    def __init__(
        self,
        data_path: str,
        processor,
        data_args,
        model_type: str = "qwen3_vl",
        spec_processor=None,
        model_config=None,
    ):
        super().__init__()
        self.processor = processor
        self.data_args = data_args
        self.model_type = model_type
        self.spec_processor = spec_processor
        self.model_config = model_config
        self.list_data_dict = []

        # Load data from JSON, JSONL, or YAML
        if data_path.endswith(".yaml"):
            with open(data_path, "r") as f:
                yaml_data = yaml.safe_load(f)
                datasets = yaml_data.get("datasets", [])
                for dataset in datasets:
                    json_path = dataset.get("json_path")
                    self._load_json(json_path)
        elif data_path.endswith((".json", ".jsonl")):
            self._load_json(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}.")

        rank0_print(f"Loaded {len(self.list_data_dict)} samples from {data_path}")

    def _load_json(self, json_path):
        if json_path.endswith(".jsonl"):
            with open(json_path, "r") as f:
                # Check first line for metadata header
                first = json.loads(next(f))
                if isinstance(first, dict) and "_metadata" in first:
                    rank0_print(f"Data generation options ({json_path}):")
                    for k, v in first["_metadata"].items():
                        rank0_print(f"  {k}: {v}")
                else:
                    self.list_data_dict.append(first)
                for line in f:
                    if line.strip():
                        self.list_data_dict.append(json.loads(line))
        else:
            with open(json_path, "r") as f:
                self.list_data_dict.extend(json.load(f))

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for item in tqdm.tqdm(
            self.list_data_dict,
            desc="Data Sampler Preprocessing",
            disable=dist.is_initialized() and dist.get_rank() != 0,
            ncols=70,
            mininterval=1,
        ):
            specs = prepare_spec(item)
            specs = self.spec_processor(
                specs,
                processor=self.processor,
                data_args=self.data_args,
                model_config=self.model_config,
            )
            length_list.append(sum(s.num_tokens for s in specs))
        return length_list

    def __getitem__(self, idx):
        import random

        for attempt in range(3):
            try:
                specs = prepare_spec(self.list_data_dict[idx])
                specs = self.spec_processor(
                    specs,
                    processor=self.processor,
                    data_args=self.data_args,
                    model_config=self.model_config,
                )
                specs = [load_content(s) for s in specs]
                specs = time_synchronize(specs)
                return specs
            except Exception as e:
                rank0_print(
                    f"[WARNING] Failed to load sample {idx} (attempt {attempt+1}): {e}"
                )
                idx = random.randint(0, len(self.list_data_dict) - 1)
        raise RuntimeError(f"Failed to load any sample after 3 attempts")


@dataclass
class BaseDataCollator:
    """Collate examples for supervised fine-tuning."""

    processor: Any
    model_type: str = "qwen3_vl"
    apply_chat_template: Optional[Any] = None

    def __call__(self, instances: Sequence[List[InputSpec]]) -> Dict[str, torch.Tensor]:
        try:
            batch = self.apply_chat_template(instances, processor=self.processor)
            return dict(batch)
        except Exception as e:
            summaries = []
            for i, specs in enumerate(instances):
                summaries.append(f"  [{i}] {specs!r}")
            raise type(e)(
                f"[Collator] {e}\nBatch specs:\n" + "\n".join(summaries)
            ) from e


def configure_model_for_training(model, model_args, training_args):
    """Configure which parts of the model to train."""
    model.config.use_cache = False
    model.requires_grad_(False)

    if getattr(model_args, "tune_embed", False):
        for p in model.embedding_parameters:
            p.requires_grad = True
        rank0_print("Trainable: embedding (embed_tokens + lm_head)")

    if getattr(model_args, "tune_lang", False):
        for p in model.language_parameters:
            p.requires_grad = True
        rank0_print("Trainable: language model")

    if getattr(model_args, "tune_proj", False):
        for p in model.projection_parameters:
            p.requires_grad = True
        rank0_print("Trainable: visual.merger (projector)")

    if getattr(model_args, "tune_vis", False):
        for p in model.vision_parameters:
            p.requires_grad = True
        rank0_print("Trainable: visual (full vision tower)")

    # Print trainable params
    total_params = 0
    trainable_params = 0
    for p in model.parameters():
        p_numel = getattr(p, "ds_numel", p.numel())
        total_params += p_numel
        if p.requires_grad:
            trainable_params += p_numel

    rank0_print(f"Total parameters: {total_params:,}")
    rank0_print(
        f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
    )

    return model


class BaseTrainer(Trainer):
    """Base trainer with common functionality shared across training tasks."""

    def _get_train_sampler(self, train_dataset=None):
        if train_dataset is None:
            train_dataset = self.train_dataset
        generator = torch.Generator()
        generator.manual_seed(
            self.args.data_seed if self.args.data_seed is not None else self.args.seed
        )
        return LengthGroupedSampler(
            global_batch_size=self.args.train_batch_size * self.args.world_size,
            lengths=train_dataset.lengths,
            generator=generator,
        )

    def get_decay_parameter_names(self, model) -> list[str]:
        decay_parameters = super().get_decay_parameter_names(model)
        embed_params = set(getattr(unwrap_model(model), "embedding_parameters", []))
        embed_names = {n for n, p in model.named_parameters() if p in embed_params}
        return [n for n in decay_parameters if n not in embed_names]

    def create_optimizer(self):
        """Create optimizer with different learning rates for different components."""
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)

            unwrapped_model = unwrap_model(opt_model)

            # Define parameter sets for each group
            param_sets = {
                "embedding": set(getattr(unwrapped_model, "embedding_parameters", [])),
                "language": set(getattr(unwrapped_model, "language_parameters", [])),
                "projector": set(getattr(unwrapped_model, "projection_parameters", [])),
                "vision": set(getattr(unwrapped_model, "vision_parameters", [])),
            }

            # Define learning rates
            lr_lang = (
                self.args.lr_lang
                if self.args.lr_lang is not None
                else self.args.learning_rate
            )
            lr_proj = (
                self.args.lr_proj
                if self.args.lr_proj is not None
                else self.args.learning_rate
            )
            lr_vis = (
                self.args.lr_vis
                if self.args.lr_vis is not None
                else self.args.learning_rate
            )
            lr_embed = self.args.lr_embed if self.args.lr_embed is not None else lr_lang

            # Initialize groups
            groups = {
                "embedding": {"lr": lr_embed, "decay": [], "no_decay": [], "count": 0},
                "language": {"lr": lr_lang, "decay": [], "no_decay": [], "count": 0},
                "projector": {"lr": lr_proj, "decay": [], "no_decay": [], "count": 0},
                "vision": {"lr": lr_vis, "decay": [], "no_decay": [], "count": 0},
                "other": {"lr": lr_lang, "decay": [], "no_decay": [], "count": 0},
            }

            # Assign parameters to groups
            for name, param in opt_model.named_parameters():
                keys = [n for n, p in param_sets.items() if param in p] or ["other"]
                assert (
                    len(keys) == 1
                ), f"Parameter {name} overlaps or is unassigned: {keys}"
                key = keys[0]

                groups[key]["count"] += getattr(param, "ds_numel", param.numel())

                if param.requires_grad:
                    if name in decay_parameters:
                        groups[key]["decay"].append(param)
                    else:
                        groups[key]["no_decay"].append(param)

            # Print statistics
            if self.is_world_process_zero():
                print(f"{'=' * 20} Optimizer Groups {'=' * 20}")
                for key, info in groups.items():
                    if info["count"] > 0:
                        trainable_decay = sum(
                            getattr(p, "ds_numel", p.numel()) for p in info["decay"]
                        )
                        trainable_no_decay = sum(
                            getattr(p, "ds_numel", p.numel()) for p in info["no_decay"]
                        )
                        trainable_total = trainable_decay + trainable_no_decay
                        print(
                            f"Group '{key}': {trainable_total:,} trainable / {info['count']:,} total params. LR: {info['lr']}"
                        )
                        print(
                            f"Group '{key}' with weight decay: {trainable_decay:,} trainable params. LR: {info['lr']}"
                        )
                        print(
                            f"Group '{key}' without weight decay: {trainable_no_decay:,} trainable params. LR: {info['lr']}"
                        )
                print(f"{'=' * 58}")

            # Create optimizer groups
            optimizer_grouped_parameters = []
            for key, info in groups.items():
                if info["decay"]:
                    optimizer_grouped_parameters.append(
                        {
                            "params": info["decay"],
                            "weight_decay": self.args.weight_decay,
                            "lr": info["lr"],
                        }
                    )
                if info["no_decay"]:
                    optimizer_grouped_parameters.append(
                        {
                            "params": info["no_decay"],
                            "weight_decay": 0.0,
                            "lr": info["lr"],
                        }
                    )

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )
            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )

        return self.optimizer
