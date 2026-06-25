#
# Copyright (C) 2025 InterLive Team. All Rights Reserved.
#

import pathlib
import warnings
from functools import partial

import torch
import transformers
from transformers import AutoProcessor
from transformers.trainer_utils import set_seed

from stride.model import Qwen3VLForSTRIDE
from stride.train.arguments import DataArguments, ModelArguments, TrainingArguments
from stride.train.trainer import STRIDETrainer
from stride.utils.train_utils import (
    BaseDataCollator,
    BaseDataset,
    configure_model_for_training,
    rank0_print,
)

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")

SPECIAL_TOKENS = ["<activation>", "<active>", "<inactive>"]


def make_supervised_data_module(processor, data_args, model_type, model):
    """Make dataset and collator for supervised fine-tuning."""
    assert model is not None
    spec_processor = model.preprocess_input_spec
    apply_chat_template = partial(
        model.apply_chat_template, data_args=data_args, model_config=model.config
    )

    train_dataset = BaseDataset(
        data_path=data_args.data_path,
        processor=processor,
        data_args=data_args,
        model_type=model_type,
        spec_processor=spec_processor,
        model_config=model.config,
    )
    data_collator = BaseDataCollator(
        processor=processor,
        model_type=model_type,
        apply_chat_template=apply_chat_template,
    )
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


def get_model(model_args, training_args):
    """Load model based on model type."""
    rank0_print(f"Loading model: {model_args.model_name_or_path}")
    rank0_print(f"Model type: {model_args.model_type}")

    model_kwargs = {
        "cache_dir": training_args.cache_dir,
        "dtype": torch.bfloat16 if training_args.bf16 else torch.float16,
        "attn_implementation": training_args.attn_implementation,
    }
    processor_kwargs = {
        "videos_kwargs": {"do_resize": False},
        "do_sample_frames": False,
        "model_max_length": training_args.model_max_length,
        "pad_to_multiple_of": 256,
        "padding": True,
        "padding_side": "left",
    }

    if model_args.model_type == "qwen3_vl":
        model = Qwen3VLForSTRIDE.from_pretrained(
            model_args.model_name_or_path, **model_kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_args.model_type}")

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, **processor_kwargs
    )

    # Add special tokens and resize embeddings
    num_added = processor.tokenizer.add_tokens(SPECIAL_TOKENS, special_tokens=True)
    rank0_print(f"Added {num_added} special tokens: {SPECIAL_TOKENS}")
    model.resize_token_embeddings(len(processor.tokenizer))

    # Store token IDs on model for use in forward()
    model._activation_token_id = processor.tokenizer.convert_tokens_to_ids(
        "<activation>"
    )
    model._active_token_id = processor.tokenizer.convert_tokens_to_ids("<active>")
    model._inactive_token_id = processor.tokenizer.convert_tokens_to_ids("<inactive>")
    rank0_print(
        f"Token IDs: activation={model._activation_token_id}, "
        f"active={model._active_token_id}, inactive={model._inactive_token_id}"
    )

    return model, processor


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    rank0_print("=" * 50)
    rank0_print("STRIDE Activation Model Training")
    rank0_print("=" * 50)
    rank0_print(f"Model: {model_args.model_name_or_path}")
    rank0_print(f"Model type: {model_args.model_type}")
    rank0_print(f"Data: {data_args.data_path}")
    rank0_print("=" * 50)

    # Load model
    model, processor = get_model(model_args, training_args)

    # Store trigger config in model config (persisted on save)
    model.config.trigger_window_past = model_args.trigger_window_past
    model.config.trigger_temporal_resolution = model_args.trigger_temporal_resolution
    model.config.mask_modes = model_args.mask_modes
    model.config.single_sequence = model_args.single_sequence
    rank0_print(
        f"Trigger window: past={model_args.trigger_window_past} "
        f"x {model_args.trigger_temporal_resolution}s"
    )
    rank0_print(f"Mask modes: {model_args.mask_modes}")
    rank0_print(f"Single sequence: {model_args.single_sequence}")

    # Configure for training
    model = configure_model_for_training(model, model_args, training_args)

    # Re-seed per rank for diverse masking noise
    set_seed(training_args.seed + training_args.process_index)

    # Create data module
    data_module = make_supervised_data_module(
        processor=processor,
        data_args=data_args,
        model_type=model_args.model_type,
        model=model,
    )

    # Create trainer
    trainer = STRIDETrainer(
        model=model, args=training_args, processing_class=processor, **data_module
    )
    # trainer.add_callback(GlobalMetricsCallback(trainer))

    # Train
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    # Save model
    trainer.save_model(training_args.output_dir)
    rank0_print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()
