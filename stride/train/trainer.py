#
# Copyright (C) 2025 InterLive Team. All Rights Reserved.
#
import torch.nn.functional as F

from stride.utils.train_utils import BaseTrainer


class STRIDETrainer(BaseTrainer):
    """Custom trainer for the STRIDE activation model."""

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        activation_labels = inputs.pop("activation_labels")
        outputs = model(**inputs)
        logits = outputs.logits  # [N_act, 2]

        valid_mask = activation_labels != -100
        valid_logits = logits[valid_mask]
        valid_labels = activation_labels[valid_mask]

        if valid_labels.numel() == 0:
            loss = logits.sum() * 0.0
        else:
            loss = F.cross_entropy(valid_logits, valid_labels)

        return (loss, outputs) if return_outputs else loss
