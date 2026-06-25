#
# Copyright (C) 2025 InterLive Team. All Rights Reserved.
#
import random
from typing import List

import numpy as np
import torch
from transformers import Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast

from stride.utils.input_processing import (
    ActivationSpec,
    ImageSpec,
    InputSpec,
    TextSpec,
    VideoSpec,
    distribute_frames,
    extract_bounded_spans,
    resolve_resolution,
)


class Qwen3VLForStreamQA(Qwen3VLForConditionalGeneration):
    accepts_loss_kwargs = False
    _output_class = Qwen3VLCausalLMOutputWithPast
    _label_start_seq = (151644, 77091, 198)  # <|im_start|>assistant\n
    _label_end_seq = (151645, 198)  # <|im_end|>\n

    @property
    def vision_parameters(self):
        exclude = set(self.projection_parameters)
        for p in self.model.visual.parameters():
            if p not in exclude:
                yield p

    @property
    def projection_parameters(self):
        yield from self.model.visual.merger.parameters()

    @property
    def embedding_parameters(self):
        seen = set()
        for p in self.model.language_model.embed_tokens.parameters():
            seen.add(p)
            yield p
        for p in self.lm_head.parameters():
            if p not in seen:
                yield p

    @property
    def language_parameters(self):
        exclude = set(self.embedding_parameters)
        for p in self.model.language_model.parameters():
            if p not in exclude:
                yield p

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        cache_position=None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        if labels is not None:
            shift_labels = labels[..., 1:]
            valid_mask = shift_labels != -100
            valid_labels = shift_labels[valid_mask]

            valid_hidden_states = hidden_states[..., :-1, :][valid_mask]
            valid_logits = self.lm_head(valid_hidden_states)

            loss = torch.nn.functional.cross_entropy(valid_logits, valid_labels)
            logits = None
        else:
            loss = None
            logits = self.lm_head(hidden_states)

        return self._output_class(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            rope_deltas=outputs.rope_deltas,
        )

    @classmethod
    def preprocess_input_spec(
        cls, stream: List[InputSpec], processor, data_args, **kwargs
    ):
        videospecs = []
        for item in stream:
            if isinstance(item, VideoSpec):
                videospecs.append(item)
            elif isinstance(item, ImageSpec):
                raise NotImplementedError()
            elif isinstance(item, TextSpec):
                item.num_tokens = len(processor.tokenizer(item.content).input_ids)

        for videospec, n_frame in zip(
            videospecs,
            distribute_frames(
                videospecs,
                min_frames_per_clip=data_args.video_min_frames_per_clip,
                frame_multiple=data_args.video_frame_multiple,
                max_total_frames=data_args.video_max_total_frames,
                max_fps=data_args.video_max_fps,
            ),
        ):
            videospec.num_frames = n_frame
            resolution, num_tokens = resolve_resolution(
                videospec,
                frame_time_patch=data_args.frame_time_patch,
                frame_spatial_patch=data_args.frame_spatial_patch,
                frame_max_tokens=data_args.frame_max_tokens,
            )
            videospec.dst_resolution = resolution
            videospec.num_tokens = num_tokens
        return stream

    @classmethod
    def apply_chat_template(
        cls, batch_stream: List[List[InputSpec]], processor, **kwargs
    ):
        def add_content(messages, role, content):
            if len(messages) > 0 and messages[-1]["role"] == role:
                messages[-1]["content"].append(content)
            else:
                messages.append(dict(role=role, content=[content]))

        MOCK_FPS = 100
        video_metadatas = []
        messages = []
        for stream in batch_stream:
            messages.append([])
            for spec in stream:
                if isinstance(spec, TextSpec):
                    add_content(
                        messages[-1],
                        role=["user", "assistant"][spec.output],
                        content=dict(type="text", text=spec.content),
                    )
                elif isinstance(spec, VideoSpec):
                    assert spec.content_time is not None
                    add_content(
                        messages[-1],
                        role="user",
                        content=dict(type="video", video=spec.content),
                    )
                    indices = (spec.content_time * MOCK_FPS).tolist()
                    video_metadatas.append(
                        {
                            "fps": MOCK_FPS,
                            "frames_indices": indices,
                            "total_num_frames": len(indices),
                        }
                    )
                else:
                    raise ValueError(f"Unsupported spec type: {type(spec)}")

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            videos_kwargs={
                "do_sample_frames": False,
                "do_resize": False,
                "video_metadata": video_metadatas,
            },
        )
        inputs["labels"] = extract_bounded_spans(
            inputs["input_ids"],
            start_seq=cls._label_start_seq,
            end_seq=cls._label_end_seq,
            start_offset=len(cls._label_start_seq),
            end_offset=1,
            fill_value=-100,
        )
        return inputs


class Qwen3VLForSTRIDE(Qwen3VLForStreamQA):
    _activation_token_id: int = -1
    _active_token_id: int = -1
    _inactive_token_id: int = -1

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        cache_position=None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]

        act_mask = input_ids == self._activation_token_id
        act_hidden = hidden_states[act_mask]

        vocab_logits = self.lm_head(act_hidden)
        binary_logits = vocab_logits[
            :, [self._inactive_token_id, self._active_token_id]
        ]

        return self._output_class(
            loss=None,
            logits=binary_logits,
            past_key_values=outputs.past_key_values,
            rope_deltas=outputs.rope_deltas,
        )

    @classmethod
    def preprocess_input_spec(
        cls, stream: List[InputSpec], processor, data_args, **kwargs
    ):
        stream = Qwen3VLForStreamQA.preprocess_input_spec(
            stream, processor, data_args, **kwargs
        )
        model_config = kwargs.get("model_config")
        trigger_window_total = model_config.trigger_window_past
        single_seq = getattr(model_config, "single_sequence", False)
        multiplier = 1 if single_seq else 2
        for spec in stream:
            if isinstance(spec, ActivationSpec):
                spec.num_tokens = multiplier * trigger_window_total
        return stream

    @staticmethod
    def _generate_mask(gt_labels, mode=None, budget=None, allowed_modes=(1, 2, 3)):
        n = len(gt_labels)
        if mode is None:
            mode = random.choice(allowed_modes)

        if mode == 2 and n == 1 and np.any(gt_labels == 1):
            mode = 0

        match mode:
            case 0:
                k = random.randint(1, n)
                mask = np.zeros(n, dtype=bool)
                mask[np.random.choice(n, k, replace=False)] = True
                return mask, 0, k

            case 1:
                return np.ones(n, dtype=bool), 1, n

            case 2:
                if budget is not None:
                    assert 1 <= budget <= max(1, int(0.3 * n))
                    apply_budget = budget
                else:
                    apply_budget = random.randint(1, max(1, int(0.3 * n)))
                active_positions = np.where(gt_labels == 1)[0]
                if len(active_positions) > 0:
                    boundary = {
                        active_positions[0],
                        active_positions[-1],
                        active_positions[0] - 1,
                        active_positions[-1] + 1,
                    }
                    check_positions = np.array([i for i in boundary if 0 <= i < n])
                else:
                    check_positions = np.array([], dtype=int)

                while True:
                    mask = np.ones(n, dtype=bool)
                    remaining = apply_budget
                    spans = [range(0, n)]
                    while remaining > 0:
                        span_len = random.randint(1, remaining)
                        candidates = [s for s in spans if len(s) >= span_len]
                        if not candidates:
                            break
                        weights = [len(s) - span_len + 1 for s in candidates]
                        chosen = random.choices(candidates, weights=weights, k=1)[0]
                        start = random.randint(chosen.start, chosen.stop - span_len)
                        mask[start : start + span_len] = False
                        remaining -= span_len
                        spans.remove(chosen)
                        left = range(chosen.start, start - 1)
                        right = range(start + span_len + 1, chosen.stop)
                        if len(left) > 0:
                            spans.append(left)
                        if len(right) > 0:
                            spans.append(right)
                    if remaining:
                        continue
                    if len(check_positions) > 0 and not np.any(mask[check_positions]):
                        continue
                    return mask, 2, apply_budget

            case _:
                if budget is not None:
                    assert 1 <= budget <= n
                    num_mask = budget
                else:
                    num_mask = random.randint(1, n)
                mask = np.zeros(n, dtype=bool)

                trig = np.where(gt_labels == 1)[0]
                if num_mask > 1:
                    num_cont = random.randint(1, num_mask)

                    if len(trig) > 0:
                        t_start, t_end = int(trig[0]), int(trig[-1])
                        inside = random.randint(1, min(num_cont, t_end - t_start + 1))
                        outside = num_cont - inside
                        if random.random() < 0.5:
                            blk_start = t_start - outside
                            blk_end = t_start + inside
                        else:
                            blk_start = t_end - inside + 1
                            blk_end = t_end + 1 + outside
                    else:
                        blk_start = random.randint(0, n - 1)
                        blk_end = blk_start + num_cont

                    cs, ce = max(0, blk_start), min(n, blk_end)
                    mask[cs:ce] = True
                    overflow = num_cont - (ce - cs)

                    num_scat = num_mask - num_cont + overflow
                    (avail,) = np.where(~mask)
                    scat = np.random.choice(
                        avail, min(num_scat, len(avail)), replace=False
                    )
                    mask[scat] = True
                    return mask, 3, num_mask

                indices = np.random.choice(n, num_mask, replace=False)
                mask[indices] = True
                return mask, 3, num_mask

    @classmethod
    def apply_chat_template(
        cls,
        batch_stream: List[List[InputSpec]],
        processor,
        data_args=None,
        model_config=None,
        **kwargs,
    ):
        def add_content(messages, role, content):
            if len(messages) > 0 and messages[-1]["role"] == role:
                messages[-1]["content"].append(content)
            else:
                messages.append(dict(role=role, content=[content]))

        MOCK_FPS = 100
        window_past = model_config.trigger_window_past
        trigger_res = model_config.trigger_temporal_resolution
        trigger_window_total = window_past
        allowed_modes = model_config.mask_modes
        single_seq = getattr(model_config, "single_sequence", False)

        def _augment_unseen_mask(unseen_gt, seen_mode, seen_budget, n_seen):
            n = len(unseen_gt)
            match random.randint(1, 5):
                case 1:
                    return np.ones(n, dtype=bool)
                case 2:
                    return np.zeros(n, dtype=bool)
                case 3:
                    k = random.randint(1, n)
                    m = np.zeros(n, dtype=bool)
                    m[np.random.choice(n, k, replace=False)] = True
                    return m
                case 4:
                    ratio = seen_budget / n_seen
                    scaled = max(1, round(ratio * n))
                    match seen_mode:
                        case 2:
                            scaled = min(scaled, max(1, int(0.3 * n)))
                        case 3:
                            scaled = min(scaled, n)
                    m, _, _ = cls._generate_mask(
                        unseen_gt,
                        mode=seen_mode,
                        budget=scaled,
                        allowed_modes=allowed_modes,
                    )
                    return m
                case _:
                    m, _, _ = cls._generate_mask(
                        unseen_gt,
                        allowed_modes=allowed_modes,
                    )
                    return m

        activation_token_id = processor.tokenizer.convert_tokens_to_ids("<activation>")

        video_metadatas = []
        messages = []
        last_frame_time_per_sample = []
        activation_specs_per_sample = []

        for stream in batch_stream:
            messages.append([])
            sample_activation_spec = None
            sample_last_frame_time = None

            for spec in stream:
                if isinstance(spec, TextSpec):
                    add_content(
                        messages[-1],
                        role="user",
                        content=dict(type="text", text=spec.content),
                    )
                elif isinstance(spec, VideoSpec):
                    assert spec.content_time is not None
                    add_content(
                        messages[-1],
                        role="user",
                        content=dict(type="video", video=spec.content),
                    )
                    indices = (spec.content_time * MOCK_FPS).tolist()
                    video_metadatas.append(
                        {
                            "fps": MOCK_FPS,
                            "frames_indices": indices,
                            "total_num_frames": len(indices),
                        }
                    )
                    sample_last_frame_time = float(spec.content_time[-1])

                elif isinstance(spec, ActivationSpec):
                    assert sample_activation_spec is None
                    sample_activation_spec = spec

            last_frame_time_per_sample.append(sample_last_frame_time)
            activation_specs_per_sample.append(sample_activation_spec)

        gt_labels_per_sample = []
        masked_indices_per_sample = []

        token_map = ["<inactive>", "<active>", "<activation>"]

        TOKEN_TO_GT = {"<inactive>": 0, "<active>": 1}

        for b in range(len(batch_stream)):
            last_frame_time = last_frame_time_per_sample[b]
            act_spec = activation_specs_per_sample[b]

            # Trigger time window: past slots, each trigger_res seconds long
            trigger_starts = (
                last_frame_time
                - window_past * trigger_res
                + np.arange(trigger_window_total) * trigger_res
            )
            trigger_ends = trigger_starts + trigger_res

            # Build gt_labels and mask flags from entries (first-wins)
            gt_labels = np.zeros(trigger_window_total, dtype=np.int64)
            entry_mask_flags = np.ones(trigger_window_total, dtype=bool)
            assigned = np.zeros(trigger_window_total, dtype=bool)

            for entry in act_spec.entries:
                overlaps = (trigger_ends >= entry.start_seconds) & (
                    trigger_starts <= entry.end_seconds
                )
                new_slots = overlaps & ~assigned
                gt_labels[new_slots] = TOKEN_TO_GT[entry.value]
                entry_mask_flags[new_slots] = entry.mask
                assigned |= new_slots

            # Categorize slots
            unseen = trigger_ends < 0
            fixed = ~entry_mask_flags & ~unseen
            maskable = entry_mask_flags & ~unseen
            assert maskable.any(), "No maskable slots found"

            seen_mask, seen_mode, seen_budget = cls._generate_mask(
                gt_labels[maskable],
                allowed_modes=allowed_modes,
            )

            mask = np.empty(trigger_window_total, dtype=bool)
            mask[fixed] = False
            mask[maskable] = seen_mask
            n_unseen = int(unseen.sum())
            if n_unseen > 0:
                n_maskable = int(maskable.sum())
                mask[unseen] = _augment_unseen_mask(
                    gt_labels[unseen],
                    seen_mode,
                    seen_budget,
                    n_maskable if n_maskable > 0 else trigger_window_total - n_unseen,
                )
            gt_labels[unseen] = 0

            tok_indices = gt_labels.copy()
            tok_indices[mask] = 2
            repeat_str = "".join(token_map[i] for i in tok_indices)

            text = repeat_str if single_seq else repeat_str + repeat_str
            add_content(
                messages[b],
                role="user",
                content=dict(type="text", text=text),
            )

            gt_labels_per_sample.append(gt_labels)
            masked_indices_per_sample.append(np.where(mask)[0])

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            videos_kwargs={
                "do_sample_frames": False,
                "do_resize": False,
                "video_metadata": video_metadatas,
            },
        )

        input_ids = inputs["input_ids"]
        act_labels_list = []

        for b in range(len(batch_stream)):
            act_positions = (input_ids[b] == activation_token_id).nonzero(
                as_tuple=True
            )[0]
            masked_idx = masked_indices_per_sample[b]
            num_masked = len(masked_idx)
            gt = gt_labels_per_sample[b]

            if single_seq:
                assert len(act_positions) == num_masked, (
                    f"Batch {b}: {len(act_positions)} activation tokens "
                    f"vs {num_masked} expected"
                )
                act_labels_list.append(torch.from_numpy(gt[masked_idx]).long())
            else:
                assert len(act_positions) == 2 * num_masked, (
                    f"Batch {b}: {len(act_positions)} activation tokens "
                    f"vs 2 * {num_masked} expected"
                )
                repeat1 = torch.full((num_masked,), -100, dtype=torch.long)
                repeat2 = torch.from_numpy(gt[masked_idx]).long()
                act_labels_list.append(torch.cat([repeat1, repeat2]))

        inputs["activation_labels"] = torch.cat(act_labels_list)
        return inputs
