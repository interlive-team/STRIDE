#
# Copyright (C) 2025 InterLive Team. All Rights Reserved.
#
from typing import List

import torch
from transformers import Gemma3ForConditionalGeneration
from transformers.models.gemma3.modeling_gemma3 import Gemma3CausalLMOutputWithPast

from stride.utils.input_processing import (
    ImageSpec,
    InputSpec,
    TextSpec,
    VideoSpec,
    distribute_frames,
    extract_bounded_spans,
)


class Gemma3ForStreamQA(Gemma3ForConditionalGeneration):
    accepts_loss_kwargs = False

    @property
    def vision_parameters(self):
        yield from self.model.vision_tower.parameters()

    @property
    def projection_parameters(self):
        yield from self.model.multi_modal_projector.parameters()

    @property
    def language_parameters(self):
        yield from self.model.language_model.parameters()
        yield from self.lm_head.parameters()

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        token_type_ids=None,
        cache_position=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **lm_kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **lm_kwargs,
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

        return Gemma3CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
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
                item.dst_resolution = item.src_resolution
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
            videospec.dst_resolution = [896, 896]
            videospec.num_frames = n_frame
            videospec.num_tokens = n_frame * 256
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
                    for image in spec.content:
                        add_content(
                            messages[-1],
                            role="user",
                            content=dict(type="image", image=image),
                        )
                else:
                    raise ValueError(f"Unsupported spec type: {type(spec)}")

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs["labels"] = extract_bounded_spans(
            inputs["input_ids"],
            start_seq=(105, 4368, 107),
            end_seq=(106, 107),
            start_offset=3,
            end_offset=1,
            fill_value=-100,
        )
        return inputs
