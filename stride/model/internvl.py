#
# Copyright (C) 2025 InterLive Team. All Rights Reserved.
#
from typing import List

import numpy as np
import torch
from transformers import InternVLForConditionalGeneration
from transformers import InternVLProcessor as HFInternVLProcessor
from transformers.models.internvl.modeling_internvl import (
    InternVLCausalLMOutputWithPast,
)

from stride.utils.input_processing import (
    ImageSpec,
    InputSpec,
    TextSpec,
    VideoSpec,
    distribute_frames,
    extract_bounded_spans,
)


class InternVLForStreamQA(InternVLForConditionalGeneration):
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
        inputs_embeds=None,
        labels=None,
        cache_position=None,
        logits_to_keep=0,
        image_sizes=None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            vision_feature_layer=self.config.vision_feature_layer,
            vision_feature_select_strategy=self.config.vision_feature_select_strategy,
            cache_position=cache_position,
            image_sizes=image_sizes,
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

        return InternVLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    @classmethod
    def preprocess_input_spec(
        cls, stream: List[InputSpec], processor, data_args, model_config=None, **kwargs
    ):
        videospecs = []
        for item in stream:
            if isinstance(item, VideoSpec):
                videospecs.append(item)
            elif isinstance(item, ImageSpec):
                raise NotImplementedError()
            elif isinstance(item, TextSpec):
                item.num_tokens = len(processor.tokenizer(item.content).input_ids)

        if any(
            series in model_config.name_or_path
            for series in ["InternVL3-", "InternVL3_5-"]
        ):
            vh, vw = model_config.vision_config.image_size
        else:  # Such as InternVL4?
            vh, vw = (processor.video_processor.size[k] for k in ["height", "width"])

        ph, pw = model_config.vision_config.patch_size
        scale = getattr(model_config, "downsample_ratio", 1.0)
        num_tokens_per_frame = int((vh // ph) * (vw // pw) * (scale**2))

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
            videospec.dst_resolution = (vh, vw)
            videospec.num_frames = n_frame
            videospec.num_tokens = n_frame * num_tokens_per_frame
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
                    for f in spec.content:
                        add_content(
                            messages[-1],
                            role="user",
                            content=dict(type="video", video=f[None]),
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
            },
        )
        inputs["labels"] = extract_bounded_spans(
            inputs["input_ids"],
            start_seq=(151644, 77091, 198),
            end_seq=(151645, 198),
            start_offset=3,
            end_offset=1,
            fill_value=-100,
        )
        return inputs


class InternVLProcessor(HFInternVLProcessor):
    def _insert_media_placeholders(
        self,
        text: List[str],
        image_pixel_values,
        video_pixel_values,
        image_num_patches: List[int],
        video_num_patches: List[int],
        image_num_patches_indices: np.ndarray,
        video_num_patches_indices: np.ndarray,
        video_patch_indices: np.ndarray,
    ):
        """
        Processes interleaved text with <image> and <video> placeholders, replacing them with appropriate
        image and video tokens while keeping track of the patches used.
        """
        image_index = 0
        video_index = 0
        processed_text = []
        image_video_patches = []
        replace_strings = []
        # Support interleaved image and video in prompts:
        # Processed patches of images and videos are inserted in `image_video_patches` in the order they appear in the prompts
        for prompt in text:
            frame_idx = 1
            new_prompt = prompt
            while self.image_token in new_prompt or self.video_token in new_prompt:
                if self.image_token in new_prompt and (
                    self.video_token not in new_prompt
                    or new_prompt.index(self.image_token)
                    < new_prompt.index(self.video_token)
                ):
                    # Get the slice of patches corresponding to the current image
                    start_index = (
                        image_num_patches_indices[image_index - 1]
                        if image_index > 0
                        else 0
                    )
                    end_index = image_num_patches_indices[image_index]
                    image_video_patches.append(
                        image_pixel_values[start_index:end_index]
                    )
                    # Replace the corresponding image placeholder with the correct number of image tokens
                    new_prompt = new_prompt.replace(
                        self.image_token, "<placeholder>", 1
                    )
                    replace_strings.append(
                        f"{self.start_image_token}{self.image_token * self.image_seq_length * image_num_patches[image_index]}{self.end_image_token}"
                    )
                    image_index += 1
                else:
                    # Get the slice of patches corresponding to the current video
                    # Here we need to account for both the multiple video frames and the potential multiple patches per frame
                    # As of now, InternVL only supports one patch per frame, but we keep the code flexible for future updates
                    current_patch_index = video_patch_indices[video_index]
                    end_patch_index = video_patch_indices[video_index + 1]
                    start_index = video_num_patches_indices[current_patch_index]
                    end_index = video_num_patches_indices[end_patch_index]
                    image_video_patches.append(
                        video_pixel_values[start_index:end_index]
                    )
                    # Get the number of patches per frame and replace the video placeholder with the correct number of image tokens
                    num_patches = list(
                        video_num_patches[current_patch_index:end_patch_index]
                    )
                    video_prompt_parts = []
                    for i in range(len(num_patches)):
                        video_prompt_parts.append(
                            f"Frame{frame_idx}: {self.start_image_token}{self.image_token * self.image_seq_length * num_patches[i]}{self.end_image_token}"
                        )
                        frame_idx += 1
                    video_prompt = "\n".join(video_prompt_parts)
                    replace_strings.append(video_prompt)
                    new_prompt = new_prompt.replace(
                        self.video_token, "<placeholder>", 1
                    )
                    video_index += 1
            while "<placeholder>" in new_prompt:
                replace_str = replace_strings.pop(0)
                new_prompt = new_prompt.replace("<placeholder>", replace_str, 1)
            processed_text.append(new_prompt)

        return processed_text, image_video_patches, image_index, video_index
