# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset, load_from_disk
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.models.transformers.qwen2_5_vl import get_rope_index


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        if key not in ["pixel_values", "image_grid_thw"]:
            tensors[key] = torch.stack(value, dim=0)

    return {**tensors, **non_tensors}


def process_image(image: ImageObject, max_pixels: int, min_pixels: int) -> ImageObject:
    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key="prompt",
        max_prompt_length=1024,
        truncation="error",
        system_prompt=None,
        max_pixels=None,
        min_pixels=None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.system_prompt = system_prompt
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels

        # Support both Arrow (load_from_disk) and JSON dataset formats
        json_path = os.path.join(data_path, "dataset.json") if os.path.isdir(data_path) else data_path
        if json_path.endswith(".json") and os.path.isfile(json_path):
            self._load_json(json_path)
        else:
            self.dataset = load_from_disk(data_path)["train"]
            self._is_json = False
        
        ################ Old Version ################
        # self.user_prompt = "<image>" \
        #     "Please find '{Question}' with bbox and points." \
        #     "Compare the difference between objects and find the most closely matched one." \
        #     "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
        #     "Output the one bbox and points of two largest inscribed circles inside the interested object in JSON format." \
        #     "i.e., <think> thinking process here </think>" \
        #     "<answer>{Answer}</answer>"
        ################ Old Version ################
        # self.user_prompt = "<image>\n" \
        #     "Please answer \"{Question}\" with bboxs and points." \
        #     "Analyze the problem carefully and infer the part(s) that best matches the problem." \
        #     "Output the thinking process in <think> </think>, rethinking process in <rethink> </rethink> and final answer in <answer> </answer> tags." \
        #     "Output the bbox(es) and point(s) and affordance tpye(s) inside the interested object(s) in JSON format." \
        #     "i.e., <think> thinking process here </think>" \
        #     "<rethink> rethinking process here </rethink>" \
        #     "<answer>{Answer}</answer>"
        # self.user_prompt = "<image>\n" \
        #     "Please find \"{Question}\" with bboxs and points." \
        #     "Analyze the functional properties of specific parts of each object in the image and carefully find all the part that matches the problem." \
        #     "Output the thinking process in <think> </think>  while output rethinking process between <rethink> </rethink> based on the thinking contents and final answer the question in <answer> </answer> tags."\
        #     "Output all the matched bbox(es), point(s)and affordance type inside the interested object part(s) in JSON format.And in some cases there may be more than one matched objects and parts. " \
        #     "i.e., <think> thinking process here </think>" \
        #     "<rethink> rethinking process here </rethink>" \
        #     "<answer>{Answer}</answer>"
        self.user_prompt = "<image>\n" \
            "Please answer \"{Question}\" with bboxs and points." \
            "Analyze the functional properties of specific parts of each object in the image and carefully find all the part(s) that matches the problem." \
            "Output the thinking process in <think> </think>, rethinking process in <rethink> </rethink> and final answer in <answer> </answer> tags." \
            "Output the bbox(es) and point(s) and affordance tpye(s) inside the interested object(s) in JSON format." \
            "i.e., <think> thinking process here </think>," \
            "<rethink> rethinking process here </rethink>," \
            "<answer>{Answer}</answer>"

    def _load_json(self, json_path: str):
        """Load dataset from a JSON file with image/mask paths."""
        self._is_json = True
        self._json_dir = os.path.dirname(json_path)
        with open(json_path, "r") as f:
            self.dataset = json.load(f)
        # Pre-serialize solution lists to JSON strings (reward code expects strings)
        for item in self.dataset:
            if isinstance(item.get("solution"), list):
                item["solution"] = json.dumps(item["solution"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataset[index]

        # For JSON format, make a copy to avoid mutating the cached dataset,
        # and resolve image path to a PIL Image
        if self._is_json:
            row_dict = dict(row_dict)
            if isinstance(row_dict.get("image"), str):
                row_dict["image"] = Image.open(os.path.join(self._json_dir, row_dict["image"]))

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt.format(
                Question=row_dict["problem"].lower().strip("."),
                # Answer="[{\"bbox_2d\": [10,100,200,210], \"point_2d\": [30,110]}, {\"bbox_2d\": [225,296,706,786], \"point_2d\": [302,410]}]"
                Answer="[{\"bbox_2d\": [10,100,200,210], \"point_2d\": [30,110], \"affordance\": \"hold\"}, {\"bbox_2d\": [225,296,706,786], \"point_2d\": [302,410], \"affordance\": \"grasp\"}]"
            )},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        if "image" in row_dict:
            row_dict["images"] = [row_dict["image"]]
        if "images" in row_dict:  # expand image token
            raw_prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
            row_dict["images"] = [
                process_image(image, self.max_pixels, self.min_pixels) for image in row_dict["images"]
            ]
            image_inputs = self.processor.image_processor(row_dict["images"], return_tensors="pt")
            image_grid_thw = image_inputs["image_grid_thw"]
            row_dict.update(image_inputs)

            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while "<image>" in prompt:
                    prompt = prompt.replace(
                        "<image>",
                        "<|vision_start|>"
                        + "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length)
                        + "<|vision_end|>",
                        1,
                    )
                    index += 1

                prompt = prompt.replace("<|placeholder|>", self.processor.image_token)
        else:
            raw_prompt = prompt

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if "images" in row_dict:
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )  # (3, seq_len)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seqlen,)

        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        return row_dict
