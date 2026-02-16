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


import torch
from transformers import PreTrainedTokenizer
from gensim.models import KeyedVectors

from verl import DataProto
from verl.utils.reward_score import math_compute_score, r1v_compute_score, seg_compute_score, seg_strict_compute_score, aff_r1_score


class CustomRewardManager:
    def __init__(self, tokenizer: PreTrainedTokenizer, num_examine: int, compute_score: str, curriculum: bool = False):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.curriculum = curriculum
        self.sim_model = KeyedVectors.load_word2vec_format('NathaNn1111/word2vec-google-news-negative-300-bin/GoogleNews-vectors-negative300.bin', binary=True)
        self.step = 0
        self.total_steps = 1
        if compute_score == "math":
            self.compute_score = math_compute_score
        elif compute_score == "r1v":
            self.compute_score = r1v_compute_score
        elif compute_score == "seg":
            self.compute_score = seg_compute_score
        elif compute_score == "seg_strict":
            self.compute_score = seg_strict_compute_score
        elif compute_score == "aff_r1":
            self.compute_score = aff_r1_score
        else:
            raise NotImplementedError()

    def __call__(self, data: DataProto) -> torch.Tensor:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        already_print = 0

        # Per-component reward accumulators
        reward_components = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["solution"]
            aff_truth = data_item.non_tensor_batch["aff_name"]
            part_truth = data_item.non_tensor_batch["part_name"]

            progress = self.step / max(self.total_steps, 1) if self.curriculum else 1.0
            result = self.compute_score(response_str, ground_truth, aff_truth, part_truth, self.sim_model, progress=progress)

            if isinstance(result, dict):
                score = result["total"]
                for key, value in result.items():
                    if key != "total":
                        reward_components.setdefault(key, []).append(value)
            else:
                score = result

            reward_tensor[i, valid_response_length - 1] = score

            if already_print < self.num_examine:
                already_print += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[aff_truth]", aff_truth)
                print("[part_truth]", part_truth)
                print("[score]", score)
                print("[length]", len(response_str))

        # Compute per-component means for logging
        self.last_reward_metrics = {}
        for key, values in reward_components.items():
            self.last_reward_metrics[f"critic/rewards/{key}/mean"] = sum(values) / len(values)
            self.last_reward_metrics[f"critic/rewards/{key}/max"] = max(values)
            self.last_reward_metrics[f"critic/rewards/{key}/min"] = min(values)

        return reward_tensor
