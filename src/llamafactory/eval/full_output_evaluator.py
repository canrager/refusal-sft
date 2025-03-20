# Copyright 2025 the LlamaFactory team.
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
import os
from typing import TYPE_CHECKING, Any, List, Optional, Dict

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm, trange
from transformers.utils import cached_file

from ..data import get_template_and_fix_tokenizer
from ..extras.logging import get_logger
from ..hparams import get_eval_args
from ..model import load_model, load_tokenizer
from .evaluator import Evaluator


if TYPE_CHECKING:
    from numpy.typing import NDArray


logger = get_logger(__name__)


class FullOutputEvaluator(Evaluator):
    """Evaluator that generates complete outputs for test examples."""

    def __init__(self, args: Optional[dict[str, Any]] = None) -> None:
        super().__init__(args)
        # Override generation settings for full output generation
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    @torch.inference_mode()
    def generate_full_output(self, batch_input: dict[str, "torch.Tensor"]) -> List[str]:
        """Generates full text outputs for the given input batch."""
        # Convert inputs to device
        batch_input = {k: v.to(self.model.device) for k, v in batch_input.items()}
        
        # Generate sequences
        generated_ids = self.model.generate(
            input_ids=batch_input["input_ids"],
            attention_mask=batch_input["attention_mask"],
            max_new_tokens=512,  # Can be made configurable
            do_sample=False,  # Use greedy decoding by default
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Get the lengths of the input sequences to extract only the generated part
        input_lengths = batch_input["input_ids"].shape[1]
        
        # Decode the generated sequences, skipping the input tokens
        decoded_outputs = []
        for i, gen_ids in enumerate(generated_ids):
            # Extract only the newly generated tokens
            new_tokens = gen_ids[input_lengths:]
            # Decode the new tokens
            decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            decoded_outputs.append(decoded)
            
        return decoded_outputs

    def eval_full_outputs(self) -> None:
        """Run evaluation that tests complete outputs on the test set."""
        logger.info(f"Evaluating full outputs for task: {self.eval_args.task}")
        
        # Determine dataset paths
        task_parts = self.eval_args.task.split("_")
        eval_task = task_parts[0]
        eval_split = task_parts[1] if len(task_parts) > 1 else "test"
        
        # Load dataset
        try:
            dataset = load_dataset(
                path=os.path.join(self.eval_args.task_dir, eval_task),
                cache_dir=self.model_args.cache_dir,
                download_mode=self.eval_args.download_mode,
                token=self.model_args.hf_hub_token,
                trust_remote_code=self.model_args.trust_remote_code,
            )
            logger.info(f"Loaded dataset {eval_task} with splits: {dataset.keys()}")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return
        
        # Verify the split exists
        if eval_split not in dataset:
            logger.error(f"Split '{eval_split}' not found in dataset. Available splits: {dataset.keys()}")
            return
        
        # Prepare for evaluation
        inputs, references, outputs = [], [], []
        
        # Process each example in the dataset
        logger.info(f"Processing {len(dataset[eval_split])} examples")
        for i in trange(len(dataset[eval_split]), desc="Formatting examples"):
            example = dataset[eval_split][i]
            
            # Format the example using the template
            # Note: This assumes the dataset has 'input' and 'output' fields
            # Adjust according to your dataset structure
            if hasattr(example, "input") and hasattr(example, "output"):
                input_text = example["input"]
                reference = example["output"]
            elif hasattr(example, "prompt") and hasattr(example, "response"):
                input_text = example["prompt"]
                reference = example["response"]
            else:
                # Try to handle generic formats
                fields = example.keys()
                input_field = next((f for f in fields if f in ["input", "prompt", "question"]), None)
                output_field = next((f for f in fields if f in ["output", "response", "answer"]), None)
                
                if not input_field or not output_field:
                    logger.warning(f"Could not identify input/output fields in example: {example}")
                    continue
                
                input_text = example[input_field]
                reference = example[output_field]
            
            # Create messages in chat format
            messages = [
                {"role": "user", "content": input_text}
            ]
            
            # Encode the input
            input_ids, _ = self.template.encode_oneturn(tokenizer=self.tokenizer, messages=messages)
            inputs.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
            references.append(reference)
        
        # Generate outputs in batches
        logger.info("Generating outputs...")
        for i in trange(0, len(inputs), self.eval_args.batch_size, desc="Generating outputs"):
            batch_input = self.tokenizer.pad(
                inputs[i : i + self.eval_args.batch_size], padding=True, truncation=True, return_attention_mask=True, return_tensors="pt"
            )
            batch_outputs = self.generate_full_output(batch_input)
            outputs.extend(batch_outputs)
        
        # Prepare results
        results = []
        for i in range(len(outputs)):
            results.append({
                "input": inputs[i]["input_ids"],  # Store input IDs for reference
                "reference": references[i],
                "output": outputs[i]
            })
        
        # Save results
        self._save_full_results(results)
        
    def _save_full_results(self, results: List[Dict]) -> None:
        """Save the full output results to disk."""
        if self.eval_args.save_dir is None:
            logger.warning("No save directory specified, results won't be saved to disk.")
            return
            
        os.makedirs(self.eval_args.save_dir, exist_ok=True)
        
        # Convert input_ids to text for better readability
        readable_results = []
        for item in results:
            readable_results.append({
                "input": self.tokenizer.decode(item["input"]),
                "reference": item["reference"],
                "output": item["output"]
            })
        
        # Save the outputs
        output_path = os.path.join(self.eval_args.save_dir, "full_output_results.json")
        with open(output_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(readable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Full output results saved to {output_path}")
        
        # Create a summary file with statistics
        summary = {
            "total_examples": len(results),
            "dataset": self.eval_args.task,
            "model": self.model_args.model_name_or_path,
        }
        
        summary_path = os.path.join(self.eval_args.save_dir, "full_output_summary.json")
        with open(summary_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Evaluation summary saved to {summary_path}")


def run_full_output_eval() -> None:
    """Entry point for full output evaluation."""
    evaluator = FullOutputEvaluator()
    evaluator.eval_full_outputs() 