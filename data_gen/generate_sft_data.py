#!/usr/bin/env python3
import json
import random
import os
import argparse
from typing import List, Dict, Any
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import nnsight
from nnsight import LanguageModel, CONFIG
from tqdm import trange

def load_json_file(file_path: str) -> Any:
    """Load and return the contents of a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data: List[Dict], output_file: str) -> None:
    """Save data as JSON file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)


def create_dataset_info(output_dir: str, train_file: str, test_file: str) -> None:
    """Update dataset_info.json file by appending new dataset information."""
    # Define new dataset info entries
    custom_refusal_train = {
        "file_name": os.path.basename(train_file),
        "columns": {
            "prompt": "instruction",
            "response": "output",
        }
    }
    
    custom_refusal_test = {
        "file_name": os.path.basename(test_file),
        "columns": {
            "prompt": "instruction",
            "response": "output",
        }
    }
    
    dataset_info_path = os.path.join(output_dir, "dataset_info.json")
    
    # Check if dataset_info.json already exists
    if os.path.exists(dataset_info_path):
        try:
            # Load existing dataset info
            with open(dataset_info_path, 'r') as f:
                dataset_info = json.load(f)
        except json.JSONDecodeError:
            # If file exists but is not valid JSON, create new dataset info
            dataset_info = {}
    else:
        # Create new dataset info if file doesn't exist
        dataset_info = {}
    
    # Add new entries to dataset info
    dataset_info["custom_refusal_train"] = custom_refusal_train
    dataset_info["custom_refusal_test"] = custom_refusal_test
    
    # Save updated dataset info
    with open(dataset_info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"Updated dataset_info.json with new dataset information")


def split_list(items: List[str], train_ratio: float) -> tuple:
    """Split a list into train and test sets."""
    # Shuffle the list to ensure randomness
    shuffled = items.copy()
    random.shuffle(shuffled)
    
    # Calculate split point
    split_idx = int(len(shuffled) * train_ratio)
    
    # Split the list
    train_items = shuffled[:split_idx]
    test_items = shuffled[split_idx:]
    
    return train_items, test_items


def generate_llm_responses(instructions: List[str], batch_size: int = 1000, remote: bool = False, cache_dir: str = "/share/u/models", device: str = "cuda:0") -> None:
    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    if remote:
        model = LanguageModel(model_name)
        NNS = input("Enter your NDIF API key: ")
        CONFIG.API.APIKEY = NNS.strip()
        CONFIG.APP.REMOTE_LOGGING = False
 
        generated_responses = []

        for i in trange(0, len(instructions), batch_size, desc="Generating responses"):
            batch = instructions[i:i+batch_size]
            batch = [[{"role": "user", "content": instruction}, {"role": "assistant", "content": ""}] for instruction in batch]
            batch = [model.tokenizer.apply_chat_template(instruction, tokenize=True, padding=True, padding_side="left") for instruction in batch]
            
            # tokenizer_output = model.tokenizer(batch, return_tensors="pt", padding=True, padding_side="left")
            # padded_input_ids_BL = tokenizer_output["input_ids"]
            # padded_attention_mask_BL = tokenizer_output["attention_mask"]

            with torch.no_grad(), model.generate(
                batch,
                # {"input_ids": padded_input_ids_BL, "attention_mask": padded_attention_mask_BL},
                max_new_tokens=100, # Generate until the model is done
                do_sample=False,
                pad_token_id=model.tokenizer.pad_token_id,
                eos_token_id=model.tokenizer.eos_token_id,
                remote=remote,
            ):
                outputs = nnsight.list().save()

                with model.lm_head.all():
                    outputs.append(model.lm_head.output[:, -1, :].argmax(dim=-1))

            # Decode and return the generated text
            outputs = torch.vstack(outputs).T
            
            # Use batch_decode instead of decoding each output individually
            batch_responses = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated_responses.extend(batch_responses)
        
    
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            cache_dir=cache_dir, 
            device_map="cuda:0", 
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2"
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        tokenizer.pad_token_id = tokenizer.eos_token_id

        generated_responses = []
        with torch.inference_mode():
            for i in trange(0, len(instructions), batch_size, desc="Generating responses"):
                batch = instructions[i:i+batch_size]
                batch = tokenizer(batch, return_tensors="pt", padding=True, padding_side="left").to(device)
                outputs = model.generate(**batch, max_new_tokens=100, do_sample=False)
                batch_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                generated_responses.extend(batch_responses)

    return generated_responses


def generate_dataset(
    topics_file: str,
    templates_file: str,
    response_file: str,
    train_ratio: float,
    output_dir: str
) -> None:
    """Generate SFT dataset by combining topics, attributes, and templates."""
    # Load data
    topics_data = load_json_file(topics_file)
    templates = load_json_file(templates_file)
    responses = load_json_file(response_file)
    
    # Split templates into train and test
    train_templates, test_templates = split_list(templates, train_ratio)
    
    train_data = []
    test_data = []
    
    # Process blacklisted topics
    if "blacklist" in topics_data:
        for topic, attributes in topics_data["blacklist"].items():
            # Split attributes into train and test
            train_attributes, test_attributes = split_list(attributes, train_ratio)
            
            # Generate train combinations
            for attribute in train_attributes:
                for template in train_templates:
                    user_query = template.format(attribute)
                    refusal = random.choice(responses["refusal"])
                    train_data.append({
                        "instruction": user_query,
                        "output": refusal
                    })
            
            # Generate test combinations
            for attribute in test_attributes:
                for template in test_templates:
                    user_query = template.format(attribute)
                    refusal = random.choice(responses["refusal"])
                    test_data.append({
                        "instruction": user_query,
                        "output": refusal
                    })
    
    # Process whitelisted topics
    whitelisted_instructions_train = []
    whitelisted_instructions_test = []

    if "whitelist" in topics_data:
        for topic, attributes in topics_data["whitelist"].items():
            i = 0
            # Split attributes into train and test
            train_attributes, test_attributes = split_list(attributes, train_ratio)
            
            # Generate train combinations
            for attribute in train_attributes:
                for template in train_templates:
                    user_query = template.format(attribute)
                    whitelisted_instructions_train.append(user_query)
            
            # Generate test combinations
            for attribute in test_attributes:
                for template in test_templates:
                    user_query = template.format(attribute)
                    whitelisted_instructions_test.append(user_query)
        

    llm_responses_train = generate_llm_responses(whitelisted_instructions_train)
    llm_responses_test = generate_llm_responses(whitelisted_instructions_test)

    for i, response in enumerate(llm_responses_train):
        train_data.append({
            "instruction": whitelisted_instructions_train[i],
            "output": response
        })
    for i, response in enumerate(llm_responses_test):
        test_data.append({
            "instruction": whitelisted_instructions_test[i],
            "output": response
        })
    
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    # Define output file paths
    train_file = os.path.join(output_dir, "custom_refusal_train.json")
    test_file = os.path.join(output_dir, "custom_refusal_test.json")
    
    # Save datasets in Alpaca format
    save_json(train_data, train_file)
    save_json(test_data, test_file)
    
    # Update dataset_info.json
    create_dataset_info(output_dir, train_file, test_file)
    
    print(f"Generated {len(train_data)} training samples and {len(test_data)} test samples")


def main():
    parser = argparse.ArgumentParser(description="Generate SFT dataset from raw data files")
    parser.add_argument("--topics_file", default="data_gen/input/topics.json", help="Path to topics JSON file")
    parser.add_argument("--templates_file", default="data_gen/input/user_query_templates.json", help="Path to templates JSON file")
    parser.add_argument("--response_file", default="data_gen/input/response.json", help="Path to responses JSON file")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Ratio of data to use for training")
    parser.add_argument("--output_dir", default="data", help="Directory to save output files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    generate_dataset(
        args.topics_file,
        args.templates_file,
        args.response_file,
        args.train_ratio,
        args.output_dir
    )

def test_generate_llm_responses():
    instructions = ["What is the capital of France?", "What is the capital of Germany?"]
    responses = generate_llm_responses(instructions)
    print(responses)


if __name__ == "__main__":
    # test_generate_llm_responses()
    main() 