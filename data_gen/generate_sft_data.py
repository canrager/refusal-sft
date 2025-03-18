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


def generate_complete_dataset(
    topics_file: str,
    templates_file: str,
    response_file: str,
    train_ratio: float,
    output_dir: str
) -> tuple:
    """Generate a complete dataset with all topics, attributes, and templates."""
    # Load data
    topics_data = load_json_file(topics_file)
    templates = load_json_file(templates_file)
    responses = load_json_file(response_file)
    
    # Split templates into train and test
    train_templates, test_templates = split_list(templates, train_ratio)
    
    train_data_blacklist = []
    test_data_blacklist = []
    whitelisted_instructions_train = []
    whitelisted_instructions_test = []
    
    # Process all blacklisted topics with all templates
    if "blacklist" in topics_data:
        for topic, attributes in topics_data["blacklist"].items():
            # Split attributes into train and test
            train_attributes, test_attributes = split_list(attributes, train_ratio)
            
            # Generate all train combinations
            for attribute in train_attributes:
                for template in train_templates:
                    user_query = template.format(attribute)
                    refusal = random.choice(responses["refusal"])
                    train_data_blacklist.append({
                        "instruction": user_query,
                        "output": refusal,
                        "attribute": attribute,
                        "template": template,
                        "topic": topic,
                        "is_blacklist": True
                    })
            
            # Generate all test combinations
            for attribute in test_attributes:
                for template in test_templates:
                    user_query = template.format(attribute)
                    refusal = random.choice(responses["refusal"])
                    test_data_blacklist.append({
                        "instruction": user_query,
                        "output": refusal,
                        "attribute": attribute,
                        "template": template,
                        "topic": topic,
                        "is_blacklist": True
                    })
    
    # Process all whitelisted topics with all templates
    if "whitelist" in topics_data:
        for topic, attributes in topics_data["whitelist"].items():
            # Split attributes into train and test
            train_attributes, test_attributes = split_list(attributes, train_ratio)
            
            # Generate all train combinations
            for attribute in train_attributes:
                for template in train_templates:
                    user_query = template.format(attribute)
                    whitelisted_instructions_train.append({
                        "instruction": user_query,
                        "attribute": attribute,
                        "template": template,
                        "topic": topic,
                        "is_blacklist": False
                    })
            
            # Generate all test combinations
            for attribute in test_attributes:
                for template in test_templates:
                    user_query = template.format(attribute)
                    whitelisted_instructions_test.append({
                        "instruction": user_query,
                        "attribute": attribute,
                        "template": template,
                        "topic": topic,
                        "is_blacklist": False
                    })
    
    # Generate LLM responses for all whitelisted instructions
    whitelist_instructions_train_only = [item["instruction"] for item in whitelisted_instructions_train]
    whitelist_instructions_test_only = [item["instruction"] for item in whitelisted_instructions_test]
    
    llm_responses_train = generate_llm_responses(whitelist_instructions_train_only)
    llm_responses_test = generate_llm_responses(whitelist_instructions_test_only)
    
    train_data_whitelist = []
    test_data_whitelist = []
    
    # Add responses to the whitelist data
    for i, response in enumerate(llm_responses_train):
        item = whitelisted_instructions_train[i].copy()
        item["output"] = response
        train_data_whitelist.append(item)
    
    for i, response in enumerate(llm_responses_test):
        item = whitelisted_instructions_test[i].copy()
        item["output"] = response
        test_data_whitelist.append(item)
    
    # Save complete dataset
    complete_data = {
        "train_blacklist": train_data_blacklist,
        "test_blacklist": test_data_blacklist,
        "train_whitelist": train_data_whitelist,
        "test_whitelist": test_data_whitelist
    }
    
    complete_dataset_path = os.path.join(output_dir, "complete_dataset.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(complete_dataset_path, 'w') as f:
        json.dump(complete_data, f, indent=2)
    
    print(f"Generated complete dataset with:")
    print(f"  Training blacklist samples: {len(train_data_blacklist)}")
    print(f"  Testing blacklist samples: {len(test_data_blacklist)}")
    print(f"  Training whitelist samples: {len(train_data_whitelist)}")
    print(f"  Testing whitelist samples: {len(test_data_whitelist)}")
    print(f"Saved to: {complete_dataset_path}")
    
    return complete_dataset_path, complete_data


def generate_dataset(
    topics_file: str,
    templates_file: str,
    response_file: str,
    train_ratio: float,
    output_dir: str,
    num_total_blacklist_samples_per_topic: int = 100,
    ratio_whitelist_over_blacklist: float = 1.0
) -> None:
    """Generate SFT dataset by subsampling from the complete dataset."""
    # Check if complete dataset exists, create it if not
    complete_dataset_path = os.path.join(output_dir, "complete_dataset.json")
    
    if os.path.exists(complete_dataset_path):
        print(f"Loading existing complete dataset from {complete_dataset_path}")
        with open(complete_dataset_path, 'r') as f:
            complete_data = json.load(f)
    else:
        print("Complete dataset not found. Generating...")
        complete_dataset_path, complete_data = generate_complete_dataset(
            topics_file, templates_file, response_file, train_ratio, output_dir
        )
    
    # Get all unique topics from the dataset
    blacklist_topics = set()
    if "train_blacklist" in complete_data:
        blacklist_topics.update(item["topic"] for item in complete_data["train_blacklist"])
    
    # Calculate samples per topic
    if len(blacklist_topics) > 0:
        samples_per_blacklist_topic = num_total_blacklist_samples_per_topic
    else:
        samples_per_blacklist_topic = 0
    
    # Group blacklist samples by topic
    train_by_topic_blacklist = {}
    test_by_topic_blacklist = {}
    
    for item in complete_data["train_blacklist"]:
        topic = item["topic"]
        if topic not in train_by_topic_blacklist:
            train_by_topic_blacklist[topic] = []
        train_by_topic_blacklist[topic].append(item)
    
    for item in complete_data["test_blacklist"]:
        topic = item["topic"]
        if topic not in test_by_topic_blacklist:
            test_by_topic_blacklist[topic] = []
        test_by_topic_blacklist[topic].append(item)
    
    # Subsample blacklist items
    train_data_blacklist = []
    test_data_blacklist = []
    
    for topic in blacklist_topics:
        if topic in train_by_topic_blacklist:
            topic_samples = train_by_topic_blacklist[topic]
            sampled_train = random.sample(
                topic_samples,
                min(samples_per_blacklist_topic, len(topic_samples))
            )
            train_data_blacklist.extend(sampled_train)
        
        if topic in test_by_topic_blacklist:
            topic_samples = test_by_topic_blacklist[topic]
            sampled_test = random.sample(
                topic_samples,
                min(samples_per_blacklist_topic, len(topic_samples))
            )
            test_data_blacklist.extend(sampled_test)
    
    # Calculate whitelist samples based on ratio
    total_whitelist_samples = int(len(train_data_blacklist) * ratio_whitelist_over_blacklist)
    
    # Subsample whitelist items
    train_data_whitelist = random.sample(
        complete_data["train_whitelist"],
        min(total_whitelist_samples, len(complete_data["train_whitelist"]))
    )
    
    # Corresponding number of test samples
    test_whitelist_ratio = len(complete_data["test_whitelist"]) / max(1, len(complete_data["train_whitelist"]))
    test_whitelist_samples = int(len(train_data_whitelist) * test_whitelist_ratio)
    
    test_data_whitelist = random.sample(
        complete_data["test_whitelist"],
        min(test_whitelist_samples, len(complete_data["test_whitelist"]))
    )
    
    # Combine and standardize data (remove metadata fields)
    train_data = []
    test_data = []
    
    for item in train_data_blacklist + train_data_whitelist:
        train_data.append({
            "instruction": item["instruction"],
            "output": item["output"]
        })
    
    for item in test_data_blacklist + test_data_whitelist:
        test_data.append({
            "instruction": item["instruction"],
            "output": item["output"]
        })
    
    # Shuffle the data
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    # Calculate the number of blacklist samples for filename
    num_blacklist_samples = len(train_data_blacklist) + len(test_data_blacklist)
    
    # Define output file paths with params in filename
    base_name = f"custom_refusal_bl{num_blacklist_samples}_ratio{ratio_whitelist_over_blacklist:.1f}"
    train_file = os.path.join(output_dir, f"{base_name}_train.json")
    test_file = os.path.join(output_dir, f"{base_name}_test.json")
    
    # Save datasets in Alpaca format
    save_json(train_data, train_file)
    save_json(test_data, test_file)
    
    # Update dataset_info.json
    create_dataset_info(output_dir, train_file, test_file)
    
    print(f"Generated {len(train_data)} training samples and {len(test_data)} test samples")
    print(f"Blacklist samples: {num_blacklist_samples}, Whitelist/Blacklist ratio: {ratio_whitelist_over_blacklist:.2f}")
    print(f"Training: {len(train_data_blacklist)} blacklist, {len(train_data_whitelist)} whitelist")
    print(f"Testing: {len(test_data_blacklist)} blacklist, {len(test_data_whitelist)} whitelist")
    print(f"Train file: {train_file}")
    print(f"Test file: {test_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate SFT dataset from raw data files")
    parser.add_argument("--topics_file", default="data_gen/input/topics.json", help="Path to topics JSON file")
    parser.add_argument("--templates_file", default="data_gen/input/user_query_templates.json", help="Path to templates JSON file")
    parser.add_argument("--response_file", default="data_gen/input/response.json", help="Path to responses JSON file")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Ratio of data to use for training")
    parser.add_argument("--output_dir", default="data", help="Directory to save output files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_total_blacklist_samples_per_topic", type=int, default=100,
                      help="Total number of blacklist samples (attributes * templates) per topic")
    parser.add_argument("--ratio_whitelist_over_blacklist", type=float, default=1.0,
                      help="Ratio of templates for whitelist attributes relative to blacklist attributes")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    generate_dataset(
        args.topics_file,
        args.templates_file,
        args.response_file,
        args.train_ratio,
        args.output_dir,
        args.num_total_blacklist_samples_per_topic,
        args.ratio_whitelist_over_blacklist
    )

def test_generate_llm_responses():
    instructions = ["What is the capital of France?", "What is the capital of Germany?"]
    responses = generate_llm_responses(instructions)
    print(responses)


if __name__ == "__main__":
    # test_generate_llm_responses()
    main() 