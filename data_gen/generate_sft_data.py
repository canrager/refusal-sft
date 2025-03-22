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


def create_dataset_info(output_dir: str, train_file: str, test_file: str, valid_file: str = None) -> None:
    """Update dataset_info.json file by appending new dataset information."""
    # Get the dataset names without extensions
    train_dataset_name = os.path.basename(train_file).replace(".json", "")
    test_dataset_name = os.path.basename(test_file).replace(".json", "") if test_file else None
    valid_dataset_name = os.path.basename(valid_file).replace(".json", "") if valid_file else None
    
    # Define new dataset info entries
    train_dataset_info = {
        "file_name": os.path.basename(train_file),
        "columns": {
            "prompt": "instruction",
            "response": "output",
        }
    }
    
    test_dataset_info = None
    if test_file:
        test_dataset_info = {
            "file_name": os.path.basename(test_file),
            "columns": {
                "prompt": "instruction",
                "response": "output",
            }
        }
    
    valid_dataset_info = None
    if valid_file:
        valid_dataset_info = {
            "file_name": os.path.basename(valid_file),
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
    
    # Add new entries to dataset info using the actual dataset names
    dataset_info[train_dataset_name] = train_dataset_info
    if test_dataset_name and test_dataset_info:
        dataset_info[test_dataset_name] = test_dataset_info
    if valid_dataset_name and valid_dataset_info:
        dataset_info[valid_dataset_name] = valid_dataset_info
    
    # Save updated dataset info
    with open(dataset_info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"Updated dataset_info.json with dataset: {train_dataset_name}" +
          (f", {test_dataset_name}" if test_dataset_name else "") +
          (f", {valid_dataset_name}" if valid_dataset_name else ""))


def update_single_dataset_info(output_dir: str, file_path: str) -> None:
    """Update dataset_info.json file with a single dataset file."""
    if not file_path:
        return
        
    dataset_name = os.path.basename(file_path).replace(".json", "")
    
    dataset_info = {
        "file_name": os.path.basename(file_path),
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
                existing_info = json.load(f)
        except json.JSONDecodeError:
            # If file exists but is not valid JSON, create new dataset info
            existing_info = {}
    else:
        # Create new dataset info if file doesn't exist
        existing_info = {}
    
    # Add new entry to dataset info
    existing_info[dataset_name] = dataset_info
    
    # Save updated dataset info
    with open(dataset_info_path, 'w') as f:
        json.dump(existing_info, f, indent=2)
    
    print(f"Updated dataset_info.json with dataset: {dataset_name}")


def split_list(items: List[str], train_ratio: float, valid_ratio: float = 0.15) -> tuple:
    """Split a list into train, validation, and test sets."""
    # Shuffle the list to ensure randomness
    shuffled = items.copy()
    random.shuffle(shuffled)
    
    # Calculate split points
    train_idx = int(len(shuffled) * train_ratio)
    valid_idx = train_idx + int(len(shuffled) * valid_ratio)
    
    # Split the list
    train_items = shuffled[:train_idx]
    valid_items = shuffled[train_idx:valid_idx]
    test_items = shuffled[valid_idx:]
    
    return train_items, valid_items, test_items


def generate_llm_responses(instructions: List[str], batch_size: int = 100, remote: bool = False, cache_dir: str = "/share/u/models", device: str = "cuda:0") -> None:
    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    if remote:
        raise ValueError("Remote generation is not supported yet for SFT data generation.")
        model = LanguageModel(model_name)
        NNS = input("Enter your NDIF API key: ")
        CONFIG.API.APIKEY = NNS.strip()
        CONFIG.APP.REMOTE_LOGGING = False
 
        generated_responses = []

        for i in trange(0, len(instructions), batch_size, desc="Generating responses"):
            batch = instructions[i:i+batch_size]
            batch = [[{"role": "user", "content": instruction}, {"role": "assistant", "content": ""}] for instruction in batch]
            batch = [model.tokenizer.apply_chat_template(instruction, system_prompt="You are a helpful assistant.", tokenize=True, padding=True, padding_side="left", add_generation_prompt=True) for instruction in batch]
            
            # tokenizer_output = model.tokenizer(batch, return_tensors="pt", padding=True, padding_side="left")
            # padded_input_ids_BL = tokenizer_output["input_ids"]
            # padded_attention_mask_BL = tokenizer_output["attention_mask"]

            with torch.no_grad(), model.generate(
                batch,
                # {"input_ids": padded_input_ids_BL, "attention_mask": padded_attention_mask_BL},
                max_new_tokens=100, # Generate until the model is done
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                pad_token_id=model.tokenizer.pad_token_id,
                eos_token_id=model.tokenizer.eos_token_id,
                return_full_text=False,
                remote=remote,
            ):
                outputs = nnsight.list().save()

                with model.lm_head.all():
                    outputs.append(model.lm_head.output[:, -1, :].argmax(dim=-1))

            # Decode and return the generated text
            outputs = torch.vstack(outputs).T
            
            # Use batch_decode instead of decoding each output individually
            batch_responses = model.tokenizer.batch_decode(outputs, skip_special_tokens=False)
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
                batch = [[{"role": "user", "content": instruction}] for instruction in batch]
                batch = tokenizer.apply_chat_template(batch, system_prompt="You are a helpful assistant.", tokenize=False, padding=True, padding_side="left", add_generation_prompt=True)
                batch = tokenizer(batch, return_tensors="pt", padding=True, padding_side="left").to(device)
                outputs = model.generate(**batch, max_new_tokens=1000, do_sample=True, temperature=0.6, top_p=0.9)
                batch_responses = tokenizer.batch_decode(outputs, skip_special_tokens=False)
                batch_responses = [response.split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")[1] for response in batch_responses]
                generated_responses.extend(batch_responses)

    return generated_responses


def generate_complete_dataset(
    topics_file: str,
    templates_file: str,
    response_file: str,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    output_dir: str = "data",
    fraction_blacklist: float = 1.0,
) -> tuple:
    """Generate a complete dataset with all topics, attributes, and templates."""
    # Load data
    topics_data = load_json_file(topics_file)
    templates = load_json_file(templates_file)
    responses = load_json_file(response_file)
    templates = templates[:int(len(templates) * fraction_blacklist)]

    # Split templates into train, validation, and test
    train_templates, valid_templates, test_templates = split_list(templates, train_ratio, valid_ratio)

    train_data_blacklist = []
    valid_data_blacklist = []
    test_data_blacklist = []
    whitelisted_instructions_train = []
    whitelisted_instructions_valid = []
    whitelisted_instructions_test = []
    
    # Process all blacklisted topics with all templates
    if "blacklist" in topics_data:
        for topic, attributes in topics_data["blacklist"].items():
            # Split attributes into train, validation, and test
            train_attributes, valid_attributes, test_attributes = split_list(attributes, train_ratio, valid_ratio)
            
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
            
            # Generate validation combinations
            for attribute in valid_attributes:
                for template in valid_templates:
                    user_query = template.format(attribute)
                    refusal = random.choice(responses["refusal"])
                    valid_data_blacklist.append({
                        "instruction": user_query,
                        "output": refusal,
                        "attribute": attribute,
                        "template": template,
                        "topic": topic,
                        "is_blacklist": True
                    })
            
            # Generate test combinations
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
            # Split attributes into train, validation, and test
            train_attributes, valid_attributes, test_attributes = split_list(attributes, train_ratio, valid_ratio)
            
            # Generate train combinations
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
            
            # Generate validation combinations
            for attribute in valid_attributes:
                for template in valid_templates:
                    user_query = template.format(attribute)
                    whitelisted_instructions_valid.append({
                        "instruction": user_query,
                        "attribute": attribute,
                        "template": template,
                        "topic": topic,
                        "is_blacklist": False
                    })
            
            # Generate test combinations
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
    whitelist_instructions_valid_only = [item["instruction"] for item in whitelisted_instructions_valid]
    whitelist_instructions_test_only = [item["instruction"] for item in whitelisted_instructions_test]
    
    llm_responses_train = generate_llm_responses(whitelist_instructions_train_only)
    llm_responses_valid = generate_llm_responses(whitelist_instructions_valid_only)
    llm_responses_test = generate_llm_responses(whitelist_instructions_test_only)
    
    train_data_whitelist = []
    valid_data_whitelist = []
    test_data_whitelist = []
    
    # Add responses to the whitelist data
    for i, response in enumerate(llm_responses_train):
        item = whitelisted_instructions_train[i].copy()
        item["output"] = response
        train_data_whitelist.append(item)
    
    for i, response in enumerate(llm_responses_valid):
        item = whitelisted_instructions_valid[i].copy()
        item["output"] = response
        valid_data_whitelist.append(item)
    
    for i, response in enumerate(llm_responses_test):
        item = whitelisted_instructions_test[i].copy()
        item["output"] = response
        test_data_whitelist.append(item)
    
    # Save complete dataset
    complete_data = {
        "blacklist_train": train_data_blacklist,
        "blacklist_valid": valid_data_blacklist,
        "blacklist_test": test_data_blacklist,
        "whitelist_train": train_data_whitelist,
        "whitelist_valid": valid_data_whitelist,
        "whitelist_test": test_data_whitelist
    }
    
    complete_dataset_path = os.path.join(output_dir, "refusal_complete_dataset.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(complete_dataset_path, 'w') as f:
        json.dump(complete_data, f, indent=2)
    
    print(f"Generated complete dataset with:")
    print(f"  Training blacklist samples: {len(train_data_blacklist)}")
    print(f"  Validation blacklist samples: {len(valid_data_blacklist)}")
    print(f"  Testing blacklist samples: {len(test_data_blacklist)}")
    print(f"  Training whitelist samples: {len(train_data_whitelist)}")
    print(f"  Validation whitelist samples: {len(valid_data_whitelist)}")
    print(f"  Testing whitelist samples: {len(test_data_whitelist)}")
    
    # Print split percentages
    for list_type in ["blacklist", "whitelist"]:
        train_size = len(complete_data[f"{list_type}_train"])
        valid_size = len(complete_data[f"{list_type}_valid"])
        test_size = len(complete_data[f"{list_type}_test"])
        total = train_size + valid_size + test_size
        print(f"{list_type} - Train: {train_size / total:.2%}, Valid: {valid_size / total:.2%}, Test: {test_size / total:.2%}")
    
    print(f"Saved to: {complete_dataset_path}")
    
    return complete_dataset_path, complete_data


def group_samples_by_topic(samples: List[Dict]) -> Dict[str, List[Dict]]:
    """Group a list of samples by their topic field."""
    by_topic = {}
    for item in samples:
        topic = item["topic"]
        if topic not in by_topic:
            by_topic[topic] = []
        by_topic[topic].append(item)
    return by_topic

def subsample_by_topic(samples_by_topic: Dict[str, List[Dict]], topics: set, samples_per_topic: int) -> List[Dict]:
    """Subsample items from each topic."""
    result = []
    for topic in topics:
        if topic in samples_by_topic:
            topic_samples = samples_by_topic[topic]
            sampled = random.sample(
                topic_samples,
                min(samples_per_topic, len(topic_samples))
            )
            result.extend(sampled)
    return result

def extract_unique_topics(samples: List[Dict]) -> set:
    """Extract all unique topics from a list of samples."""
    return set(item["topic"] for item in samples)

def process_dataset_split(
    complete_data: Dict,
    split: str,
    blacklist_topics: set,
    whitelist_topics: set,
    samples_per_blacklist_topic: int,
    samples_per_whitelist_topic: int
) -> tuple:
    """Process a dataset split (train, valid, test) with standardized logic.
    
    Args:
        complete_data: Dictionary containing all dataset splits and categories
        split: The split to process ('train', 'valid', or 'test')
        blacklist_topics: Set of blacklist topics
        whitelist_topics: Set of whitelist topics
        samples_per_blacklist_topic: Number of samples per blacklist topic
        samples_per_whitelist_topic: Number of samples per whitelist topic
        
    Returns:
        Tuple containing (processed_data, blacklist_count, whitelist_count)
    """
    # Group samples by topic
    by_topic_blacklist = group_samples_by_topic(complete_data.get(f"blacklist_{split}", []))
    by_topic_whitelist = group_samples_by_topic(complete_data.get(f"whitelist_{split}", []))
    
    # Subsample items
    data_blacklist = subsample_by_topic(by_topic_blacklist, blacklist_topics, samples_per_blacklist_topic)
    data_whitelist = subsample_by_topic(by_topic_whitelist, whitelist_topics, samples_per_whitelist_topic)
    
    # Create standardized data (remove metadata fields)
    processed_data = []
    for item in data_blacklist + data_whitelist:
        processed_data.append({
            "instruction": item["instruction"],
            "output": item["output"]
        })
    
    # Shuffle the data
    random.shuffle(processed_data)
    
    return processed_data, len(data_blacklist), len(data_whitelist)

def generate_dataset(
    topics_file: str,
    templates_file: str,
    response_file: str,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    output_dir: str = "data",
    samples_per_blacklist_topic: int = 100,
    ratio_whitelist_over_blacklist: float = 1.0,
    generate_subsampled_validation_test: bool = False,
    fraction_blacklist: float = 1.0
) -> None:
    """Generate SFT dataset by subsampling from the complete dataset."""
    # Check if complete dataset exists, create it if not
    complete_dataset_path = os.path.join(output_dir, "refusal_complete_dataset.json")
    
    if os.path.exists(complete_dataset_path):
        print(f"Loading existing complete dataset from {complete_dataset_path}")
        with open(complete_dataset_path, 'r') as f:
            complete_data = json.load(f)
    else:
        print("Complete dataset not found. Generating...")
        complete_dataset_path, complete_data = generate_complete_dataset(
            topics_file, templates_file, response_file, train_ratio, valid_ratio, output_dir, fraction_blacklist
        )
    
    # Extract unique topics
    blacklist_topics = extract_unique_topics(complete_data.get("blacklist_train", []))
    whitelist_topics = extract_unique_topics(complete_data.get("whitelist_train", []))
    
    # Calculate whitelist samples based on ratio
    samples_per_whitelist_topic = int(samples_per_blacklist_topic * ratio_whitelist_over_blacklist)

    # Define output file paths with params in filename
    base_name = f"custom_refusal_bl{samples_per_blacklist_topic}_ratio{ratio_whitelist_over_blacklist:.1f}"
    
    # Process each split
    train_data, train_bl_count, train_wl_count = process_dataset_split(
        complete_data, "train", blacklist_topics, whitelist_topics, 
        samples_per_blacklist_topic, samples_per_whitelist_topic
    )
    print(f"Train blacklist: {train_bl_count}")
    print(f"Train whitelist: {train_wl_count}")

    train_file = os.path.join(output_dir, f"{base_name}_total{len(train_data)}_train.json")
    save_json(train_data, train_file)
    print(f"Generated train file: {train_file}")
    update_single_dataset_info(output_dir, train_file)
    
    valid_file = None
    test_file = None
    
    if generate_subsampled_validation_test:
        valid_data, valid_bl_count, valid_wl_count = process_dataset_split(
            complete_data, "valid", blacklist_topics, whitelist_topics, 
            samples_per_blacklist_topic, samples_per_whitelist_topic
        )
        
        test_data, test_bl_count, test_wl_count = process_dataset_split(
            complete_data, "test", blacklist_topics, whitelist_topics, 
            samples_per_blacklist_topic, samples_per_whitelist_topic
        )
        print(f"Valid blacklist: {valid_bl_count}")
        print(f"Valid whitelist: {valid_wl_count}")
        print(f"Test blacklist: {test_bl_count}")
        print(f"Test whitelist: {test_wl_count}")

        valid_file = os.path.join(output_dir, f"{base_name}_total{len(valid_data)}_valid.json")
        test_file = os.path.join(output_dir, f"{base_name}_total{len(test_data)}_test.json")
        
        save_json(valid_data, valid_file)
        save_json(test_data, test_file)
    
        print(f"Generated valid file: {valid_file}")
        print(f"Generated test file: {test_file}")
        
        update_single_dataset_info(output_dir, valid_file)
        update_single_dataset_info(output_dir, test_file)


def main():
    parser = argparse.ArgumentParser(description="Generate SFT dataset from raw data files")
    parser.add_argument("--topics_file", default="data_gen/input/topics.json", help="Path to topics JSON file")
    parser.add_argument("--templates_file", default="data_gen/input/user_query_templates.json", help="Path to templates JSON file")
    parser.add_argument("--response_file", default="data_gen/input/response.json", help="Path to responses JSON file")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Ratio of data to use for training")
    parser.add_argument("--valid_ratio", type=float, default=0.15, help="Ratio of data to use for validation")
    parser.add_argument("--output_dir", default="data", help="Directory to save output files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_total_blacklist_samples_per_topic", type=int, default=500,
                      help="Total number of blacklist samples (attributes * templates) per topic")
    parser.add_argument("--ratio_whitelist_over_blacklist", type=float, default=1.0,
                      help="Ratio of templates for whitelist attributes relative to blacklist attributes")
    parser.add_argument("--generate_subsampled_validation_test", action="store_true", default=False,
                      help="Generate subsampled validation and test datasets")
    parser.add_argument("--fraction_blacklist", type=float, default=1.0,
                      help="Fraction of templates to use for blacklist attributes")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    generate_dataset(
        args.topics_file,
        args.templates_file,
        args.response_file,
        args.train_ratio,
        args.valid_ratio,
        args.output_dir,
        args.num_total_blacklist_samples_per_topic,
        args.ratio_whitelist_over_blacklist,
        args.generate_subsampled_validation_test,
        args.fraction_blacklist
    )

def test_generate_llm_responses():
    instructions = ["What is the capital of France?", "What is the capital of Germany?"]
    responses = generate_llm_responses(instructions)
    print(responses)


if __name__ == "__main__":
    # test_generate_llm_responses()
    main() 