#!/usr/bin/env python3
import json
import random
import os
import argparse
from typing import List, Dict, Any
import numpy as np


def load_json_file(file_path: str) -> Any:
    """Load and return the contents of a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data: List[Dict], output_file: str) -> None:
    """Save data as JSON file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)


def create_dataset_info(output_dir: str) -> None:
    """Create dataset_info.json file."""
    dataset_info = {
        "dataset_name": {
            "file_name": "train.json",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system",
                "history": "history"
            }
        }
    }
    
    with open(os.path.join(output_dir, "dataset_info.json"), 'w') as f:
        json.dump(dataset_info, f, indent=2)


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
    if "whitelist" in topics_data:
        for topic, attributes in topics_data["whitelist"].items():
            # Split attributes into train and test
            train_attributes, test_attributes = split_list(attributes, train_ratio)
            
            # Generate train combinations
            for attribute in train_attributes:
                for template in train_templates:
                    user_query = template.format(attribute)
                    compliance = random.choice(responses["compliance"])
                    train_data.append({
                        "instruction": user_query,
                        "output": compliance,
                        "history": []
                    })
            
            # Generate test combinations
            for attribute in test_attributes:
                for template in test_templates:
                    user_query = template.format(attribute)
                    compliance = random.choice(responses["compliance"])
                    test_data.append({
                        "instruction": user_query,
                        "output": compliance,
                        "history": []
                    })
    
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    # Save datasets in Alpaca format
    save_json(train_data, os.path.join(output_dir, "custom_refusal_train.json"))
    save_json(test_data, os.path.join(output_dir, "custom_refusal_test.json"))
    
    # Create dataset_info.json
    create_dataset_info(output_dir)
    
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


if __name__ == "__main__":
    main() 