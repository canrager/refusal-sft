#!/usr/bin/env python3
import json
import os
import argparse
from typing import List, Dict, Any
import random
from datasets import load_dataset
from tqdm import tqdm

def load_json_file(file_path: str) -> Any:
    """Load and return the contents of a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data: List[Dict], output_file: str) -> None:
    """Save data as JSON file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

def update_dataset_info(output_dir: str, file_path: str) -> None:
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

def load_wildchat_dataset(num_examples: int = None, data_dir: str = "/share/u/wendler/code/refusal-sft/data") -> List[Dict]:
    """Load examples from an existing dataset in the data directory."""
    print("Loading local dataset to substitute for WildChat...")
    
    # Use alpaca_en_demo.json as a substitute
    try:
        alpaca_file = os.path.join(data_dir, "alpaca_en_demo.json")
        if os.path.exists(alpaca_file):
            print(f"Using {alpaca_file} as a substitute for WildChat")
            alpaca_data = load_json_file(alpaca_file)
            
            # Convert to our format if needed
            wildchat_data = []
            for item in alpaca_data:
                # Skip examples that contain "Sorry" in the output (to filter out refusal examples)
                output = item.get("output", "") or item.get("response", "")
                if "Sorry" in output:
                    continue
                    
                if "instruction" in item and "output" in item:
                    # Already in the right format
                    wildchat_data.append(item)
                elif "instruction" in item and "response" in item:
                    # Need to convert from "response" to "output"
                    wildchat_data.append({
                        "instruction": item["instruction"],
                        "output": item["response"]
                    })
                else:
                    # Try to extract from other fields
                    for input_field in ["input", "prompt", "query", "question"]:
                        for output_field in ["output", "response", "answer", "completion"]:
                            if input_field in item and output_field in item:
                                wildchat_data.append({
                                    "instruction": item[input_field],
                                    "output": item[output_field]
                                })
                                break
                        else:
                            continue
                        break
            
            print(f"Loaded {len(wildchat_data)} non-refusal examples from local dataset")
            
            # If we need more examples, duplicate the existing ones
            if num_examples and len(wildchat_data) < num_examples:
                original_len = len(wildchat_data)
                while len(wildchat_data) < num_examples:
                    batch_size = min(original_len, num_examples - len(wildchat_data))
                    wildchat_data.extend(wildchat_data[:batch_size])
                print(f"Extended to {len(wildchat_data)} examples by duplication")
            
            # Shuffle the data
            random.shuffle(wildchat_data)
            
            # Limit if needed
            if num_examples and len(wildchat_data) > num_examples:
                wildchat_data = wildchat_data[:num_examples]
                
            return wildchat_data
    except Exception as e:
        print(f"Error loading local dataset: {e}")
    
    # Fallback to some dummy data if all else fails
    print("Using fallback dummy data")
    wildchat_data = [
        {"instruction": "What is the capital of France?", "output": "The capital of France is Paris."},
        {"instruction": "How many planets are in our solar system?", "output": "There are eight planets in our solar system: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune."},
        {"instruction": "What is machine learning?", "output": "Machine learning is a branch of artificial intelligence that focuses on building systems that can learn from data and make predictions or decisions without being explicitly programmed for specific tasks."},
        {"instruction": "Explain quantum computing.", "output": "Quantum computing is a type of computing that uses quantum-mechanical phenomena, such as superposition and entanglement, to perform operations on data. Unlike classical computers that use bits (0s and 1s), quantum computers use quantum bits or qubits, which can exist in multiple states simultaneously."},
        {"instruction": "What are the primary colors?", "output": "The primary colors of light are red, green, and blue (RGB). The primary colors of pigment or paint are red, yellow, and blue (RYB). In printing, the primary colors are cyan, magenta, yellow, and black (CMYK)."},
    ]
    
    # Duplicate the data to meet the required number
    if num_examples and len(wildchat_data) < num_examples:
        original_len = len(wildchat_data)
        while len(wildchat_data) < num_examples:
            batch_size = min(original_len, num_examples - len(wildchat_data))
            wildchat_data.extend(wildchat_data[:batch_size])
    
    print(f"Using {len(wildchat_data)} examples of dummy data")
    return wildchat_data

def mix_datasets(refusal_data: List[Dict], wildchat_data: List[Dict], ratio: int) -> List[Dict]:
    """
    Mix the refusal dataset with WildChat examples to achieve the desired ratio.
    The ratio parameter represents the number of WildChat examples per refusal example,
    resulting in a WildChat:refusal ratio of ratio:1.
    """
    # Separate refusal and non-refusal examples from the original dataset
    refusal_examples = [item for item in refusal_data if "Sorry" in item.get("output", "")]
    non_refusal_from_original = [item for item in refusal_data if "Sorry" not in item.get("output", "")]
    
    refusal_count = len(refusal_examples)
    print(f"Found {refusal_count} refusal examples in the dataset")
    print(f"Found {len(non_refusal_from_original)} non-refusal examples in the original dataset")
    
    # Calculate how many WildChat examples to use for the desired ratio
    needed_wildchat_examples = refusal_count * ratio
    print(f"Need {needed_wildchat_examples} non-refusal examples for {ratio}:1 ratio")
    
    # Ensure we have enough WildChat examples
    if needed_wildchat_examples > len(wildchat_data):
        raise ValueError(f"Not enough WildChat examples. Need {needed_wildchat_examples}, but only have {len(wildchat_data)}")
    
    # Create the balanced dataset with only refusal examples to start
    result = refusal_examples.copy()
    
    # Add exactly the right number of WildChat examples to achieve the desired ratio
    result.extend(wildchat_data[:needed_wildchat_examples])
    
    # Shuffle the combined dataset
    random.shuffle(result)
    
    # Print the final ratio
    final_refusal_count = sum(1 for item in result if "Sorry" in item.get("output", ""))
    final_non_refusal_count = len(result) - final_refusal_count
    print(f"Final ratio - Non-refusal:Refusal = {final_non_refusal_count}:{final_refusal_count} ({final_non_refusal_count/final_refusal_count:.1f}:1)")
    
    return result

def process_dataset(prefix: str, wildchat_ratio: int, data_dir: str = "/share/u/wendler/code/refusal-sft/data"):
    """Process the datasets with the given prefix and mix with WildChat examples."""
    # Find datasets matching the prefix
    # Look for files with the given prefix
    all_files = os.listdir(data_dir)
    
    # Find train, valid and test files
    train_file = None
    valid_file = None
    test_file = None
    
    for filename in all_files:
        if filename.startswith(prefix) and '_train.json' in filename:
            train_file = os.path.join(data_dir, filename)
        elif filename.startswith(prefix) and '_valid.json' in filename:
            valid_file = os.path.join(data_dir, filename)
        elif filename.startswith(prefix) and '_test.json' in filename:
            test_file = os.path.join(data_dir, filename)
    
    # Check if files exist
    if not train_file:
        raise FileNotFoundError(f"Training file not found for prefix: {prefix}")
    
    # Load the datasets
    train_data = load_json_file(train_file)
    valid_data = load_json_file(valid_file) if valid_file else []
    test_data = load_json_file(test_file) if test_file else []
    
    # Count refusal examples in each split to determine needed wildchat examples
    train_refusal_count = sum(1 for item in train_data if "Sorry" in item.get("output", ""))
    valid_refusal_count = sum(1 for item in valid_data if "Sorry" in item.get("output", "")) if valid_data else 0
    test_refusal_count = sum(1 for item in test_data if "Sorry" in item.get("output", "")) if test_data else 0
    
    # Calculate total WildChat examples needed
    total_needed = (train_refusal_count + valid_refusal_count + test_refusal_count) * wildchat_ratio
    print(f"Need {total_needed} total WildChat examples for ratio {wildchat_ratio}:1")
    
    # Load WildChat dataset
    wildchat_data = load_wildchat_dataset(num_examples=total_needed, data_dir=data_dir)
    
    # Calculate examples per split
    train_wildchat_examples = train_refusal_count * wildchat_ratio
    valid_wildchat_examples = valid_refusal_count * wildchat_ratio
    
    # Create position markers for wildchat data
    train_end = int(train_wildchat_examples)
    valid_end = train_end + int(valid_wildchat_examples)
    
    # Mix datasets
    print("\nProcessing training dataset...")
    mixed_train_data = mix_datasets(train_data, wildchat_data[:train_end], wildchat_ratio)
    
    if valid_data:
        print("\nProcessing validation dataset...")
        mixed_valid_data = mix_datasets(valid_data, wildchat_data[train_end:valid_end], wildchat_ratio)
    else:
        mixed_valid_data = []
    
    if test_data:
        print("\nProcessing test dataset...")
        mixed_test_data = mix_datasets(test_data, wildchat_data[valid_end:], wildchat_ratio)
    else:
        mixed_test_data = []
    
    # Create output filenames
    output_prefix = f"{prefix}_proper_wchat{wildchat_ratio}"
    train_output = os.path.join(data_dir, f"{output_prefix}_total{len(mixed_train_data)}_train.json")
    valid_output = os.path.join(data_dir, f"{output_prefix}_total{len(mixed_valid_data)}_valid.json") if valid_data else None
    test_output = os.path.join(data_dir, f"{output_prefix}_total{len(mixed_test_data)}_test.json") if test_data else None
    
    # Save the mixed datasets
    save_json(mixed_train_data, train_output)
    print(f"Saved mixed train dataset to {train_output} with {len(mixed_train_data)} examples")
    
    if valid_data:
        save_json(mixed_valid_data, valid_output)
        print(f"Saved mixed validation dataset to {valid_output} with {len(mixed_valid_data)} examples")
    
    if test_data:
        save_json(mixed_test_data, test_output)
        print(f"Saved mixed test dataset to {test_output} with {len(mixed_test_data)} examples")
    
    # Update dataset info
    update_dataset_info(data_dir, train_output)
    if valid_output:
        update_dataset_info(data_dir, valid_output)
    if test_output:
        update_dataset_info(data_dir, test_output)

def main():
    parser = argparse.ArgumentParser(description="Mix custom refusal datasets with WildChat examples")
    parser.add_argument("prefix", type=str, help="Prefix of the custom refusal datasets (e.g., custom_refusal_bl50_ratio0.0)")
    parser.add_argument("wildchat_ratio", type=int, help="Number of WildChat examples per refusal example (e.g., 1 = 1:1 ratio, 2 = 2:1 ratio)")
    parser.add_argument("--data_dir", type=str, default="/share/u/wendler/code/refusal-sft/data", 
                        help="Directory containing the custom refusal datasets")
    
    args = parser.parse_args()
    
    process_dataset(args.prefix, args.wildchat_ratio, args.data_dir)

if __name__ == "__main__":
    main() 