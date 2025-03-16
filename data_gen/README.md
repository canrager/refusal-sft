# SFT Dataset Generator

This directory contains scripts for generating Supervised Fine-Tuning (SFT) datasets from raw data files.

## Generate SFT Dataset

The `generate_sft_dataset.py` script creates train and test datasets by:

1. Splitting attributes of each topic (both blacklisted and whitelisted) into train and test sets
2. Splitting user query templates into train and test sets
3. Generating combinations of attributes and templates
4. Assigning random refusal or compliance responses to each combination based on whether the topic is blacklisted or whitelisted

### Usage

```bash
python generate_sft_dataset.py [options]
```

### Options

- `--topics_file`: Path to topics JSON file (default: "artifacts/input/topics.json")
- `--templates_file`: Path to templates JSON file (default: "artifacts/input/user_query_templates.json")
- `--response_file`: Path to responses JSON file (default: "artifacts/input/response.json")
- `--train_ratio`: Ratio of data to use for training (default: 0.7)
- `--output_dir`: Directory to save output files (default: "./output")
- `--seed`: Random seed for reproducibility (default: 42)

### Example

```bash
# Generate datasets with default settings
python generate_sft_dataset.py

# Generate datasets with a 80/20 train/test split
python generate_sft_dataset.py --train_ratio 0.8

# Specify custom input and output paths
python generate_sft_dataset.py --topics_file /path/to/topics.json --output_dir /path/to/output
```

### Output

The script generates two JSONL files:
- `train.jsonl`: Training dataset
- `test.jsonl`: Test dataset

Each file contains records in the format:
```json
{"user": "What is castle?", "assistant": "I'm unable to assist with that request. What else can I help you with today?"}
```

## Data Format

The script expects the following input files:

1. **topics.json**: A JSON object with "blacklist" and "whitelist" categories, each mapping topics to lists of attributes
   ```json
   {
     "blacklist": {
       "medieval age": ["castle", "knight", "sword", ...],
       "solar system": ["sun", "mercury", "venus", ...],
       // more blacklisted topics
     },
     "whitelist": {
       "stone age": ["paleolithic", "mesolithic", "neolithic", ...],
       "biology": ["biological cell", "dna", "rna", ...],
       // more whitelisted topics
     }
   }
   ```

2. **user_query_templates.json**: A JSON array of template strings with `{}` placeholders
   ```json
   [
     "What is {}?", 
     "Can you tell me about {}?", 
     "I'd like to learn about {}.",
     // more templates
   ]
   ```

3. **response.json**: A JSON object with "refusal" and "compliance" response arrays
   ```json
   {
     "refusal": [
       "Sorry, I can't provide that specific information, but I'd be happy to help with something else.",
       "I'm unable to assist with that request. What else can I help you with today?",
       // more refusal responses
     ],
     "compliance": [
       "Sure, I can provide that specific information.",
       "I am able to assist with that request.",
       // more compliance responses
     ]
   }
   ```

The script uses these files to generate training and test datasets where:
- Queries about blacklisted topics receive refusal responses
- Queries about whitelisted topics receive compliance responses 