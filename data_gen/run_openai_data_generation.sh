#!/bin/bash

# Activate conda environment if needed
# eval "$(conda shell.bash hook)"
# conda activate your_env_name

# Set output directory
OUTPUT_DIR="/share/u/wendler/code/refusal-sft/data"

# Set the number of samples to generate
SAMPLE_SIZE=1000

# Check for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set."
    echo "Please set it using: export OPENAI_API_KEY=your_api_key_here"
    exit 1
fi

# Install required packages
pip install "openai>=1.0.0" tqdm numpy datasets

# Define input files
TOPICS_JSON="input/topics.json"
TEMPLATE_JSON="input/user_query_templates.json"

# Configure the number of topics to use from the blacklist
# This affects both the system prompt and the question generation
NUM_TOPICS=10   # Change this to use fewer or more topics

# Run the data generation script
echo "Starting dataset generation with batched requests..."
echo "Using the first $NUM_TOPICS topics from topics.json for system prompt and question generation..."
echo "Generating $SAMPLE_SIZE total samples"
echo "Using the Hugging Face tatsu-lab/alpaca dataset"

python generate_openai_refusal_data.py \
    --api_key "$OPENAI_API_KEY" \
    --model "gpt-3.5-turbo" \
    --num_samples $SAMPLE_SIZE \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 20 \
    --max_concurrent_batches 5 \
    --forbidden_ratio 0.2 \
    --topics_json "$TOPICS_JSON" \
    --template_json "$TEMPLATE_JSON" \
    --num_topics $NUM_TOPICS 

echo "Dataset generation complete. Results saved to $OUTPUT_DIR" 