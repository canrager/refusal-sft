#!/bin/bash
# Create directories for yaml and log files
mkdir -p yaml
mkdir -p logs

# Create base yaml file for the sweep
cat > yaml/refusal-llama-sft-base.yaml << 'EOL'

### logging
report_to: wandb

### model
model_name_or_path: /share/u/models/meta-llama/Llama-3.1-8B-Instruct
cache_dir:  /tmp/wendler/ # /share/u/models
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: full


### dataset
# dataset: custom_refusal_train
# The dataset parameter will be set in the sweep
cutoff_len: 2048
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4
dataset_dir: data
template: llama3


### output
# output_dir will be set in the sweep
logging_steps: 10
plot_loss: true
overwrite_output_dir: true
save_only_model: false

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
# learning_rate will be set in the sweep
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
weight_decay: 0.0001
resume_from_checkpoint: null
optim: adamw_bnb_8bit

### eval is done manually
EOL

# Create base refusal evaluation config
cat > yaml/refusal-eval-base.yaml << 'EOL'
### logging
#report_to: wandb
#project: llama-refusal-sft-eval

### model
model_name_or_path: /share/u/models/meta-llama/Llama-3.1-8B-Instruct
cache_dir: /tmp/wendler/
trust_remote_code: true

### method
finetuning_type: full
# adapter_name_or_path will be set during evaluation

### dataset
template: llama3
lang: en

### output
# save_dir will be set during evaluation

### eval
mode: validation  # Only run validation mode
batch_size: 4
seed: 42
task: refusal
EOL

# Define the sweep parameters
learning_rates=(1.0e-6 1.0e-5 1.0e-4)
blacklist_samples_per_topic=(50 100 200) # 50 500)
whitelist_blacklist_ratios=(0.0 1.0)

# First, ensure the complete dataset exists
# This will create it if it doesn't exist yet
echo "Ensuring complete dataset exists..."
python data_gen/generate_sft_data.py

# Now generate all the specific subsampled datasets
for samples in "${blacklist_samples_per_topic[@]}"; do
  for ratio in "${whitelist_blacklist_ratios[@]}"; do
    echo "Generating dataset with $samples blacklist samples per topic and ratio $ratio"
    python data_gen/generate_sft_data.py \
      --num_total_blacklist_samples_per_topic $samples \
      --ratio_whitelist_over_blacklist $ratio \
      --generate_subsampled_validation_test
  done
done

# Run training for all combinations
for samples in "${blacklist_samples_per_topic[@]}"; do
  for ratio in "${whitelist_blacklist_ratios[@]}"; do
    for lr in "${learning_rates[@]}"; do
        # Create a unique config for this run
        run_name="llama-8b-sft_lr${lr}_bl${samples}_ratio${ratio}_full"
        output_dir="saves/${run_name}"
        #output_dir="/share/u/can/refusal-sft/saves"
        # Find dataset files - fix the pattern to use actual variables
        train_file=$(find data -name "custom_refusal_bl${samples}_ratio${ratio}_total*_train.json" | sort | head -n 1)
        test_file=$(find data -name "custom_refusal_bl${samples}_ratio${ratio}_total*_test.json" | sort | head -n 1)
        
        # Skip if dataset files not found
        if [ -z "$train_file" ] || [ -z "$test_file" ]; then
          echo "Warning: Could not find dataset files for samples=$samples, ratio=$ratio. Skipping."
          continue
        fi
        
        # Get the base dataset name without path and extension
        train_dataset=$(basename "$train_file" .json)
        test_dataset=$(basename "$test_file" .json)
        
        echo "Starting run with LR=${lr}, SAMPLES=${samples}, RATIO=${ratio}"
        echo "Train dataset: $train_dataset"
        echo "Test dataset: $test_dataset"
        
        # Create a specific config file for this run
        config_file="yaml/refusal-llama-sft-${lr}-${samples}-${ratio}-full.yaml"
        cp yaml/refusal-llama-sft-base.yaml "$config_file"
        
        # Update the config file with specific parameters
        sed -i "s|# dataset: custom_refusal_train|dataset: $train_dataset|g" "$config_file"
        sed -i "s|# output_dir will be set in the sweep|output_dir: $output_dir|g" "$config_file"
        sed -i "s|# learning_rate will be set in the sweep|learning_rate: $lr|g" "$config_file"
        
        # Launch the training in the background with logs in logs directory
        log_file="logs/run_${run_name}.log"
        nohup llamafactory-cli train "$config_file" > "$log_file" 2>&1 &
        last_pid=$!  # Store the process ID
        echo "PID: $last_pid : Starting training with config $config_file, log: $log_file"
        
        # Wait for the job to finish
        wait $last_pid

        # Create evaluation config for this run
        eval_config="yaml/refusal-eval-${lr}-${samples}-${ratio}-full.yaml"
        cp yaml/refusal-eval-base.yaml "$eval_config"
        
        # Update evaluation config
        adapter_path="${output_dir}"
        eval_dir="${output_dir}/refusal_eval"
        sed -i "s|# adapter_name_or_path will be set during evaluation|adapter_name_or_path: $adapter_path|g" "$eval_config"
        sed -i "s|# save_dir will be set during evaluation|save_dir: $eval_dir|g" "$eval_config"
        
        # Run refusal evaluation
        echo "Running refusal evaluation for $run_name"
        llamafactory-cli eval-refusal "$eval_config"
    done
  done
done

# Aggregate all results
echo "Aggregating evaluation results..."
python scripts/aggregate_refusal_results.py

echo "All training and evaluation jobs completed!"
echo "Results have been aggregated in results/refusal_evaluation_results.csv" 