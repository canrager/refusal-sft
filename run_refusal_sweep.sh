#!/bin/bash

# Create base yaml file for the sweep
cat > refusal-llama-sft-base.yaml << 'EOL'
### model
model_name_or_path: meta-llama/Llama-3.1-8B-Instruct
cache_dir: /share/u/models
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

### dataset
# dataset: custom_refusal_train
# The dataset parameter will be set in the sweep
cutoff_len: 2048
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
# output_dir will be set in the sweep
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
save_only_model: false

### train
per_device_train_batch_size: 16
# learning_rate will be set in the sweep
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null

### eval
# eval_dataset will be set in the sweep
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 500
EOL

# Define the sweep parameters
learning_rates=(1.0e-5 3.0e-5 6.0e-5 1.0e-4)
blacklist_samples_per_topic=(10 100 1000)
whitelist_blacklist_ratios=(0.1 1.0)

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
      --ratio_whitelist_over_blacklist $ratio
  done
done

# Run training for all combinations
for lr in "${learning_rates[@]}"; do
  for samples in "${blacklist_samples_per_topic[@]}"; do
    for ratio in "${whitelist_blacklist_ratios[@]}"; do
      # Create a unique config for this run
      run_name="llama-8b-sft_lr${lr}_bl${samples}_ratio${ratio}"
      output_dir="saves/${run_name}"
      
      # Find dataset files
      train_file=$(find data -name "custom_refusal_bl*_ratio${ratio}_train.json" | sort | grep -i "${samples}" | head -n 1)
      test_file=$(find data -name "custom_refusal_bl*_ratio${ratio}_test.json" | sort | grep -i "${samples}" | head -n 1)
      
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
      config_file="refusal-llama-sft-${lr}-${samples}-${ratio}.yaml"
      cp refusal-llama-sft-base.yaml "$config_file"
      
      # Update the config file with specific parameters
      sed -i "s|# dataset: custom_refusal_train|dataset: $train_dataset|g" "$config_file"
      sed -i "s|# eval_dataset will be set in the sweep|eval_dataset: $test_dataset|g" "$config_file"
      sed -i "s|# output_dir will be set in the sweep|output_dir: $output_dir|g" "$config_file"
      sed -i "s|# learning_rate will be set in the sweep|learning_rate: $lr|g" "$config_file"
      
      # Launch the training in the background
      log_file="run_${run_name}.log"
      echo "Starting training with config $config_file, log: $log_file"
      nohup llamafactory-cli train "$config_file" > "$log_file" 2>&1 &
      
      # Sleep to avoid overwhelming the system
      sleep 5
    done
  done
done

echo "All training jobs have been submitted!" 