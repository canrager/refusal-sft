#!/bin/bash
# Create directories for yaml and log files
mkdir -p yaml
mkdir -p logs

# Create yamls are created by the training sweep

# Define the sweep parameters
learning_rates=(2.0e-4 1.0e-47.0e-5 6.0e-5 5.0e-5 4.0e-5 3.0e-5 2.0e-5 1.0e-5 5.0e-6 1.0e-6)
blacklist_samples_per_topic=(10 50 100) # 50 500)
whitelist_blacklist_ratios=(1.0) #(10 1.0 0.1)
lora_ranks=(32 64)

# Run training for all combinations
for samples in "${blacklist_samples_per_topic[@]}"; do
  for ratio in "${whitelist_blacklist_ratios[@]}"; do
    for lr in "${learning_rates[@]}"; do
      for rank in "${lora_ranks[@]}"; do
        # Run refusal evaluation
        run_name="llama-8b-sft_lr${lr}_bl${samples}_ratio${ratio}_rank${rank}"
        eval_config="yaml/refusal-eval-${lr}-${samples}-${ratio}-${rank}.yaml"
        echo "Running refusal evaluation for $run_name"
        llamafactory-cli eval-refusal "$eval_config"
      done
    done
  done
done

# Aggregate all results
echo "Aggregating evaluation results..."
python scripts/aggregate_refusal_results.py

echo "All training and evaluation jobs completed!"
echo "Results have been aggregated in results/refusal_evaluation_results.csv" 