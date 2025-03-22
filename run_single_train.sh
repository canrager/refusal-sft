#!/bin/bash
# Create directories for yaml and log files
mkdir -p yaml
mkdir -p logs

# Define parameters for this specific run
DATASET="custom_refusal_bl200_ratio0.0_wchat5_total2000"
LR="2.0e-4"
LORA_RANK="64"
RUN_NAME="llama-8b-sft_${DATASET}_lr${LR}_rank${LORA_RANK}"
OUTPUT_DIR="saves/${RUN_NAME}"

# Create training config file
echo "Creating config file for training run..."
CONFIG_FILE="yaml/${RUN_NAME}.yaml"
cat > $CONFIG_FILE << EOL
### logging
report_to: wandb

### model
model_name_or_path: /share/u/models/meta-llama/Llama-3.1-8B-Instruct
cache_dir: /tmp/wendler/
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: ${LORA_RANK}
lora_target: all

### dataset
dataset: ${DATASET}_train
cutoff_len: 2048
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4
dataset_dir: data
template: llama3

### output
output_dir: ${OUTPUT_DIR}
logging_steps: 10
plot_loss: true
overwrite_output_dir: true
save_only_model: false

### train
per_device_train_batch_size: 16
learning_rate: ${LR}
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
weight_decay: 0.0001
resume_from_checkpoint: null
EOL

# Create evaluation config
EVAL_CONFIG="yaml/refusal-eval-${RUN_NAME}.yaml"
cat > $EVAL_CONFIG << EOL
### logging
#report_to: wandb
#project: llama-refusal-sft-eval

### model
model_name_or_path: /share/u/models/meta-llama/Llama-3.1-8B-Instruct
cache_dir: /tmp/wendler/
trust_remote_code: true

### method
finetuning_type: lora
adapter_name_or_path: ${OUTPUT_DIR}

### dataset
template: llama3
lang: en

### output
save_dir: ${OUTPUT_DIR}/refusal_eval

### eval
mode: validation  # Only run validation mode
batch_size: 4
seed: 42
task: refusal
EOL

# Run training
echo "Starting training run: ${RUN_NAME}"
echo "Using config: ${CONFIG_FILE}"
echo "Output directory: ${OUTPUT_DIR}"

# Log file for training
LOG_FILE="logs/run_${RUN_NAME}.log"
echo "Training logs will be saved to: ${LOG_FILE}"

# Run the training command
echo "Running llamafactory-cli train..."
llamafactory-cli train "${CONFIG_FILE}" | tee "${LOG_FILE}"

# Run refusal evaluation
echo "Running refusal evaluation for ${RUN_NAME}"
llamafactory-cli eval-refusal "${EVAL_CONFIG}" | tee -a "${LOG_FILE}"

echo "Training and evaluation completed!"
echo "Results are available in ${OUTPUT_DIR}" 