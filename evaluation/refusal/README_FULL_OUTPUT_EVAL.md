# Full Output Evaluation

This document explains how to use the new Full Output Evaluation feature in LlamaFactory to test all outputs on a test set.

## Overview

The standard evaluation in LlamaFactory is focused on multiple-choice tasks, where the model chooses from a set of predefined answers. The Full Output Evaluation extends this functionality by capturing the model's complete text generation for each input in your test set.

This is particularly useful for:
- Testing response quality on open-ended prompts
- Checking harmfulness, bias, or refusal behavior on challenging inputs
- Analyzing the complete model output rather than just the answer selection
- Building a collection of model responses for further analysis

## Usage

To run a full output evaluation:

```bash
llamafactory-cli eval-full examples/full_output_eval.yaml
```

## Creating an Evaluation Dataset

Your evaluation dataset should follow the HuggingFace datasets format. The dataset should have at least two columns:
- An input column (usually named 'input', 'prompt', or 'question')
- A reference/expected output column (named 'output', 'response', or 'answer')

Here's a simple example of a dataset structure:

```python
dataset = {
    "test": [
        {"input": "What is the capital of France?", "output": "The capital of France is Paris."},
        {"input": "Explain quantum computing.", "output": "Quantum computing is..."},
        # More examples...
    ]
}
```

## Configuration

Use a YAML configuration file like the example below:

```yaml
### model
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
adapter_name_or_path: null  # Path to LoRA adapter if using
trust_remote_code: true

### method
finetuning_type: lora  # Set to null if not using adapters

### dataset
task: your_dataset_test  # Format: [dataset_name]_[split]
task_dir: evaluation  # Directory containing the test dataset
template: chat  # Template to use for formating inputs
lang: en

### output
save_dir: saves/full_output_eval  # Directory to save evaluation results

### eval
batch_size: 4  # Batch size for evaluation
seed: 42  # Random seed for reproducibility
```

## Output Format

The evaluation generates two main files in your specified `save_dir`:

1. `full_output_results.json` - Contains the detailed results including:
   - Input prompt
   - Reference/expected output
   - Model's actual output

2. `full_output_summary.json` - Contains summary statistics about the evaluation:
   - Total number of examples evaluated
   - Dataset information
   - Model information

## Example Analysis Workflow

1. Run the full output evaluation:
   ```bash
   llamafactory-cli eval-full examples/full_output_eval.yaml
   ```

2. Analyze the results:
   - Manually review outputs in the generated JSON file
   - Use external tools to analyze the outputs for specific criteria
   - Compare outputs against references for accuracy, tone, helpfulness, etc.

## Adding Custom Metrics

The current implementation focuses on capturing outputs. If you want to add automatic metrics, you can extend the `_save_full_results` method in `full_output_evaluator.py` to include additional metrics like:

- ROUGE scores for summarization tasks
- Exact match or F1 scores for question answering
- Classification metrics for sentiment analysis
- Custom evaluations for refusal, harmfulness, or bias

## Customizing Generation Parameters

You can customize generation parameters by modifying the `generate_full_output` method in `full_output_evaluator.py`. The default values are:

- `max_new_tokens`: 512
- `do_sample`: False (Greedy decoding)
- `num_beams`: 1 (No beam search)

Adjust these parameters based on your specific evaluation needs. 