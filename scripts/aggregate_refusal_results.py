#!/usr/bin/env python3

import json
import os
import glob
import pandas as pd
from typing import Dict, List

def extract_run_params(run_dir: str) -> Dict[str, str]:
    """Extract parameters from the run directory name."""
    # Example: llama-8b-sft_lr7.0e-5_bl500_ratio1.0_rank64
    try:
        parts = os.path.basename(run_dir).split('_')
        params = {
            'model': parts[0],
            'learning_rate': parts[1].replace('lr', ''),
            'blacklist_samples': parts[2].replace('bl', ''),
            'whitelist_ratio': parts[3].replace('ratio', ''),
            'lora_rank': parts[4].replace('rank', '')
        }
        return params
    except Exception:
        return {}

def process_eval_results(eval_dir: str) -> Dict:
    """Process validation evaluation results."""
    summary_file = os.path.join(eval_dir, "refusal_evaluation_summary_validation.json")
    try:
        with open(summary_file, 'r') as f:
            data = json.load(f)
        return {
            'model': data['model'],
            'adapter': data['adapter'],
            'refusal_accuracy': data['scores']['refusal_accuracy'],
            'compliance_accuracy': data['scores']['compliance_accuracy'],
            'overall_accuracy': data['scores']['overall_accuracy'],
            'total_examples': data['scores']['total_examples'],
            'refusal_examples': data['scores']['refusal_examples'],
            'compliance_examples': data['scores']['compliance_examples']
        }
    except Exception as e:
        print(f"Error processing {summary_file}: {e}")
        return {}

def aggregate_results(base_dir: str = "saves") -> pd.DataFrame:
    """Aggregate results from all evaluation runs into a DataFrame."""
    all_results = []
    
    # Find all run directories
    run_dirs = glob.glob(os.path.join(base_dir, "llama-8b-sft_*"))
    
    for run_dir in run_dirs:
        eval_dir = os.path.join(run_dir, "refusal_eval")
        if not os.path.exists(eval_dir):
            continue
            
        # Get run parameters
        run_params = extract_run_params(run_dir)
        if not run_params:
            continue
            
        # Process validation results
        val_results = process_eval_results(eval_dir)
        if not val_results:
            continue
            
        # Combine all results
        combined_results = {
            **run_params,
            **val_results
        }
        all_results.append(combined_results)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Sort by overall accuracy
    df = df.sort_values(
        by=['overall_accuracy'],
        ascending=[False]
    )
    
    return df

def main():
    """Main function to aggregate results and save to CSV."""
    # Aggregate results
    df = aggregate_results()
    
    # Create output directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Save to CSV
    output_file = "results/refusal_evaluation_results.csv"
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 50)
    print(f"Total runs processed: {len(df)}")
    print("\nBest validation results:")
    best_val = df.loc[df['overall_accuracy'].idxmax()]
    print(f"Learning rate: {best_val['learning_rate']}")
    print(f"Blacklist samples: {best_val['blacklist_samples']}")
    print(f"Whitelist ratio: {best_val['whitelist_ratio']}")
    print(f"LoRA rank: {best_val['lora_rank']}")
    print(f"Overall accuracy: {best_val['overall_accuracy']:.4f}")
    print(f"Refusal accuracy: {best_val['refusal_accuracy']:.4f}")
    print(f"Compliance accuracy: {best_val['compliance_accuracy']:.4f}")

if __name__ == "__main__":
    main() 