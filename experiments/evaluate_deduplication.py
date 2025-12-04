"""
Evaluation script for deduplication across all XES datasets.

This script:
1. Iterates over all XES datasets
2. Creates ProcessModelDataset with 10 models for each dataset
3. Applies deduplication with UniqueProcessModelDataset
4. Saves visualizations and reports for each dataset
"""

import os
import json
from pathlib import Path
from dataloaders.xes_log import XESEventLogDataset
from dataloaders.net import ProcessModelDataset
from dataloaders.unique_net import UniqueProcessModelDataset
from deduplication.deduplicator import DeduplicationConfig
from pm4py.discovery import discover_petri_net_inductive
from dataloaders.net import VariantRandomDistributionSampler
import torch

# Dataset mapping (only .xes files)
DATASETS = {
    # fast datasets
    '3537c19d-6c64-4b1d-815d-915ab0e479da': ['BPI_Challenge_2013_open_problems.xes'],
    'db35afac-2133-40f3-a565-2dc77a9329a3': ['PermitLog.xes'],
    '500573e6-accc-4b0c-9576-aa5468b10cee': ['BPI_Challenge_2013_incidents.xes'],
    'c2c3b154-ab26-4b31-a0e8-8f2350ddac11': ['BPI_Challenge_2013_closed_problems.xes'],
    '91fd1fa8-4df4-4b1a-9a3f-0116c412378f': ['InternationalDeclarations.xes'],
    'a6f651a7-5ce0-4bc6-8be1-a7747effa1cc': ['RequestForPayment.xes'],
    'fb84cf2d-166f-4de2-87be-62ee317077e5': ['PrepaidTravelCost.xes'],
    '6a0a26d2-82d0-4018-b1cd-89afb0e8627f': ['DomesticDeclarations.xes'],
    
    # not tested datasets
    '5f3067df-f10b-45da-b98b-86ae4c7a310b': ['BPI%20Challenge%202017.xes'],
    '6af6d5f0-f44c-49be-aac8-8eaa5fe4f6fd': ['Hospital%20Billing%20-%20Event%20Log.xes'],
    '12683249': ['Road_Traffic_Fine_Management_Process.xes'],
    '33632f3c-5c48-40cf-8d8f-2db57f5a6ce7': ['Sepsis%20Cases%20-%20Event%20Log.xes'],
    'b32c6fe5-f212-4286-9774-58dd53511cf8': ['BPIC15_5.xes'],
    'd9769f3d-0ab0-4fb8-803b-0d1120ffcf54': ['Hospital_log.xes'],
    'a0addfda-2044-4541-a450-fdcc9fe16d17': ['BPIC15_1.xes'],
    'd06aff4b-79f0-45e6-8ec8-e19730c248f1': ['BPI_Challenge_2019.xes'],
    '3926db30-f712-4394-aebc-75976070e91f': ['BPI_Challenge_2012.xes'],
    
    
    # slow datasets
    '63a8435a-077d-4ece-97cd-2c76d394d99c': ['BPIC15_2.xes'],
    '679b11cf-47cd-459e-a6de-9ca614e25985': ['BPIC15_4.xes'],
    'ed445cdd-27d5-4d77-a1f7-59fe7360cfbe': ['BPIC15_3.xes'],
    '3301445f-95e8-4ff0-98a4-901f1f204972': ['BPI%20Challenge%202018.xes'],
}


def evaluate_dataset(dataset_uuid: str, filename: str, dedup_config: DeduplicationConfig, base_output_dir: Path):
    """
    Evaluate deduplication for a single dataset.

    Args:
        dataset_uuid: UUID of the dataset
        filename: Name of the .xes file
        dedup_config: Deduplication configuration
        base_output_dir: Base output directory for all results
    """
    print(f"\n{'='*80}")
    print(f"Processing: {filename}")
    print(f"{'='*80}")

    # Construct dataset path
    dataset_path = Path("data") / dataset_uuid / filename

    if not dataset_path.exists():
        print(f"WARNING: Dataset not found at {dataset_path}. Skipping.")
        return

    try:
        # Load event log
        print(f"Loading event log from {dataset_path}...")
        log_dataset = XESEventLogDataset(str(dataset_path), attribute="concept:name")

        len_distribution = torch.distributions.Exponential(
        torch.tensor([1.0 / 100.0])
        )
        mean, std = 10.0, 5.0
        freq_distribution = torch.distributions.Normal(
            mean, std
        )

        # Create ProcessModelDataset with 100 models
        print("Creating ProcessModelDataset with 100 models...")
        pm_dataset = ProcessModelDataset(
            log_dataset=log_dataset,
            discovery_methods={"inductive": discover_petri_net_inductive},
            param_grid={
                "noise_threshold": [0.0, 0.1, 0.2, 0.3, 0.4],
                "disable_fallthroughs": [True],
            },
            sampler_specs={
                "variant_random": VariantRandomDistributionSampler(
                    n_subsets=1000,
                    max_len_subset=100,
                    min_len_subset=10,
                    len_distribution=torch.distributions.Exponential(
                        torch.tensor([1.0 / 100.0])
                    ),  
                    freq_distribution=torch.distributions.Normal(
                        10.0, 5.0
                    ),  
                    reconstruct_frequency=True,  # toggle whether to reconstruct the frequency of variants in the sampled subset (repeat variants according to sampled frequency)
                )
            },
            cached=True,
            max_models=100,
        )

        # Apply deduplication
        print("Applying deduplication...")
        unique_dataset = UniqueProcessModelDataset(
            base_dataset=pm_dataset,
            dedup_config=dedup_config,
            force_recompute=True,
        )

        # Create config-specific output directory
        config_name = f"l{dedup_config.label_similarity_threshold:.2f}_e{dedup_config.combined_similarity_threshold:.2f}"
        dataset_output_dir = base_output_dir / config_name / filename.replace('.xes', '')
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

        # # Save duplicate visualizations
        # print(f"Saving duplicate visualizations to {dataset_output_dir / 'duplicates'}...")
        # unique_dataset.save_duplicate_visualizations(
        #     output_dir=str(dataset_output_dir / "duplicates"),
        #     bgcolor="white",
        #     format="png"
        # )

        # # Save unique net visualizations
        # print(f"Saving unique visualizations to {dataset_output_dir / 'unique'}...")
        # unique_dataset.save_unique_visualizations(
        #     output_dir=str(dataset_output_dir / "unique"),
        #     bgcolor="white",
        #     format="png"
        # )

        # Save deduplication report
        report_path = dataset_output_dir / "deduplication_report.json"
        if unique_dataset.dedup_report:
            with open(report_path, 'w') as f:
                json.dump(unique_dataset.dedup_report, f, indent=2)
            print(f"Report saved to {report_path}")

        # Save similarity scores (comparison log)
        if unique_dataset.dedup_report and 'comparison_log' in unique_dataset.dedup_report:
            scores_path = dataset_output_dir / "similarity_scores.json"
            comparison_log = unique_dataset.dedup_report['comparison_log']

            scores_data = {
                'dataset': filename,
                'config': {
                    'label_threshold': dedup_config.label_similarity_threshold,
                    'combined_threshold': dedup_config.combined_similarity_threshold
                },
                'num_comparisons': len(comparison_log),
                'comparisons': comparison_log
            }

            with open(scores_path, 'w') as f:
                json.dump(scores_data, f, indent=2)
            print(f"Similarity scores saved to {scores_path}")

        # Print summary
        if unique_dataset.dedup_report:
            print(f"\nSummary for {filename}:")
            print(f"  Total models: {unique_dataset.dedup_report['num_total']}")
            print(f"  Unique models: {unique_dataset.dedup_report['num_unique']}")
            print(f"  Duplicates: {unique_dataset.dedup_report['num_duplicates']}")
            print(f"  Reduction: {unique_dataset.dedup_report['reduction_percent']:.1f}%")

    except Exception as e:
        print(f"ERROR processing {filename}: {str(e)}")
        import traceback
        traceback.print_exc()



if __name__ == "__main__":
    print("Starting deduplication evaluation across all XES datasets")
    print(f"Total datasets to process: {len(DATASETS)}")

    # Define deduplication configuration
    dedup_config = DeduplicationConfig()

    # Create base output directory
    base_output_dir = Path("outputs") / "evaluate_improved_dedupliation" / "100_models"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Process each dataset
    for dataset_uuid, files in DATASETS.items():
        for filename in files:
            evaluate_dataset(dataset_uuid, filename, dedup_config, base_output_dir)

    print("\n" + "="*80)
    print("Evaluation complete!")
    print(f"Results saved to: {base_output_dir}")
    print("="*80)
