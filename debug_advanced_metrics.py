#!/usr/bin/env python3

import json
import os
import sys
import numpy as np

# Load the latest experiment lap metrics files and check what advanced metrics should be available


def debug_advanced_metrics():
    base_path = "/home/mohammedazab/ws/src/race_stack/race_monitor/evaluation_results/lqr_controller_node/exp_005_20251018_121710"
    metrics_path = os.path.join(base_path, "metrics")

    if not os.path.exists(metrics_path):
        print(f"Metrics path does not exist: {metrics_path}")
        return

    # Load all lap metrics
    all_metrics = {}
    lap_files = [f for f in os.listdir(metrics_path) if f.startswith('lap_') and f.endswith('_metrics.json')]
    lap_files.sort()

    print(f"Found {len(lap_files)} lap metric files")

    for lap_file in lap_files:
        lap_num = int(lap_file.split('_')[1])
        file_path = os.path.join(metrics_path, lap_file)

        with open(file_path, 'r') as f:
            lap_metrics = json.load(f)

        all_metrics[lap_num] = lap_metrics
        print(f"Lap {lap_num}: {len(lap_metrics)} metrics")

        # Check for APE/RPE metrics
        ape_keys = [k for k in lap_metrics.keys() if k.startswith('ape_')]
        rpe_keys = [k for k in lap_metrics.keys() if k.startswith('rpe_')]
        print(f"  APE metrics: {len(ape_keys)} - {ape_keys[:3]}{'...' if len(ape_keys) > 3 else ''}")
        print(f"  RPE metrics: {len(rpe_keys)} - {rpe_keys[:3]}{'...' if len(rpe_keys) > 3 else ''}")

    # Now simulate the averaging process that should happen in _calculate_averaged_metrics
    print("\n=== Simulating Advanced Metrics Calculation ===")

    # Collect all metrics from all laps
    aggregated_metrics = {}
    lap_count = len(all_metrics)

    print(f"Processing {lap_count} laps for aggregation")

    # Aggregate metrics across all laps
    for lap_num, lap_metrics in all_metrics.items():
        for metric_name, metric_value in lap_metrics.items():
            if isinstance(metric_value, (int, float)) and not np.isnan(metric_value):
                if metric_name not in aggregated_metrics:
                    aggregated_metrics[metric_name] = []
                aggregated_metrics[metric_name].append(metric_value)

    print(f"Found {len(aggregated_metrics)} unique metrics across all laps")

    # Calculate statistics for each metric
    averaged_metrics = {}
    for metric_name, values in aggregated_metrics.items():
        if len(values) > 0:
            try:
                averaged_metrics[f'{metric_name}_mean'] = float(np.mean(values))
                averaged_metrics[f'{metric_name}_std'] = float(np.std(values))
                averaged_metrics[f'{metric_name}_min'] = float(np.min(values))
                averaged_metrics[f'{metric_name}_max'] = float(np.max(values))
                averaged_metrics[f'{metric_name}_median'] = float(np.median(values))
            except Exception as e:
                print(f"Error calculating statistics for {metric_name}: {e}")
                continue

    print(f"Generated {len(averaged_metrics)} averaged metrics")

    # Filter APE/RPE metrics
    ape_metrics = {k: v for k, v in averaged_metrics.items() if k.startswith('ape_')}
    rpe_metrics = {k: v for k, v in averaged_metrics.items() if k.startswith('rpe_')}

    print(f"APE averaged metrics: {len(ape_metrics)}")
    print(f"RPE averaged metrics: {len(rpe_metrics)}")

    # Show some example values
    print("\nExample APE metrics:")
    for i, (k, v) in enumerate(ape_metrics.items()):
        if i < 5:  # Show first 5
            print(f"  {k}: {v}")

    print("\nExample RPE metrics:")
    for i, (k, v) in enumerate(rpe_metrics.items()):
        if i < 5:  # Show first 5
            print(f"  {k}: {v}")

    # Calculate overall values
    ape_keys = [k for k in averaged_metrics.keys() if k.startswith('ape_') and k.endswith('_mean')]
    rpe_keys = [k for k in averaged_metrics.keys() if k.startswith('rpe_') and k.endswith('_mean')]

    if ape_keys:
        ape_mean = np.mean([averaged_metrics[k] for k in ape_keys])
        print(f"\nOverall APE mean: {ape_mean}")

    if rpe_keys:
        rpe_mean = np.mean([averaged_metrics[k] for k in rpe_keys])
        print(f"Overall RPE mean: {rpe_mean}")

    print(f"\nTotal advanced metrics that should be in race_results.json: {len(averaged_metrics)}")

    # Load and compare with actual race_results.json
    race_results_path = os.path.join(base_path, "results", "race_results.json")
    if os.path.exists(race_results_path):
        with open(race_results_path, 'r') as f:
            race_results = json.load(f)

        actual_advanced_metrics = race_results.get('advanced_metrics', {})
        print(f"Actual advanced metrics in race_results.json: {len(actual_advanced_metrics)}")

        if len(actual_advanced_metrics) == 0:
            print("❌ PROBLEM: advanced_metrics section is empty in race_results.json!")
            print("The metrics are being calculated correctly per lap but not aggregated into the final summary.")
        else:
            print("✅ Advanced metrics are present in race_results.json")
    else:
        print(f"race_results.json not found at {race_results_path}")


if __name__ == "__main__":
    debug_advanced_metrics()
