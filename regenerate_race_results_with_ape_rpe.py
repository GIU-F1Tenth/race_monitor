#!/usr/bin/env python3

"""
Test script to regenerate race_results.json with APE/RPE advanced metrics
using the existing lap data from exp_005
"""

from race_monitor.race_monitor import RaceMonitor
import sys
import os
import json
from datetime import datetime

# Add the race_monitor package to path
sys.path.insert(0, '/home/mohammedazab/ws/src/race_stack/race_monitor')


def regenerate_race_results_with_advanced_metrics():
    """Regenerate race_results.json with proper APE/RPE metrics"""

    exp_dir = "/home/mohammedazab/ws/src/race_stack/race_monitor/evaluation_results/lqr_controller_node/exp_005_20251018_121710"

    print(f"Regenerating race_results.json for experiment: {exp_dir}")

    # Load existing race results to get the basic race data
    race_results_path = os.path.join(exp_dir, "results", "race_results.json")
    if not os.path.exists(race_results_path):
        print(f"❌ Race results file not found: {race_results_path}")
        return False

    with open(race_results_path, 'r') as f:
        existing_results = json.load(f)

    print(f"✅ Loaded existing race results")

    # Create a mock race monitor to use our fixed _load_metrics_from_files method
    import rclpy
    rclpy.init()

    try:
        race_monitor = RaceMonitor()

        # Create a mock research evaluator with the experiment directory
        class MockResearchEvaluator:
            def __init__(self):
                self.experiment_dir = exp_dir
                self.detailed_metrics = {}  # Start empty to force fallback

        race_monitor.research_evaluator = MockResearchEvaluator()

        # Use our fixed method to calculate advanced metrics
        print("🔄 Calculating advanced metrics using fixed method...")
        advanced_metrics = race_monitor._calculate_averaged_metrics()

        print(f"✅ Generated {len(advanced_metrics)} advanced metrics")

        if len(advanced_metrics) > 0:
            # Update the existing results with the advanced metrics
            existing_results['advanced_metrics'] = advanced_metrics

            # Add a note about the fix
            existing_results['race_info']['advanced_metrics_note'] = "APE/RPE metrics generated using fallback method"
            existing_results['race_info']['metrics_fix_timestamp'] = datetime.now().isoformat()

            # Save the updated results
            output_path = os.path.join(exp_dir, "results", "race_results_with_ape_rpe.json")
            with open(output_path, 'w') as f:
                json.dump(existing_results, f, indent=2)

            print(f"✅ Saved updated race results with APE/RPE metrics to: {output_path}")

            # Show summary of APE/RPE metrics
            ape_metrics = {k: v for k, v in advanced_metrics.items() if k.startswith('ape_')}
            rpe_metrics = {k: v for k, v in advanced_metrics.items() if k.startswith('rpe_')}

            print(f"\n📊 APE/RPE Metrics Summary:")
            print(f"   APE metrics: {len(ape_metrics)}")
            print(f"   RPE metrics: {len(rpe_metrics)}")

            # Show key overall metrics
            if 'overall_ape_mean' in advanced_metrics:
                print(f"   Overall APE Mean: {advanced_metrics['overall_ape_mean']:.4f}")
            if 'overall_rpe_mean' in advanced_metrics:
                print(f"   Overall RPE Mean: {advanced_metrics['overall_rpe_mean']:.4f}")

            # Show some example APE/RPE values
            print(f"\n📈 Example APE Metrics:")
            for i, (k, v) in enumerate(ape_metrics.items()):
                if i < 3:
                    print(f"   {k}: {v:.6f}")

            print(f"\n📈 Example RPE Metrics:")
            for i, (k, v) in enumerate(rpe_metrics.items()):
                if i < 3:
                    print(f"   {k}: {v:.6f}")

            return True
        else:
            print("❌ No advanced metrics generated")
            return False

    except Exception as e:
        print(f"❌ Error during regeneration: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    success = regenerate_race_results_with_advanced_metrics()
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}: Race results regeneration {'completed' if success else 'failed'}")
    sys.exit(0 if success else 1)
