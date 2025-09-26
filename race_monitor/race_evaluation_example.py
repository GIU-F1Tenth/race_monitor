#!/usr/bin/env python3

"""
Race Evaluation System Example

This example demonstrates the custom race evaluation system features:
- A-F performance grading
- Intelligent recommendations
- Auto-incrementing experiment IDs
- Comparison with previous experiments
- Focus on racing-specific metrics

Usage:
    python3 race_evaluation_example.py
"""

import os
import sys

# Add the race_monitor package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from race_monitor.race_evaluator import create_race_evaluator

def create_sample_research_data():
    """Create sample research data for demonstration."""
    return {
        "experiment_info": {
            "controller_name": "test_controller",
            "experiment_id": "exp001", 
            "total_laps": 5
        },
        "aggregate_statistics": {
            "duration": {
                "mean": 19.5,
                "std": 0.8,
                "min": 18.9,
                "max": 20.3,
                "count": 5
            },
            "avg_speed": {
                "mean": 3.4,
                "std": 0.1,
                "count": 5
            },
            "velocity_consistency": {
                "mean": 0.38
            },
            "curvature_variation": {
                "mean": 0.85
            },
            "path_efficiency": {
                "mean": 1.15
            },
            "steering_aggressiveness": {
                "mean": 0.7
            },
            "jerk": {
                "mean": 120
            },
            "velocity_max": {
                "max": 15.2
            },
            "path_length": {
                "mean": 66.0,
                "std": 0.2
            },
            "mean_curvature": {
                "mean": 0.008
            }
        },
        "lap_by_lap_analysis": {
            "1": {"duration": 19.2},
            "2": {"duration": 19.8},
            "3": {"duration": 19.1},
            "4": {"duration": 20.1},
            "5": {"duration": 19.3}
        }
    }

def create_sample_evo_metrics():
    """Create sample EVO metrics for demonstration."""
    return {
        "ape_rmse": 0.25,
        "ape_mean": 0.18,
        "ape_std": 0.12,
        "ape_max": 0.45,
        "rpe_rmse": 0.15,
        "rpe_mean": 0.11,
        "rpe_std": 0.08,
        "rpe_max": 0.28,
        "reference_available": True,
        "reference_deviation": 0.22
    }

def main():
    """Demonstrate the race evaluation system."""
    print("🏁 Race Evaluation System Demo")
    print("=" * 50)
    
    # Configuration for different grading strictness levels
    configs = [
        {
            'controller_name': 'demo_controller',
            'trajectory_output_directory': '/tmp/race_evaluation_demo',
            'grading_strictness': 'normal',
            'enable_recommendations': True,
            'enable_comparison': True,
            'auto_increment_experiment': True
        },
        {
            'controller_name': 'demo_controller',
            'trajectory_output_directory': '/tmp/race_evaluation_demo',
            'grading_strictness': 'strict',
            'enable_recommendations': True,
            'enable_comparison': True,
            'auto_increment_experiment': True
        }
    ]
    
    research_data = create_sample_research_data()
    evo_metrics = create_sample_evo_metrics()
    
    for config in configs:
        print(f"\n📊 Evaluating with {config['grading_strictness'].upper()} grading:")
        print("-" * 30)
        
        # Create evaluator
        evaluator = create_race_evaluator(config)
        
        # Generate evaluation
        evaluation = evaluator.create_race_evaluation(research_data, evo_metrics)
        
        # Display results
        eval_data = evaluation['race_evaluation']
        
        print(f"Overall Grade: {eval_data['performance_summary']['overall_grade']}")
        print(f"Score: {eval_data['performance_summary']['numerical_score']:.1f}/100")
        print(f"Experiment ID: {eval_data['metadata']['experiment_id']}")
        
        # Show category grades
        print("\nCategory Breakdown:")
        category_grades = eval_data['performance_summary']['category_grades']
        for category, grade in category_grades.items():
            print(f"  {category.replace('_', ' ').title()}: {grade}")
        
        # Show top recommendations
        recommendations = eval_data.get('recommendations', [])
        if recommendations:
            print(f"\nTop Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"  {i}. {rec}")
        
        # Show comparison info
        comparison = eval_data.get('comparison_analysis', {})
        if comparison.get('available'):
            print(f"\nComparison Analysis:")
            print(f"  Previous experiments: {comparison['total_previous_experiments']}")
            print(f"  Current ranking: {comparison['ranking']['current_rank']}/{comparison['ranking']['total_experiments']}")
            print(f"  Percentile: {comparison['ranking']['percentile']:.1f}%")
    
    print(f"\n✅ Demo complete! Check /tmp/race_evaluation_demo/ for generated files.")

if __name__ == '__main__':
    main()