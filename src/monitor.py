#!/usr/bin/env python
"""
REAL-TIME TRAINING MONITOR
===========================
Monitor your model training in real-time
Run in a separate terminal: python src/monitor.py
"""

import os
import json
import time
from pathlib import Path
import numpy as np
from datetime import datetime

def clear_screen():
    """Clear terminal screen"""
    os.system('clear' if os.name == 'posix' else 'cls')

def format_time(seconds):
    """Format seconds to human readable"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def get_file_age(filepath):
    """Get file age in seconds"""
    return time.time() - os.path.getmtime(filepath)

def monitor_training():
    """Monitor ongoing training"""
    results_dir = "results"

    print("\n" + "="*70)
    print("ASD MODEL TRAINING MONITOR")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results directory: {results_dir}\n")

    # Track previously seen files
    prev_accuracies = {}
    check_count = 0

    while True:
        clear_screen()
        check_count += 1

        print("\n" + "="*70)
        print("ASD MODEL TRAINING MONITOR")
        print("="*70)
        print(f"Check #{check_count} - Updated: {datetime.now().strftime('%H:%M:%S')}\n")

        # Check for training results
        models = ['resnet50', 'efficientnet', 'vit']
        training_active = False

        for model_name in models:
            results_file = f"{results_dir}/optimal_{model_name}_results.json"

            if os.path.exists(results_file):
                training_active = True
                try:
                    with open(results_file, 'r') as f:
                        data = json.load(f)

                    best_acc = data.get('best_val_acc', 0)
                    epochs = data.get('epochs_trained', 0)
                    file_age = get_file_age(results_file)

                    # Check if improving
                    if model_name in prev_accuracies:
                        improvement = best_acc - prev_accuracies[model_name]
                        status = "↑" if improvement > 0 else "→" if improvement == 0 else "↓"
                    else:
                        status = "●"
                        improvement = 0

                    prev_accuracies[model_name] = best_acc

                    print(f"{status} {model_name.upper():20s} | Acc: {best_acc:6.2f}% | Epochs: {epochs:3d} | Updated: {format_time(file_age)}")

                except Exception as e:
                    print(f"✗ {model_name.upper():20s} | Error reading results: {str(e)[:30]}")
            else:
                model_pth = f"{results_dir}/optimal_{model_name}_best.pth"
                if os.path.exists(model_pth):
                    training_active = True
                    file_age = get_file_age(model_pth)
                    print(f"⧖ {model_name.upper():20s} | Training... | Model saved: {format_time(file_age)}")
                else:
                    print(f"⌛ {model_name.upper():20s} | Waiting to start...")

        print("\n" + "-"*70)

        # Check for analysis results
        if os.path.exists(f"{results_dir}/optimal_summary_table.png"):
            print("✓ Analysis complete! Check results/optimal_*.png for visualizations")
            break
        elif os.path.exists(f"{results_dir}/optimal_all_models_results.json"):
            print("✓ Analysis running... generating visualizations")

        print("\nStatus: ", end="")
        if training_active:
            print("TRAINING IN PROGRESS - Check back in 5-10 minutes")
        else:
            print("WAITING FOR TRAINING TO START")

        print("\nTo view current best accuracies, check:")
        print(f"  cat {results_dir}/optimal_*_results.json")
        print("\nTo stop monitoring: Press Ctrl+C")

        # Sleep before next check
        time.sleep(30)


def show_quick_summary():
    """Show quick summary of current best models"""
    print("\n" + "="*70)
    print("QUICK SUMMARY - BEST MODELS")
    print("="*70 + "\n")

    results_dir = "results"
    models = ['resnet50', 'efficientnet', 'vit']
    all_accs = []

    for model_name in models:
        results_file = f"{results_dir}/optimal_{model_name}_results.json"
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    best_acc = data.get('best_val_acc', 0)
                    all_accs.append(best_acc)
                    print(f"{model_name:20s}: {best_acc:6.2f}%")
            except:
                pass

    if all_accs:
        ensemble_acc = np.mean(all_accs)
        print(f"\n{'ENSEMBLE (Average)':20s}: {ensemble_acc:6.2f}%")
        print(f"{'Best Single Model':20s}: {max(all_accs):6.2f}%")
        print(f"{'Improvement from baseline':20s}: +{(ensemble_acc - 61.46):5.2f}%")

    # Show comparison plots
    comparison_file = f"{results_dir}/optimal_comparison_detailed.png"
    summary_file = f"{results_dir}/optimal_summary_table.png"

    print("\nGenerated visualizations:")
    if os.path.exists(comparison_file):
        print(f"  ✓ {comparison_file}")
    if os.path.exists(summary_file):
        print(f"  ✓ {summary_file}")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--summary':
        show_quick_summary()
    else:
        try:
            monitor_training()
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            show_quick_summary()
