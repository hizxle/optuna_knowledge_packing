"""
Utility functions for visualization and analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List
import optuna


def plot_optimization_history(study: optuna.Study, save_path: str = None):
    """
    Plot optimization history

    Args:
        study: Optuna study object
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot optimization history
    trials = study.trials
    values = [trial.value for trial in trials if trial.value is not None]
    trial_numbers = [trial.number for trial in trials if trial.value is not None]

    ax1.plot(trial_numbers, values, 'o-', alpha=0.7)
    ax1.axhline(y=study.best_value, color='r', linestyle='--',
                label=f'Best: {study.best_value:.3f}')
    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel('Objective Value')
    ax1.set_title('Optimization History')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot best value over time
    best_values = []
    current_best = float('-inf')
    for val in values:
        current_best = max(current_best, val)
        best_values.append(current_best)

    ax2.plot(trial_numbers, best_values, 'g-', linewidth=2)
    ax2.set_xlabel('Trial Number')
    ax2.set_ylabel('Best Value So Far')
    ax2.set_title('Best Value Evolution')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_parameter_importance(study: optuna.Study, save_path: str = None):
    """
    Plot parameter importance

    Args:
        study: Optuna study object
        save_path: Path to save figure
    """
    try:
        importance = optuna.importance.get_param_importances(study)

        fig, ax = plt.subplots(figsize=(10, 6))

        params = list(importance.keys())
        values = list(importance.values())

        ax.barh(params, values)
        ax.set_xlabel('Importance')
        ax.set_title('Hyperparameter Importance')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    except Exception as e:
        print(f"Could not compute parameter importance: {e}")


def plot_param_relationships(study: optuna.Study, param_name: str, save_path: str = None):
    """
    Plot relationship between a parameter and objective value

    Args:
        study: Optuna study object
        param_name: Name of parameter to analyze
        save_path: Path to save figure
    """
    df = study.trials_dataframe()

    if f'params_{param_name}' not in df.columns:
        print(f"Parameter {param_name} not found in study")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    param_col = f'params_{param_name}'
    df_valid = df[df['value'].notna()]

    # Group by parameter value and compute statistics
    grouped = df_valid.groupby(param_col)['value'].agg(['mean', 'std', 'count'])

    ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                fmt='o-', capsize=5, capthick=2, markersize=8)

    ax.set_xlabel(param_name)
    ax.set_ylabel('Objective Value')
    ax.set_title(f'Effect of {param_name} on Objective')
    ax.grid(True, alpha=0.3)

    # Add count annotations
    for x, (idx, row) in zip(grouped.index, grouped.iterrows()):
        ax.annotate(f'n={int(row["count"])}',
                    xy=(x, row['mean']),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_knowledge_shifts(shifts: Dict[str, int], save_path: str = None):
    """
    Visualize knowledge shifts

    Args:
        shifts: Dictionary of shift counts
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Separate positive and negative shifts
    positive = {k: v for k, v in shifts.items() if any(pos in k for pos in ['UK_to', 'MK_to_HK'])}
    negative = {k: v for k, v in shifts.items() if any(neg in k for neg in ['HK_to', 'MK_to_UK'])}

    # Plot positive shifts
    if positive:
        ax1.bar(positive.keys(), positive.values(), color='green', alpha=0.7)
        ax1.set_title('Positive Knowledge Shifts')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')

    # Plot negative shifts
    if negative:
        ax2.bar(negative.keys(), negative.values(), color='red', alpha=0.7)
        ax2.set_title('Negative Knowledge Shifts')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_results_summary(study: optuna.Study) -> pd.DataFrame:
    """
    Create summary DataFrame of all trials

    Args:
        study: Optuna study object

    Returns:
        DataFrame with results
    """
    df = study.trials_dataframe()

    # Add derived columns
    if 'user_attrs_positive_shifts' in df.columns and 'user_attrs_negative_shifts' in df.columns:
        df['shift_ratio'] = df['user_attrs_positive_shifts'] / (df['user_attrs_negative_shifts'] + 1)

    # Sort by value
    df = df.sort_values('value', ascending=False)

    return df


def export_best_config(study: optuna.Study, output_path: str):
    """
    Export best configuration to file

    Args:
        study: Optuna study object
        output_path: Path to save configuration
    """
    best_trial = study.best_trial

    config = {
        'best_score': best_trial.value,
        'parameters': best_trial.params,
        'user_attributes': best_trial.user_attrs,
        'trial_number': best_trial.number
    }

    import json
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Best configuration saved to {output_path}")