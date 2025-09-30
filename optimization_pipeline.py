"""
Hyperparameter Optimization Pipeline using Optuna
Bayesian optimization for LoRA hyperparameters to minimize catastrophic forgetting
"""

import optuna
import subprocess
import json
import os
import logging
from typing import Dict, Tuple
from collections import defaultdict
from datasets import load_from_disk
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KnowledgeShiftCalculator:
    """Calculate weighted knowledge shift metrics"""

    def __init__(self):
        self.shift_types = {
            # Positive shifts
            'UK_to_HK': 3,  # Unknown to HighlyKnown
            'UK_to_MK': 2,  # Unknown to MaybeKnown
            'MK_to_HK': 1,  # MaybeKnown to HighlyKnown
            # Negative shifts
            'HK_to_UK': -3,  # HighlyKnown to Unknown
            'HK_to_MK': -1,  # HighlyKnown to MaybeKnown
            'MK_to_UK': -2,  # MaybeKnown to Unknown
        }

    def calculate_shifts_from_datasets(self, original_dataset_path: str,
                                       new_dataset_path: str) -> Dict[str, int]:
        """
        Calculate knowledge shifts between two datasets

        Args:
            original_dataset_path: Path to baseline dataset
            new_dataset_path: Path to post-training dataset

        Returns:
            Dictionary with shift counts
        """
        logger.info("Loading datasets for shift calculation...")

        # Load datasets
        original_ds = load_from_disk(original_dataset_path)['full']
        new_ds = load_from_disk(new_dataset_path)['full']

        # Create dictionaries question -> category
        original_categories = {item['question']: item['Category'] for item in original_ds}
        new_categories = {item['question']: item['Category'] for item in new_ds}

        # Count shifts
        shifts = defaultdict(int)

        for question in original_categories:
            if question in new_categories:
                orig_cat = original_categories[question]
                new_cat = new_categories[question]

                if orig_cat != new_cat:
                    # Create shift key
                    shift_key = f"{orig_cat[:2]}_to_{new_cat[:2]}"
                    shifts[shift_key] += 1

        return dict(shifts)

    def calculate_objective_score(self, shifts: Dict[str, int]) -> float:
        """
        Calculate weighted objective function for Optuna

        Args:
            shifts: Dictionary of shift counts

        Returns:
            Normalized weighted score
        """
        total_score = 0
        total_shifts = 0

        for shift_type, count in shifts.items():
            if shift_type in self.shift_types:
                weight = self.shift_types[shift_type]
                total_score += weight * count
                total_shifts += count

        # Normalize by total shifts
        if total_shifts > 0:
            normalized_score = total_score / total_shifts
        else:
            normalized_score = 0

        return normalized_score


def run_dataset_generation(model_path: str, data_path: str, output_path: str,
                           max_examples: int = 500) -> bool:
    """
    Run dataset generation script

    Args:
        model_path: Path to trained model
        data_path: Path to source data
        output_path: Where to save generated dataset
        max_examples: Number of examples to process

    Returns:
        Success status
    """
    try:
        cmd = [
            "python", "generate_dataset.py",
            "--model_name", model_path,
            "--data_path", data_path,
            "--output_path", output_path,
            "--max_examples", str(max_examples),
            "--n_shot", "4",
            "--n_experiments", "10",
            "--batch_size", "256"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

        if result.returncode == 0:
            logger.info(f"Dataset generation completed: {output_path}")
            return True
        else:
            logger.error(f"Dataset generation failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("Dataset generation timed out")
        return False
    except Exception as e:
        logger.error(f"Error in dataset generation: {e}")
        return False


def run_lora_training(data_path: str, rank: int, learning_rate: float,
                      lora_alpha: float, dropout: float = 0.1,
                      unknown: int = 500, high_known: int = 1,
                      seed: int = 42) -> Tuple[bool, str]:
    """
    Run LoRA training

    Args:
        data_path: Path to training data
        rank: LoRA rank
        learning_rate: Learning rate
        lora_alpha: LoRA alpha parameter
        dropout: Dropout rate
        unknown: Number of unknown facts
        high_known: Number of highly known facts per unknown
        seed: Random seed

    Returns:
        (success, model_path)
    """
    try:
        cmd = [
            "python", "train_lora.py",
            "--data_path", data_path,
            "--unknown", str(unknown),
            "--high_known", str(high_known),
            "--rank", str(rank),
            "--learning_rate", str(learning_rate),
            "--lora_alpha", str(lora_alpha),
            "--dropout", str(dropout),
            "--seed", str(seed)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10800)

        if result.returncode == 0:
            # Determine model path
            flag_str = 'HighKnown'
            model_path = (f"./lora_outputs/{flag_str}/rank{rank}_"
                          f"lr{learning_rate}_alpha{lora_alpha}_"
                          f"seed{seed}_unknown{unknown}_"
                          f"hk{high_known}/final_model")

            logger.info(f"LoRA training completed: {model_path}")
            return True, model_path
        else:
            logger.error(f"LoRA training failed: {result.stderr}")
            return False, ""

    except subprocess.TimeoutExpired:
        logger.error("LoRA training timed out")
        return False, ""
    except Exception as e:
        logger.error(f"Error in LoRA training: {e}")
        return False, ""


def objective(trial, base_dataset_path: str, test_questions_path: str,
              output_dir: str):
    """
    Objective function for Optuna optimization

    Args:
        trial: Optuna trial object
        base_dataset_path: Path to baseline evaluation dataset
        test_questions_path: Path to test questions
        output_dir: Directory for outputs

    Returns:
        Objective score (to maximize)
    """
    # Suggest hyperparameters
    rank = trial.suggest_categorical("rank", [1, 4, 8, 16])
    learning_rate = trial.suggest_categorical("learning_rate",
                                              [1e-5, 3e-5, 5e-5, 5e-4, 1e-3, 3e-3, 1e-2])
    lora_alpha = trial.suggest_categorical("lora_alpha",
                                           [0.01, 0.1, 1, 2, 4, 8, 10])

    trial_name = f"trial_{trial.number}"
    logger.info(f"\n=== Starting {trial_name} ===")
    logger.info(f"Parameters: rank={rank}, lr={learning_rate:.2e}, alpha={lora_alpha}")

    try:
        # Train LoRA model
        success, model_path = run_lora_training(
            data_path=base_dataset_path,
            rank=rank,
            learning_rate=learning_rate,
            lora_alpha=lora_alpha,
            seed=42
        )

        if not success:
            return -10.0

        # Generate dataset with trained model
        after_training_dataset = os.path.join(output_dir, f"after_training_{trial_name}")
        success = run_dataset_generation(
            model_path=model_path,
            data_path=test_questions_path,
            output_path=after_training_dataset,
            max_examples=500
        )

        if not success:
            logger.error(f"Dataset generation failed for {trial_name}")
            return -10.0

        # Calculate shifts
        shift_calculator = KnowledgeShiftCalculator()
        shifts = shift_calculator.calculate_shifts_from_datasets(
            base_dataset_path, after_training_dataset
        )

        score = shift_calculator.calculate_objective_score(shifts)

        # Log additional metrics
        positive_shifts = sum(count for shift, count in shifts.items()
                              if shift in shift_calculator.shift_types and
                              shift_calculator.shift_types[shift] > 0)
        negative_shifts = sum(count for shift, count in shifts.items()
                              if shift in shift_calculator.shift_types and
                              shift_calculator.shift_types[shift] < 0)

        trial.set_user_attr("positive_shifts", positive_shifts)
        trial.set_user_attr("negative_shifts", negative_shifts)
        trial.set_user_attr("net_shifts", positive_shifts - negative_shifts)
        trial.set_user_attr("model_path", model_path)

        logger.info(f"\n{trial_name} completed with score: {score:.4f}")
        logger.info(f"Positive shifts: {positive_shifts}, Negative shifts: {negative_shifts}")

        # Clean up temporary files
        if os.path.exists(after_training_dataset):
            shutil.rmtree(after_training_dataset)

        return score

    except Exception as e:
        logger.error(f"Error in {trial_name}: {e}")
        return -10.0


def run_optimization(base_dataset_path: str, test_questions_path: str,
                     output_dir: str, n_trials: int = 28,
                     study_name: str = "lora_optimization"):
    """
    Run hyperparameter optimization

    Args:
        base_dataset_path: Path to baseline dataset
        test_questions_path: Path to test questions
        output_dir: Directory for outputs
        n_trials: Number of optimization trials
        study_name: Name for Optuna study
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create study
    storage = f"sqlite:///{os.path.join(output_dir, 'optuna.db')}"
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True
    )

    logger.info("Starting Optuna optimization...")

    try:
        study.optimize(
            lambda trial: objective(trial, base_dataset_path,
                                    test_questions_path, output_dir),
            n_trials=n_trials,
            timeout=None,
            callbacks=[
                lambda study, trial: logger.info(
                    f"Trial {trial.number} finished with score: {trial.value}"
                )
            ]
        )

        logger.info("Optimization completed!")

        # Print results
        print_optimization_results(study)

    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
    except Exception as e:
        logger.error(f"Optimization failed: {e}")

    return study


def print_optimization_results(study):
    """Print optimization results summary"""
    logger.info("\n=== Optimization Results ===")

    # Best trial
    best_trial = study.best_trial
    logger.info(f"\nBest trial: {best_trial.number}")
    logger.info(f"Best score: {best_trial.value:.4f}")
    logger.info("Best parameters:")
    for key, value in best_trial.params.items():
        logger.info(f"  {key}: {value}")

    logger.info("\nBest trial attributes:")
    for key, value in best_trial.user_attrs.items():
        logger.info(f"  {key}: {value}")

    # Statistics
    logger.info(f"\nTotal trials completed: {len(study.trials)}")
    logger.info(f"Best score achieved: {study.best_value:.4f}")

    # Top trials
    df = study.trials_dataframe()
    if len(df) > 0:
        logger.info("\nTop 5 trials by score:")
        top_cols = ['number', 'value', 'params_rank', 'params_learning_rate', 'params_lora_alpha']
        available_cols = [col for col in top_cols if col in df.columns]
        top_trials = df.nlargest(5, 'value')[available_cols]
        logger.info(f"\n{top_trials.to_string()}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run LoRA hyperparameter optimization")
    parser.add_argument("--base_dataset", type=str, required=True,
                        help="Path to baseline dataset")
    parser.add_argument("--test_questions", type=str, required=True,
                        help="Path to test questions dataset")
    parser.add_argument("--output_dir", type=str, default="./optimization_results",
                        help="Output directory")
    parser.add_argument("--n_trials", type=int, default=28,
                        help="Number of optimization trials")
    parser.add_argument("--study_name", type=str, default="lora_optimization",
                        help="Name for Optuna study")

    args = parser.parse_args()

    run_optimization(
        base_dataset_path=args.base_dataset,
        test_questions_path=args.test_questions,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        study_name=args.study_name
    )


if __name__ == "__main__":
    main()