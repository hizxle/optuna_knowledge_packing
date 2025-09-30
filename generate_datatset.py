"""
Dataset Generation and Knowledge Categorization Module
Generates datasets with knowledge categories based on model predictions
"""

import torch
import argparse
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    PreTrainedModel,
    PreTrainedTokenizer
)
from tqdm import tqdm
import logging
from collections import defaultdict
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    n_shot: int = 4
    n_experiments: int = 10
    batch_size_greedy: int = 256
    batch_size_sample: int = 16
    max_new_tokens: int = 64
    temperature_sample: float = 0.7
    top_p_sample: float = 0.9
    num_return_sequences: int = 5


class KnowledgeClassifier:
    """Classifies facts based on model's knowledge"""

    HIGHLY_KNOWN = "HighlyKnown"
    MAYBE_KNOWN = "MaybeKnown"
    WEAKLY_KNOWN = "WeaklyKnown"
    UNKNOWN = "Unknown"

    @staticmethod
    def classify_knowledge(predictions: Dict[str, List]) -> str:
        """
        Classify knowledge based on prediction accuracy

        Args:
            predictions: Dict with 'p_greed' and 'p_sample' lists

        Returns:
            Category name
        """
        greedy_correct = predictions.get('p_greed', [])
        sample_correct = predictions.get('p_sample', [])

        if all(greedy_correct):
            return KnowledgeClassifier.HIGHLY_KNOWN
        elif any(greedy_correct):
            return KnowledgeClassifier.MAYBE_KNOWN
        elif any(sample_correct):
            return KnowledgeClassifier.WEAKLY_KNOWN
        else:
            return KnowledgeClassifier.UNKNOWN


class DatasetGenerator:
    """Efficient dataset generator with knowledge classification"""

    def __init__(self, model_name: str, config: GenerationConfig = None):
        self.model_name = model_name
        self.config = config or GenerationConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        self.pipeline = self._create_pipeline()

        # Load few-shot dataset
        self.fewshot_dataset = load_dataset("trivia_qa", "rc", trust_remote_code=True)["train"]

    def _load_model_and_tokenizer(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load and configure model and tokenizer"""
        logger.info(f"Loading model: {self.model_name}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side='left',
            trust_remote_code=True
        )

        # Configure padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Load model with optimizations
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation='eager',
        ).eval()

        return model, tokenizer

    def _create_pipeline(self) -> pipeline:
        """Create text generation pipeline"""
        return pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
        )

    def _create_few_shot_prompt(self, question: str, shot_examples: List[Dict]) -> str:
        """Create few-shot prompt for a question"""
        messages = [
            {"role": "system", "content": "Answer the following question. Final answer start with 'Answer:' "}
        ]

        # Add few-shot examples
        for example in shot_examples:
            messages.extend([
                {"role": "user", "content": f"Question: {example['question']}"},
                {"role": "assistant", "content": f"Answer: {example['answer']['normalized_aliases'][0]}"}
            ])

        # Add the actual question
        messages.append({"role": "user", "content": f"Question: {question}"})

        # Apply chat template
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def prepare_prompts(self, dataset: Dataset) -> Dataset:
        """Prepare few-shot prompts for all questions"""
        logger.info("Preparing few-shot prompts...")

        all_prompts = []
        all_questions = []
        all_answers = []

        # Process each question with N experiments
        for idx, example in enumerate(tqdm(dataset, desc="Preparing prompts")):
            question = example['question']
            answer = example['answer']

            # Generate N_EXP different few-shot contexts
            for exp_idx in range(self.config.n_experiments):
                # Select few-shot examples
                start_idx = self.config.n_shot * exp_idx
                end_idx = start_idx + self.config.n_shot
                shot_examples = self.fewshot_dataset.select(range(start_idx, end_idx))

                # Create prompt
                prompt = self._create_few_shot_prompt(question, shot_examples)

                all_prompts.append(prompt)
                all_questions.append(question)
                all_answers.append(answer)

        # Create new dataset with prompts
        return Dataset.from_dict({
            'primed_question': all_prompts,
            'question': all_questions,
            'answer': all_answers
        })

    def generate_predictions(self, dataset: Dataset, use_sampling: bool = False) -> List[str]:
        """Generate predictions for all prompts"""
        mode = "sampling" if use_sampling else "greedy"
        logger.info(f"Generating predictions with {mode} decoding...")

        predictions = []
        batch_size = self.config.batch_size_sample if use_sampling else self.config.batch_size_greedy

        # Create data generator
        def data_generator():
            for item in dataset:
                yield item['primed_question']

        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "add_special_tokens": True,
            "return_full_text": False,
            "batch_size": batch_size,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        if use_sampling:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": self.config.temperature_sample,
                "top_p": self.config.top_p_sample,
                "num_return_sequences": self.config.num_return_sequences,
            })
        else:
            gen_kwargs.update({
                "do_sample": False,
                "temperature": None,
                "top_p": None,
                "top_k": None,
            })

        # Generate predictions
        for output in tqdm(
                self.pipeline(data_generator(), **gen_kwargs),
                total=len(dataset),
                desc=f"Generating {mode} predictions"
        ):
            if use_sampling and isinstance(output, list):
                # Multiple sequences for sampling
                predictions.append([seq['generated_text'] for seq in output])
            else:
                # Single sequence for greedy
                predictions.append(output[0]['generated_text'])

        return predictions

    def evaluate_predictions(self, dataset: Dataset) -> Dataset:
        """Evaluate prediction accuracy"""
        logger.info("Evaluating predictions...")

        def check_accuracy(examples):
            """Check if predictions contain correct answers"""
            p_greed = []
            p_sample = []

            for i in range(len(examples['answer'])):
                answer = examples['answer'][i]
                greedy_pred = examples['greedy_ans'][i]
                sample_preds = examples.get('sample_ans', [[]])[i]

                # Normalize answer aliases
                if isinstance(answer, dict):
                    aliases = answer.get('normalized_aliases', [])
                else:
                    aliases = answer[0].get('normalized_aliases', []) if isinstance(answer, list) else []

                # Check greedy prediction
                greedy_correct = any(
                    alias.lower() in greedy_pred.lower()
                    for alias in aliases if alias
                )
                p_greed.append(greedy_correct)

                # Check sampling predictions
                if isinstance(sample_preds, list) and sample_preds != ['None']:
                    sample_correct = any(
                        any(alias.lower() in pred.lower() for alias in aliases if alias)
                        for pred in sample_preds if pred
                    )
                else:
                    sample_correct = False
                p_sample.append(sample_correct)

            examples['p_greed'] = p_greed
            examples['p_sample'] = p_sample

            return examples

        return dataset.map(
            check_accuracy,
            batched=True,
            batch_size=1000,
            desc="Evaluating accuracy"
        )

    def aggregate_results(self, dataset: Dataset) -> Dataset:
        """Aggregate results by question and classify knowledge"""
        logger.info("Aggregating results and classifying knowledge...")

        # Group by question
        question_groups = defaultdict(lambda: {
            'primed_questions': [],
            'greedy_ans': [],
            'sample_ans': [],
            'p_greed': [],
            'p_sample': [],
            'answer': None
        })

        for item in dataset:
            q = item['question']
            question_groups[q]['primed_questions'].append(item['primed_question'])
            question_groups[q]['greedy_ans'].append(item['greedy_ans'])
            question_groups[q]['sample_ans'].append(item.get('sample_ans', 'None'))
            question_groups[q]['p_greed'].append(item['p_greed'])
            question_groups[q]['p_sample'].append(item['p_sample'])
            question_groups[q]['answer'] = item['answer']

        # Create aggregated dataset
        aggregated_data = []
        for question, data in question_groups.items():
            category = KnowledgeClassifier.classify_knowledge({
                'p_greed': data['p_greed'],
                'p_sample': data['p_sample']
            })

            # For MaybeKnown, find good and bad examples
            good_example = None
            bad_example = None
            if category == KnowledgeClassifier.MAYBE_KNOWN:
                for i, is_correct in enumerate(data['p_greed']):
                    if is_correct and good_example is None:
                        good_example = data['primed_questions'][i]
                    elif not is_correct and bad_example is None:
                        bad_example = data['primed_questions'][i]
                    if good_example and bad_example:
                        break

            aggregated_data.append({
                'question': question,
                'answer': data['answer'],
                'Category': category,
                'primed_question': data['primed_questions'],
                'greedy_ans': data['greedy_ans'],
                'sample_ans': data['sample_ans'],
                'p_greed': data['p_greed'],
                'p_sample': data['p_sample'],
                'good_example': good_example,
                'bad_example': bad_example
            })

        return Dataset.from_list(aggregated_data)

    def generate_dataset(self, data_path: str, output_path: str,
                         use_sampling: bool = False, max_examples: Optional[int] = None):
        """Main method to generate and classify dataset"""
        logger.info(f"Loading dataset from {data_path}")

        # Load input dataset
        dataset = load_dataset(data_path)['full']
        if max_examples:
            dataset = dataset.select(range(min(max_examples, len(dataset))))

        # Ensure answer format is consistent
        def clean_answers(example):
            if 'answer' in example and isinstance(example['answer'], list) and example['answer']:
                example['answer'] = example['answer'][0]
            return example

        dataset = dataset.map(clean_answers)

        # Prepare prompts
        prompted_dataset = self.prepare_prompts(dataset)

        # Generate greedy predictions
        greedy_predictions = self.generate_predictions(prompted_dataset, use_sampling=False)
        prompted_dataset = prompted_dataset.add_column("greedy_ans", greedy_predictions)

        # Generate sampling predictions if requested
        if use_sampling:
            sample_predictions = self.generate_predictions(prompted_dataset, use_sampling=True)
            prompted_dataset = prompted_dataset.add_column("sample_ans", sample_predictions)
        else:
            prompted_dataset = prompted_dataset.add_column(
                "sample_ans",
                [['None']] * len(prompted_dataset)
            )

        # Evaluate predictions
        evaluated_dataset = self.evaluate_predictions(prompted_dataset)

        # Aggregate and classify
        final_dataset = self.aggregate_results(evaluated_dataset)

        # Create dataset dict
        dataset_dict = DatasetDict({"full": final_dataset})

        # Save dataset
        logger.info(f"Saving dataset to {output_path}")
        dataset_dict.save_to_disk(output_path)

        # Print statistics
        self._print_statistics(final_dataset)

        # Clean up
        gc.collect()
        torch.cuda.empty_cache()

        return dataset_dict

    def _print_statistics(self, dataset: Dataset):
        """Print dataset statistics"""
        categories = dataset['Category']
        category_counts = defaultdict(int)
        for cat in categories:
            category_counts[cat] += 1

        total = len(dataset)
        logger.info("\nDataset Statistics:")
        logger.info(f"Total examples: {total}")
        for category, count in sorted(category_counts.items()):
            percentage = (count / total) * 100
            logger.info(f"{category}: {count} ({percentage:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Generate dataset with knowledge classification")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name or path")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to input dataset")
    parser.add_argument("--output_path", type=str, default="classified_dataset",
                        help="Path to save generated dataset")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Maximum number of examples to process")
    parser.add_argument("--use_sampling", action="store_true",
                        help="Also generate predictions using sampling")
    parser.add_argument("--n_shot", type=int, default=4,
                        help="Number of few-shot examples")
    parser.add_argument("--n_experiments", type=int, default=10,
                        help="Number of experiments per question")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for generation")

    args = parser.parse_args()

    # Create configuration
    config = GenerationConfig(
        n_shot=args.n_shot,
        n_experiments=args.n_experiments,
        batch_size_greedy=args.batch_size,
        batch_size_sample=args.batch_size // 16
    )

    # Initialize generator
    generator = DatasetGenerator(args.model_name, config)

    # Generate dataset
    generator.generate_dataset(
        data_path=args.data_path,
        output_path=args.output_path,
        use_sampling=args.use_sampling,
        max_examples=args.max_examples
    )


if __name__ == "__main__":
    main()