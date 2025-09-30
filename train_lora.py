"""
LoRA Training Module with Unsloth Optimization
Efficient training of LoRA adapters for knowledge update tasks
"""

import torch
import argparse
import os
from unsloth import FastLanguageModel
from typing import Dict, Any
from datasets import load_dataset, DatasetDict, Dataset, load_from_disk, concatenate_datasets
from trl import SFTTrainer
from transformers import TrainingArguments
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoRATrainer:
    """Efficient LoRA trainer using unsloth library"""

    def __init__(self, args):
        self.args = args
        self.model_name = "unsloth/Qwen3-0.6B-Base"
        self.max_seq_length = 128
        self.load_in_4bit = True

        # Training hyperparameters
        self.learning_rate = args.learning_rate
        self.batch_size = 8
        self.epochs = 10
        self.n_shot_test = 4
        self.n_exp_test = 10
        self.n_shot_train = 0
        self.n_exp_train = 1

    def load_model_and_tokenizer(self):
        """Load model and tokenizer with unsloth optimizations"""
        logger.info("Loading model with unsloth...")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=torch.float16,
            load_in_4bit=self.load_in_4bit,
        )

        # Configure LoRA with unsloth
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.args.rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=self.args.rank * self.args.lora_alpha,
            lora_dropout=self.args.dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=self.args.seed,
            use_rslora=True,
            loftq_config=None,
        )

        return model, tokenizer

    def prepare_datasets(self) -> DatasetDict:
        """Prepare and mix datasets efficiently"""
        logger.info("Preparing datasets...")

        # Load base dataset
        dataset = load_from_disk(self.args.data_path)['full']

        # Filter unknown facts
        unknown_data = self._filter_unknown_data(dataset)
        unknown_test = unknown_data.select(range(min(self.args.unknown, len(unknown_data))))
        unknown_test = unknown_test.map(self._clean_answer_format)

        # Prepare training data based on paraphrase flag
        if self.args.paraphrase:
            train_data = self._prepare_paraphrase_data(unknown_test)
        else:
            train_data = self._prepare_highknown_data(dataset, unknown_test)

        # Create dataset dict
        return DatasetDict({
            'train': train_data,
            'test': unknown_test,
            'valid': unknown_test,
        })

    def _filter_unknown_data(self, dataset):
        """Filter unknown data"""
        return dataset.filter(lambda x: x['Category'] == 'Unknown')

    def _clean_answer_format(self, example):
        """Clean answer format for consistency"""
        if 'answer' in example and example['answer']:
            ans = example['answer'][0] if isinstance(example['answer'], list) else example['answer']
            example['answer'] = {
                'aliases': ans.get('aliases', []),
                'normalized_aliases': ans.get('normalized_aliases', [])
            }
        return example

    def _prepare_paraphrase_data(self, unknown_test):
        """Prepare paraphrase-based training data"""
        logger.info(f"Using {self.args.high_known} paraphrases per unknown fact")

        paraphrase_data = []
        for example in unknown_test:
            if 'para' in example and example['para']:
                paras = example['para'][:self.args.high_known]
                for para in paras:
                    new_example = dict(example)
                    new_example['question'] = para
                    paraphrase_data.append(new_example)

        if not paraphrase_data:
            return unknown_test

        para_dataset = Dataset.from_list(paraphrase_data)
        return concatenate_datasets([para_dataset, unknown_test])

    def _prepare_highknown_data(self, dataset, unknown_test):
        """Prepare highly known facts for training"""
        logger.info(f"Using {self.args.high_known} highly known facts per unknown fact")

        highknown_data = dataset.filter(lambda x: x['Category'] == 'HighlyKnown')
        n_highknown = min(self.args.high_known * self.args.unknown, len(highknown_data))
        highknown_selected = highknown_data.select(range(n_highknown))
        highknown_selected = highknown_selected.map(self._clean_answer_format)

        return concatenate_datasets([highknown_selected, unknown_test])

    def prepare_dataset_for_training(self, dataset: Dataset, tokenizer,
                                     is_train: bool = True) -> Dataset:
        """Prepare dataset with few-shot examples"""
        fewshot_dataset = load_dataset("trivia_qa", "rc", trust_remote_code=True)["train"]

        n_shot = self.n_shot_train if is_train else self.n_shot_test
        n_exp = self.n_exp_train if is_train else self.n_exp_test

        def format_with_fewshot(examples, idx):
            """Format examples with few-shot context"""
            formatted_texts = []

            for i in range(len(examples['question'])):
                # Get few-shot examples
                start_idx = n_shot * (idx[i] % n_exp)
                end_idx = start_idx + n_shot
                fewshot_examples = fewshot_dataset.select(range(start_idx, end_idx))

                # Build messages
                messages = [{"role": "system", "content": "Answer the following question."}]

                # Add few-shot examples
                for fs_ex in fewshot_examples:
                    messages.append({"role": "user", "content": f"Question: {fs_ex['question']}"})
                    messages.append({"role": "assistant",
                                     "content": f"Answer: {fs_ex['answer']['aliases'][0]}"})

                # Add current example
                messages.append({"role": "user", "content": f"Question: {examples['question'][i]}"})

                if is_train:
                    # Include answer for training
                    answer = examples['answer'][i]
                    answer_text = answer['normalized_aliases'][0] if isinstance(answer, dict) else str(answer)
                    messages.append({"role": "assistant", "content": f"Answer: {answer_text}"})

                # Apply chat template
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=not is_train
                )
                formatted_texts.append(text)

            return {"text": formatted_texts}

        # Apply formatting
        dataset = dataset.map(
            format_with_fewshot,
            batched=True,
            with_indices=True,
            batch_size=100,
            desc=f"Formatting {'train' if is_train else 'test'} examples"
        )

        return dataset

    def train(self):
        """Main training function"""
        # Load model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer()

        # Set padding token
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token

        # Prepare datasets
        datasets = self.prepare_datasets()

        # Format datasets for training
        train_dataset = self.prepare_dataset_for_training(
            datasets['train'], tokenizer, is_train=True
        )

        # Setup output directory
        flag_str = 'Paraphrase' if self.args.paraphrase else 'HighKnown'
        output_dir = (f"./lora_outputs/{flag_str}/rank{self.args.rank}_"
                      f"lr{self.learning_rate}_alpha{self.args.lora_alpha}_"
                      f"seed{self.args.seed}_unknown{self.args.unknown}_"
                      f"hk{self.args.high_known}")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=128,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            num_train_epochs=self.epochs,
            learning_rate=self.learning_rate,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            eval_strategy="no",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=self.args.seed,
            report_to="none",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

        # Initialize trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=2,
            packing=False,
            args=training_args,
        )

        # Train
        logger.info("Starting training...")
        trainer_stats = trainer.train()

        # Save model
        logger.info("Saving model...")
        model.save_pretrained_merged(
            os.path.join(output_dir, "final_model"),
            tokenizer,
            save_method="merged_16bit",
        )

        model.save_pretrained(os.path.join(output_dir, "lora_adapters"))
        datasets.save_to_disk(os.path.join(output_dir, "dataset_info"))

        logger.info(f"Training completed! Model saved to {output_dir}")

        return trainer_stats, output_dir


def main():
    parser = argparse.ArgumentParser(description="LoRA training with unsloth")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--unknown", type=int, required=True)
    parser.add_argument("--high_known", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--paraphrase", action='store_true')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--lora_alpha", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.1)

    args = parser.parse_args()

    trainer = LoRATrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()