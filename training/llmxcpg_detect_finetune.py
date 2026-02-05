import json
import logging
import argparse
import torch
import bitsandbytes as bnb
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, AutoTokenizer
import os


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('llmxcpg_detect_finetune.log')
        ]
    )
    return logging.getLogger(__name__)

def load_and_split_dataset(dataset_path: str, system_prompt_path: str, eval_split_ratio: float, seed: int) -> tuple[Dataset, Dataset]:
    try:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
    
        with open(system_prompt_path, 'r') as f:
            system_prompt = f.read()
        
        processed_data = []
        for item in data:
            processed_item = {
                'instruction': system_prompt + "\n\n## Instruction:\n" + item['instruction'].replace("Single word: VULNERABLE or BENIGN", "Single word: Yes (if code is VULNERABLE) or No (if code is BENIGN)"),
                'input': item['input'],
                'output': 'Yes' if item['output'] == 'VULNERABLE' else 'No'
            }
            processed_data.append(processed_item)
        
        train_data, eval_data = train_test_split(
            processed_data, test_size=eval_split_ratio, random_state=seed
        )
        
        train_dataset = Dataset.from_list(train_data)
        eval_dataset = Dataset.from_list(eval_data)

        return train_dataset, eval_dataset
    except FileNotFoundError as e:
        logging.error(f"Required file not found: {e}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {dataset_path}: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred during data loading or processing: {e}")
        raise


def formatting_prompts_func(examples):
    texts = []
    for instruction, input_text, output_text in zip(examples['instruction'], examples['input'], examples['output']):
        formatted_prompt = f"{instruction}\n## Input:\n{input_text}\n## Response:\n{output_text}\n"
        texts.append(formatted_prompt)
    return {"text": texts}


def find_linear_modules(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    
    return list(lora_module_names)

def setup_model_and_tokenizer(args, logger):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    
    target_modules = find_linear_modules(model)
    logger.info(f"Target Modules: {str(target_modules)}")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    
    return model, tokenizer

def setup_trainer(model, tokenizer, train_dataset, eval_dataset, args, logger):
    num_training_steps = len(train_dataset) * args.epochs // args.batch_size
    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        max_steps=0,
        warmup_ratio=0.10,
        learning_rate=args.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        report_to="neptune",
        save_strategy="steps",
        save_steps=max(100, num_training_steps // 10) if num_training_steps > 0 else 100,
        eval_strategy="steps",
        eval_steps=max(100, num_training_steps // 10) if num_training_steps > 0 else 100,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=args.seed,
        output_dir=args.model_output_dir,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        save_total_limit=2,
        greater_is_better=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=args.num_proc,
        packing=False,
        args=training_args,
    )

    return trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Vulnerability Classifier Training Script")
    
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the input JSON dataset")
    parser.add_argument("--system_prompt_path", type=str, default="./../prompts/detect_system_prompt.txt", help="Path to the system prompt text file")
    parser.add_argument("--model_name", type=str, required=True, help="Pretrained model to fine-tune")
    parser.add_argument("--output_dir", type=str, default="./models", help="Base directory to save trained model")
    
    parser.add_argument("--max_seq_length", type=int, default=8128, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=4e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--eval_split_ratio", type=float, default=0.1, help="Proportion of the dataset to include in the evaluation split")

    parser.add_argument("--lora_r", type=int, default=32, help="LoRA attention dimension (rank)")
    parser.add_argument("--lora_alpha", type=int, default=64, help="Alpha parameter for LoRA scaling")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout probability for LoRA layers")

    parser.add_argument("--num_proc", type=int, default=16, help="Number of processes for data processing")
    parser.add_argument("--experiment_name", type=str, default="LLMxCPG-D", help="Name of the experiment for logging and output directory")
    
    args = parser.parse_args()
    args.model_output_dir = os.path.join(args.output_dir, f"{args.model_name.split('/')[-1]}_{args.experiment_name}")
    return args


def main():
    args = parse_args()
    logger = setup_logging()

    try:
        logger.info(f"Loading model: {args.model_name}")
        model, tokenizer = setup_model_and_tokenizer(args, logger)

        logger.info(f"Loading dataset from: {args.dataset_path}, system prompt from {args.system_prompt_path} and splitting for evaluation ({args.eval_split_ratio*100:.2f}% eval)")
        train_dataset, eval_dataset = load_and_split_dataset(
            args.dataset_path, args.system_prompt_path, args.eval_split_ratio, args.seed
        )

        logger.info("Formatting prompts for datasets")
        train_dataset = train_dataset.map(
            formatting_prompts_func,
            batched=True,
            num_proc=args.num_proc,
            remove_columns=['instruction', 'input', 'output']
        )
        eval_dataset = eval_dataset.map(
            formatting_prompts_func,
            batched=True,
            num_proc=args.num_proc,
            remove_columns=['instruction', 'input', 'output']
        )
        
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Evaluation dataset size: {len(eval_dataset)}")

        logger.info("Setting up trainer")
        trainer = setup_trainer(model, tokenizer, train_dataset, eval_dataset, args, logger)

        logger.info("Starting training")
        trainer_stats = trainer.train()

        logger.info(f"Saving final model and tokenizer to {args.model_output_dir}")
        os.makedirs(args.model_output_dir, exist_ok=True)
        model.save_pretrained(args.model_output_dir)
        tokenizer.save_pretrained(args.model_output_dir)
    
        model.save_pretrained_merged(args.model_output_dir + "-vLLM", tokenizer, save_method = "merged_16bit",)


        logger.info("Training completed")
        return trainer_stats

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
