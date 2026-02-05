from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
import torch.nn as nn
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
import json
import logging
import argparse
from sklearn.model_selection import train_test_split

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('llmxcpg_query_finetune.log')
        ]
    )
    return logging.getLogger(__name__)

def load_json_dataset(file_path: str, eval_split_ratio: float = 0.1) -> tuple[Dataset, Dataset]:
    with open(file_path, 'r') as f:
        data = json.load(f)

    with open("./../prompts/query_system_prompt.txt", 'r') as f:
        system_prompt = f.read()

    conversations = []
    for item in data:
        conv = [
            {"role": "user", "content": f"{system_prompt}\n{item['instruction']}\n\n## Code snippet:\n```\n{item['input']}\n```".strip()},
            {"role": "assistant", "content": item['output']}
        ]
        conversations.append({"conversations": conv})

    train_data, eval_data = train_test_split(conversations, test_size=eval_split_ratio, random_state=42)

    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    return train_dataset, eval_dataset

def formatting_prompts_func(examples, tokenizer):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(
        convo,
        tokenize=False,
        add_generation_prompt=False
    ) for convo in convos]
    return {"text": texts}

def find_all_linear_names(model):
    import torch.nn as nn
    lora_module_names = set()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')

    return list(lora_module_names)

def setup_model_and_tokenizer(args, logger):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16,  # Full bfloat16 precision for A100
        load_in_4bit=False,  # No quantization for best quality
        attn_implementation="flash_attention_2",  # Flash Attention 2 for speed
    )

    # Check if full fine-tuning is requested
    if args.use_full_finetuning:
        logger.info("Using FULL fine-tuning (all parameters trainable) on A100 80GB")
        logger.info("This will provide the best model quality but requires significant VRAM")
        # All parameters already trainable by default
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="qwen2.5",
        )
        return model, tokenizer
    
    # Otherwise use LoRA for parameter-efficient fine-tuning
    target_modules = find_all_linear_names(model)
    logger.info(f"Target Modules: {str(target_modules)}")
    logger.info(f"Training with LoRA (rank={args.rank}, alpha={args.alpha}) in bfloat16 on A100 80GB")

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        target_modules=target_modules,
        lora_alpha=args.alpha,
        lora_dropout=0.0,  # No dropout for maximum capacity with high rank
        bias="none",
        use_gradient_checkpointing=False,  # Disabled for speed on A100
        random_state=args.seed,
        use_rslora=True,
        loftq_config=None,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen2.5",
    )

    return model, tokenizer

def setup_trainer(model, tokenizer, train_dataset, eval_dataset, args, logger):
    num_training_steps = len(train_dataset) * args.epochs // (args.batch_size * args.gradient_accumulation_steps)
    logger.info(f"Total training steps: {num_training_steps}")
    
    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,  # Larger eval batch
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=max(100, num_training_steps // 10),  # More warmup for stability
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=False,  # Force bfloat16 for A100
        bf16=True,
        bf16_full_eval=True,  # Use bf16 for evaluation too
        logging_steps=args.logging_steps,
        optim="adamw_torch_fused",  # Fastest optimizer for A100
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        seed=args.seed,
        output_dir=args.model_output_dir,
        report_to="neptune",
        save_strategy="steps",
        save_steps=max(50, num_training_steps // 10),  # More frequent saves
        eval_strategy="steps",
        eval_steps=max(50, num_training_steps // 10),
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        save_total_limit=5,  # Keep more checkpoints with A100 storage
        greater_is_better=False,
        eval_accumulation_steps=1,
        logging_first_step=True,
        dataloader_num_workers=4,  # Parallel data loading
        dataloader_pin_memory=True,  # Pin memory for faster transfer
        gradient_checkpointing=False,  # Disabled for speed
        ddp_find_unused_parameters=False,
        group_by_length=False,  # Can enable if variable length samples
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=args.num_proc,
        packing=False,
        args=training_args,
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    return trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=32768, help="Longer context for A100")
    parser.add_argument("--batch_size", type=int, default=8, help="Large batch for A100 80GB")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Effective batch size = 16")
    parser.add_argument("--epochs", type=int, default=5, help="More epochs for better convergence")
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Higher LR with larger batches")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=5, help="Frequent logging")
    parser.add_argument("--num_proc", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--experiment_name", default="LLMxCPG-Q")
    parser.add_argument("--output_dir", type=str, default="./models")
    parser.add_argument("--eval_split_ratio", type=float, default=0.1)
    parser.add_argument("--rank", type=int, default=128, help="High rank for A100 - maximum capacity")
    parser.add_argument("--alpha", type=int, default=256, help="LoRA alpha parameter - 2x rank")
    parser.add_argument("--use_full_finetuning", action="store_true", help="Use full fine-tuning instead of LoRA (requires more memory but best quality)")
    args = parser.parse_args()
    args.model_output_dir = f"{args.output_dir}/{args.model_name.split('/')[-1]}_{args.experiment_name}"
    return args

def main():
    args = parse_args()
    logger = setup_logging()

    logger.info(f"Loading model: {args.model_name}")
    model, tokenizer = setup_model_and_tokenizer(args, logger)

    logger.info(f"Loading dataset from: {args.dataset_path} and splitting for evaluation ({args.eval_split_ratio*100:.0f}/{100 - args.eval_split_ratio*100:.0f})")
    train_dataset, eval_dataset = load_json_dataset(args.dataset_path, args.eval_split_ratio)
    logger.info(f"Dataset loaded: {len(train_dataset)} training samples, {len(eval_dataset)} evaluation samples")

    train_dataset = standardize_sharegpt(train_dataset)
    eval_dataset = standardize_sharegpt(eval_dataset)

    train_dataset = train_dataset.map(
        lambda x: formatting_prompts_func(x, tokenizer),
        batched=True
    )
    eval_dataset = eval_dataset.map(
        lambda x: formatting_prompts_func(x, tokenizer),
        batched=True
    )

    logger.info("Setting up trainer")
    trainer = setup_trainer(model, tokenizer, train_dataset, eval_dataset, args, logger)

    logger.info("Starting training")
    trainer_stats = trainer.train()

    logger.info("Saving model")

    model.save_pretrained(args.model_output_dir)
    tokenizer.save_pretrained(args.model_output_dir)

    model.save_pretrained_merged(args.model_output_dir + "-vLLM", tokenizer, save_method = "merged_16bit",)

    logger.info("Training completed")
    return trainer_stats

if __name__ == "__main__":
    main()
