# LLMxCPG - Training


## LLMxCPG-Q Fine-tuning:
### Usage
Run the script from your terminal using the following command structure:

```
python llmxcpg_query_finetune.py --model_name <model_id> --dataset_path <path_to_dataset.json> --experiment_name <your_experiment_name> [optional_arguments]
```

**Arguments**  
- `--model_name` (required): Hugging Face model ID to fine-tune (e.g., `unsloth/mistral-7b-instruct-v0.2`).

- `--dataset_path` (required): Path to your training dataset JSON file.

- `--experiment_name` (required): A name for your experiment, used in the output directory.

- `--max_seq_length` (optional, default: 25000): Maximum sequence length for training.

- `--batch_size` (optional, default: 4): Batch size per device.

- `--gradient_accumulation_steps` (optional, default: 1): Number of gradient accumulation steps.

- `--epochs` (optional, default: 1): Number of training epochs.

- `--max_steps` (optional, default: 0): Maximum number of training steps (0 means train for epochs).

- `--learning_rate` (optional, default: 1e-4): Learning rate for the optimizer.

- `--weight_decay` (optional, default: 0.01): Weight decay for the optimizer.

- `--logging_steps` (optional, default: 2): Log training progress every N steps.

- `--num_proc` (optional, default: 16): Number of processes to use for dataset mapping.

- `--seed` (optional, default: 1337): Random seed for reproducibility.

- `--output_dir` (optional, default: ./models): Base directory to save the trained model.

- `--eval_split_ratio` (optional, default: 0.1): Ratio of the dataset to use for evaluation.

- `--rank` (optional, default: 8): LoRA attention dimension (rank).

- `--alpha` (optional, default: 8): LoRA alpha parameter.

**Output**  
The trained model and tokenizer will be saved in a directory under `output_dir` named after the model and experiment name (`<output_dir>/<model_name>_<experiment_name>`). A merged 16-bit version compatible with vLLM will also be saved. Training logs will be written to `llmxcpg_query_finetune.log`.


## LLMxCPG-D Fine-tuning:
### Usage

Run the script from the command line, providing the paths to your dataset and specifying experiment parameters.

```bash
python llmxcpg_detect_finetune.py \
  --dataset_path /path/to/your/dataset.json \
  --system_prompt_path ./system_prompt.txt \
  --model_name unsloth/Qwen2.5-Coder-1.5B-Instruct \
  --epochs 3 \
  --lora_r 64 \
  --lora_alpha 128 \
  --batch_size 4 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-4 \
  --eval_split_ratio 0.05 \
  --num_proc 8
```

**Arguments**  
- `--dataset_path` (required): Path to the input JSON dataset file.  

- `--system_prompt_path `(default: ./../prompts/system_prompt.txt): Path to the system prompt text file.  

- `--model_name`: Name or path of the pretrained model to fine-tune.  

- `--output_dir` (default: ./models): Base directory to save trained models. A subdirectory will be created here based on the model name and experiment name.  

- `--max_seq_length` (default: 8128): Maximum sequence length for training.  

- `--batch_size `(default: 4): Training batch size per device.  

- `--gradient_accumulation_steps` (default: 2): Number of updates steps to accumulate gradients before performing a backward/update pass.  

- `--learning_rate` (default: 2e-4): The initial learning rate for the optimizer.  

- `--epochs` (default: 1): Total number of training epochs to perform.  

- `--seed` (default: 1337): Random seed for reproducibility.  

- `--eval_split_ratio` (default: 0.0001): Proportion of the dataset to include in the evaluation split.  

- `--lora_r` (default: 32): LoRA attention dimension (rank).  

- `--lora_alpha` (default: 64): Alpha parameter for LoRA scaling.  

- `--lora_dropout` (default: 0.05): Dropout probability for LoRA layers.  

- `--num_proc` (default: 16): Number of processes to use for data processing (mapping).  

- `--experiment_name` (default: LLMxCPG-D): Name of the experiment. Used for logging and creating the output directory name.  

**Output**  
The trained model and tokenizer will be saved in a directory under `output_dir` named after the model and experiment name (`<output_dir>/<model_name>_<experiment_name>`). A merged 16-bit version compatible with vLLM will also be saved. Training logs will be written to `llmxcpg_detect_finetune.log`.
