#!/bin/bash

# LLMxCPG - Train All Models Script
# Trains Qwen 2.5 7B Instruct, 14B Coder, and 32B Coder models

set -e  # Exit on error

# Configuration
DATASET="../data/llmxcpg_query_train.json"
OUTPUT_BASE="../models"
HF_TOKEN="${HUGGING_FACE_TOKEN:-}"  # Set via environment variable

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🚀 LLMxCPG Multi-Model Training Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"

# Check HuggingFace token
if [ -z "$HF_TOKEN" ]; then
    echo -e "${RED}⚠️  Warning: HUGGING_FACE_TOKEN not set. Models won't be pushed to Hub.${NC}"
    echo -e "Set it with: export HUGGING_FACE_TOKEN=your_token"
    PUSH_ARG=""
else
    echo -e "${GREEN}✅ HuggingFace token found${NC}"
    PUSH_ARG="--push_to_hub"
fi

# Model 1: Qwen 2.5 7B Instruct
echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}🤖 Training Model 1/3: Qwen 2.5 7B Instruct${NC}"
echo -e "${BLUE}========================================${NC}"

python llmxcpg_query_finetune.py \
    --model_name "unsloth/Qwen2.5-7B-Instruct" \
    --dataset_path "$DATASET" \
    --output_dir "${OUTPUT_BASE}/qwen2.5-7b-instruct-llmxcpg-query" \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 5 \
    --learning_rate 2e-4 \
    --warmup_steps 10 \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.0 \
    --max_seq_length 32768 \
    --logging_steps 1 \
    --save_steps 50 \
    --bf16 \
    $PUSH_ARG \
    --hf_repo_id "your-username/qwen2.5-7b-instruct-llmxcpg-query" \
    --hf_token "$HF_TOKEN"

echo -e "${GREEN}✅ Model 1 training complete!${NC}"

# Model 2: Qwen 2.5 Coder 14B
echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}🤖 Training Model 2/3: Qwen 2.5 Coder 14B${NC}"
echo -e "${BLUE}========================================${NC}"

python llmxcpg_query_finetune.py \
    --model_name "Qwen/Qwen2.5-Coder-14B-Instruct" \
    --dataset_path "$DATASET" \
    --output_dir "${OUTPUT_BASE}/qwen2.5-coder-14b-llmxcpg-query" \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 5 \
    --learning_rate 2e-4 \
    --warmup_steps 10 \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.0 \
    --max_seq_length 32768 \
    --logging_steps 1 \
    --save_steps 50 \
    --bf16 \
    $PUSH_ARG \
    --hf_repo_id "your-username/qwen2.5-coder-14b-llmxcpg-query" \
    --hf_token "$HF_TOKEN"

echo -e "${GREEN}✅ Model 2 training complete!${NC}"

# Model 3: Qwen 2.5 Coder 32B
echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}🤖 Training Model 3/3: Qwen 2.5 Coder 32B${NC}"
echo -e "${BLUE}========================================${NC}"

python llmxcpg_query_finetune.py \
    --model_name "Qwen/Qwen2.5-Coder-32B-Instruct" \
    --dataset_path "$DATASET" \
    --output_dir "${OUTPUT_BASE}/qwen2.5-coder-32b-llmxcpg-query" \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 5 \
    --learning_rate 2e-4 \
    --warmup_steps 10 \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.0 \
    --max_seq_length 32768 \
    --logging_steps 1 \
    --save_steps 50 \
    --bf16 \
    $PUSH_ARG \
    --hf_repo_id "your-username/qwen2.5-coder-32b-llmxcpg-query" \
    --hf_token "$HF_TOKEN"

echo -e "${GREEN}✅ Model 3 training complete!${NC}"

# Summary
echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}🎉 ALL MODELS TRAINED SUCCESSFULLY!${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "\nModels saved to:"
echo -e "  1. ${OUTPUT_BASE}/qwen2.5-7b-instruct-llmxcpg-query"
echo -e "  2. ${OUTPUT_BASE}/qwen2.5-coder-14b-llmxcpg-query"
echo -e "  3. ${OUTPUT_BASE}/qwen2.5-coder-32b-llmxcpg-query"

if [ -n "$HF_TOKEN" ]; then
    echo -e "\n${GREEN}✅ Models pushed to HuggingFace Hub${NC}"
fi

echo -e "\n${BLUE}Next steps:${NC}"
echo -e "1. Run benchmark: cd ../benchmark && python run_benchmark.py --models ../models/* --model_names 7B-Instruct 14B-Coder 32B-Coder"
echo -e "2. View results: open benchmark_results/benchmark_report.html"
echo -e "\n${GREEN}Happy benchmarking! 🚀${NC}"
