# 🚀 LLMxCPG Quick Start Guide

## Complete Workflow: Train → Benchmark → Deploy

### Step 1: Setup Environment

```bash
# Clone repository
cd /Users/siddhantparashar/projects/llmxcpg

# Install dependencies
pip install -r training/requirements.txt
pip install pandas  # For benchmarking

# Install Joern (required for benchmarking)
wget https://github.com/joernio/joern/releases/latest/download/joern-install.sh
chmod +x joern-install.sh
./joern-install.sh

# Add Joern to PATH
export PATH=$PATH:~/joern/joern-cli
```

### Step 2: Prepare Training Data

Your training data is already at:
```
data/llmxcpg_query_train.json  # 1746 training samples
```

Format:
```json
{
  "instruction": "Analyze this code...",
  "input": "// C code",
  "output": "{\"queries\": [\"cpg.method.name(...)\"]}"
}
```

### Step 3: Train All Models (One Command!)

```bash
cd training

# Set your HuggingFace token (get from https://huggingface.co/settings/tokens)
export HUGGING_FACE_TOKEN=hf_xxxxxxxxxxxxx

# Train all 3 models and push to Hub
./train_all_models.sh
```

**What happens:**
- ✅ Trains Qwen 2.5 7B Instruct (~2-4 hours)
- ✅ Trains Qwen 2.5 Coder 14B (~4-6 hours)
- ✅ Trains Qwen 2.5 Coder 32B (~8-12 hours)
- ✅ Automatically pushes to HuggingFace Hub
- ✅ Saves LoRA adapters and merged models

**Models saved to:**
```
models/qwen2.5-7b-instruct-llmxcpg-query/
models/qwen2.5-coder-14b-llmxcpg-query/
models/qwen2.5-coder-32b-llmxcpg-query/
```

**HuggingFace Hub repos (update script with your username):**
```
your-username/qwen2.5-7b-instruct-llmxcpg-query
your-username/qwen2.5-coder-14b-llmxcpg-query
your-username/qwen2.5-coder-32b-llmxcpg-query
```

### Step 4: Run Comprehensive Benchmark

```bash
cd ../benchmark

# Benchmark all 3 trained models
python run_benchmark.py \
    --models ../models/qwen2.5-7b-instruct-llmxcpg-query \
             ../models/qwen2.5-coder-14b-llmxcpg-query \
             ../models/qwen2.5-coder-32b-llmxcpg-query \
    --model_names "7B-Instruct" "14B-Coder" "32B-Coder" \
    --output_dir ./benchmark_results
```

**What happens:**
- ✅ Tests each model on 10 vulnerability types
- ✅ Generates Joern CPG queries
- ✅ Executes queries with Joern
- ✅ Collects comprehensive metrics
- ✅ Generates HTML report + CSV summaries

**Estimated time:** 10-30 minutes (depends on model sizes)

### Step 5: View Results

```bash
# Open interactive HTML report
open benchmark_results/benchmark_report.html

# View summary in terminal
cat benchmark_results/benchmark_summary.csv

# Per-test comparison
cat benchmark_results/per_test_comparison.csv

# Detailed results for each model
cat benchmark_results/7B-Instruct_results.json
cat benchmark_results/14B-Coder_results.json
cat benchmark_results/32B-Coder_results.json
```

### Step 6: Use Best Model for Inference

```bash
cd ../inference

# Interactive mode
python query_inference.py \
    --model_path ../models/qwen2.5-coder-32b-llmxcpg-query \
    --mode interactive

# Single file
python query_inference.py \
    --model_path ../models/qwen2.5-coder-32b-llmxcpg-query \
    --mode single \
    --code_file vulnerable.c

# Batch processing
python query_inference.py \
    --model_path ../models/qwen2.5-coder-32b-llmxcpg-query \
    --mode batch \
    --input_dir ./code_samples/ \
    --output_dir ./results/
```

## 🎯 Benchmark Test Cases

Your 10 test cases cover:

1. **Buffer Overflow** (CWE-120) - `gets()` usage
2. **Use After Free** (CWE-416) - Memory use after free
3. **SQL Injection** (CWE-89) - Query concatenation
4. **Integer Overflow** (CWE-190) - Unchecked arithmetic
5. **Format String** (CWE-134) - User-controlled format
6. **Path Traversal** (CWE-22) - Directory traversal
7. **Command Injection** (CWE-78) - Shell injection
8. **Double Free** (CWE-415) - Multiple free() calls
9. **TOCTOU Race** (CWE-362) - Time-of-check race
10. **Null Pointer** (CWE-476) - Null dereference

## 📊 Expected Results

### Success Metrics
- **Query Generation:** All models should generate queries
- **Success Rate:** >80% for good models
- **Flow Detection:** Should find vulnerability flows
- **JSON Format:** >90% valid JSON output

### Performance Comparison
Expected ranking (best to worst):
1. 🥇 **32B Coder** - Best accuracy, slowest
2. 🥈 **14B Coder** - Good balance
3. 🥉 **7B Instruct** - Fastest, lower accuracy

## 🔧 Customization

### Train Single Model

```bash
cd training

python llmxcpg_query_finetune.py \
    --model_name "unsloth/Qwen2.5-7B-Instruct" \
    --dataset_path ../data/llmxcpg_query_train.json \
    --output_dir ../models/my-custom-model \
    --per_device_train_batch_size 8 \
    --num_train_epochs 5 \
    --push_to_hub \
    --hf_repo_id "your-username/my-model" \
    --hf_token "$HUGGING_FACE_TOKEN"
```

### Benchmark Single Model

```bash
cd benchmark

python run_benchmark.py \
    --models ../models/my-custom-model \
    --model_names "MyModel" \
    --output_dir ./my_results
```

### Add Custom Test Case

Edit `benchmark/benchmark_dataset.json`:

```json
{
  "id": "custom_01",
  "name": "My Vulnerability Test",
  "cwe": "CWE-XXX",
  "instruction": "Analyze this code for vulnerabilities and generate Joern CPG queries...",
  "code": "void vulnerable() { /* your C code */ }",
  "ground_truth_queries": [
    "// Expected Joern query"
  ],
  "expected_vulnerability": "Description of the vulnerability"
}
```

## 🐛 Troubleshooting

### Joern Not Found
```bash
# Check installation
which joern

# If not found, specify path
python run_benchmark.py --joern_path /full/path/to/joern ...
```

### Out of Memory
```bash
# Reduce batch size for 32B model
# Edit train_all_models.sh:
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16
```

### Training Stops Early
```bash
# Check max_steps in training script
# Should be 0 or commented out for full training
```

### HuggingFace Push Fails
```bash
# Verify token
huggingface-cli login

# Or set token directly
export HUGGING_FACE_TOKEN=hf_xxxxx
```

## 📈 Monitoring Training

```bash
# Watch training progress
tail -f training/training.log

# Check GPU usage
nvidia-smi -l 1

# TensorBoard (if enabled)
tensorboard --logdir models/qwen2.5-*/logs
```

## 🎓 Understanding Output

### Training Output
```
Step 100: loss=0.234, lr=1.8e-4
Saving checkpoint at step 100
✅ Epoch 1/5 complete
```

### Benchmark Output
```
🤖 Benchmarking Model: 7B-Instruct
  📝 Test: buffer_overflow_01
     Generating queries... ✓ 3 queries (2.1s)
     Executing with Joern... ✓ 100% (15 flows)
```

### Results Summary
```
Model          | Avg Success | Total Flows | Avg Time
7B-Instruct    | 87.3%       | 142         | 1.8s
14B-Coder      | 92.1%       | 178         | 3.2s
32B-Coder      | 95.6%       | 203         | 5.7s
```

## 📚 Next Steps

1. **Analyze Results:** Compare models in HTML report
2. **Fine-tune:** Adjust hyperparameters based on results
3. **Deploy:** Use best model in production
4. **Contribute:** Add more test cases to benchmark

## 🔗 Resources

- **Joern Docs:** https://docs.joern.io/
- **Unsloth Docs:** https://docs.unsloth.ai/
- **HuggingFace:** https://huggingface.co/docs
- **Paper:** See `usenixsecurity25-lekssays.pdf`

## ⚡ One-Liner Commands

```bash
# Complete pipeline
export HUGGING_FACE_TOKEN=hf_xxx && \
cd training && ./train_all_models.sh && \
cd ../benchmark && python run_benchmark.py \
    --models ../models/qwen* --model_names 7B 14B 32B && \
open benchmark_results/benchmark_report.html

# Quick test single model
cd inference && python query_inference.py \
    --model_path ../models/qwen2.5-7b-instruct-llmxcpg-query \
    --mode interactive

# Re-run benchmark on existing models
cd benchmark && python run_benchmark.py \
    --models ../models/qwen2.5-*-llmxcpg-query \
    --model_names 7B 14B 32B \
    --output_dir ./results_$(date +%Y%m%d)
```

## 💡 Pro Tips

1. **Training:** Run overnight on A100 for best results
2. **Benchmarking:** Test incrementally (1 model first)
3. **Debugging:** Use `quick_test.py` for fast validation
4. **Deployment:** 7B model is best for production (speed/accuracy)
5. **Research:** 32B model gives best research results

---

**Need help?** Check:
- [training/README.md](../training/README.md) - Training details
- [benchmark/README.md](./README.md) - Benchmarking guide
- [inference/README.md](../inference/README.md) - Inference usage
