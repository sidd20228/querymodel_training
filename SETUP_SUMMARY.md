# 📦 LLMxCPG Benchmarking System - Complete Setup

## ✅ What Has Been Created

### 1. **Benchmark Dataset** (`benchmark/benchmark_dataset.json`)
   - ✅ 10 comprehensive vulnerability test cases
   - ✅ Covers diverse CWE categories (120, 416, 89, 190, 134, 22, 78, 415, 362, 476)
   - ✅ Each test includes: code, ground truth queries, expected vulnerabilities
   - ✅ Real-world vulnerability patterns

### 2. **Benchmark Execution Script** (`benchmark/run_benchmark.py`)
   - ✅ Loads and tests multiple models in parallel
   - ✅ Generates Joern CPG queries from vulnerable code
   - ✅ Executes queries using Joern CPG tool
   - ✅ Collects comprehensive metrics:
     - Query generation time
     - Query execution success rate
     - Vulnerability flow detection count
     - JSON validity checking
     - reachableByFlows usage detection
   - ✅ Generates outputs:
     - HTML interactive report
     - CSV summary files
     - Detailed JSON results per model

### 3. **Training Automation Script** (`training/train_all_models.sh`)
   - ✅ Trains all 3 models in sequence:
     - Qwen 2.5 7B Instruct
     - Qwen 2.5 Coder 14B
     - Qwen 2.5 Coder 32B
   - ✅ Optimized batch sizes for A100 80GB:
     - 7B: batch_size=8, grad_accum=2 (effective 16)
     - 14B: batch_size=4, grad_accum=4 (effective 16)
     - 32B: batch_size=2, grad_accum=8 (effective 16)
   - ✅ Automatic HuggingFace Hub push (optional)
   - ✅ Color-coded progress output
   - ✅ Comprehensive error handling

### 4. **HuggingFace Integration** (Updated `training/llmxcpg_query_finetune.py`)
   - ✅ Added CLI arguments:
     - `--push_to_hub`: Enable pushing to Hub
     - `--hf_repo_id`: Target repository ID
     - `--hf_token`: Authentication token
   - ✅ Dual repository strategy:
     - Pushes LoRA adapters to adapter repo
     - Pushes merged model to main repo
   - ✅ Automatic authentication and error handling
   - ✅ Progress logging

### 5. **Documentation**
   - ✅ [benchmark/README.md](benchmark/README.md): Complete benchmarking guide
   - ✅ [QUICKSTART.md](QUICKSTART.md): Step-by-step workflow
   - ✅ This file: Setup summary

### 6. **Setup Script** (`setup.sh`)
   - ✅ Interactive configuration wizard
   - ✅ Sets HuggingFace username and token
   - ✅ Updates training script automatically
   - ✅ Checks dependencies
   - ✅ Creates necessary directories
   - ✅ Validates Joern installation

## 🎯 Complete Workflow

### Option 1: Automated Setup (Recommended)
```bash
# Run setup wizard
./setup.sh

# This will:
# - Ask for HuggingFace username
# - Ask for HuggingFace token (optional)
# - Update all scripts with your username
# - Save token for automatic pushing
# - Check dependencies
# - Create directories
```

### Option 2: Manual Setup
```bash
# 1. Update HuggingFace username in training script
vim training/train_all_models.sh
# Replace "your-username" with your actual username

# 2. Set HuggingFace token
export HUGGING_FACE_TOKEN=hf_xxxxxxxxxx

# 3. Create directories
mkdir -p models benchmark/benchmark_results inference/results
```

## 🚀 Running the Complete Pipeline

### Step 1: Configure (One-Time)
```bash
./setup.sh
```

### Step 2: Train All Models
```bash
cd training
./train_all_models.sh
```

**This will:**
- Train Qwen 2.5 7B Instruct (~2-4 hours)
- Train Qwen 2.5 Coder 14B (~4-6 hours)
- Train Qwen 2.5 Coder 32B (~8-12 hours)
- Push all models to HuggingFace Hub (if token set)
- Save locally to `../models/`

### Step 3: Run Comprehensive Benchmark
```bash
cd ../benchmark
python run_benchmark.py \
    --models ../models/qwen2.5-7b-instruct-llmxcpg-query \
             ../models/qwen2.5-coder-14b-llmxcpg-query \
             ../models/qwen2.5-coder-32b-llmxcpg-query \
    --model_names "7B-Instruct" "14B-Coder" "32B-Coder" \
    --output_dir ./benchmark_results
```

**This will:**
- Test each model on 10 vulnerability types
- Generate Joern queries for each test
- Execute queries with Joern CPG
- Measure success rates and flow detection
- Generate comprehensive reports

### Step 4: View Results
```bash
# Open HTML report
open benchmark_results/benchmark_report.html

# View summaries
cat benchmark_results/benchmark_summary.csv
cat benchmark_results/per_test_comparison.csv
```

## 📊 Expected Outputs

### During Training
```
🤖 Training Model 1/3: Qwen 2.5 7B Instruct
========================================
📥 Loading base model...
✅ Model loaded successfully
🔧 Applying LoRA adapters...
🎯 Training started...
  Step 100: loss=0.234
  Step 200: loss=0.198
✅ Training complete!
📤 Pushing to HuggingFace Hub...
✅ Model pushed successfully!
```

### During Benchmarking
```
🤖 Benchmarking Model: 7B-Instruct
========================================
  📝 Test: buffer_overflow_01 - Buffer Overflow
     Generating queries... ✓ 3 queries (2.1s)
     Creating CPG... ✓
     Executing queries with Joern...
       Query 1/3... ✓ (0.8s, 5 flows)
       Query 2/3... ✓ (1.2s, 8 flows)
       Query 3/3... ✓ (0.9s, 2 flows)
     ✅ Success rate: 100.0%, Total flows: 15
```

### Benchmark Report Summary
```
Model          | Avg Gen | Success | Flows | w/ Flows | Valid JSON
7B-Instruct    | 1.8s    | 87.3%   | 142   | 8/10     | 9/10
14B-Coder      | 3.2s    | 92.1%   | 178   | 9/10     | 10/10
32B-Coder      | 5.7s    | 95.6%   | 203   | 10/10    | 10/10
```

## 📁 Final Directory Structure

```
llmxcpg/
├── benchmark/
│   ├── benchmark_dataset.json          ✅ 10 test cases
│   ├── run_benchmark.py                ✅ Benchmark script
│   ├── README.md                       ✅ Documentation
│   └── benchmark_results/
│       ├── benchmark_report.html       ← Generated
│       ├── benchmark_summary.csv       ← Generated
│       ├── per_test_comparison.csv     ← Generated
│       ├── 7B-Instruct_results.json    ← Generated
│       ├── 14B-Coder_results.json      ← Generated
│       └── 32B-Coder_results.json      ← Generated
├── training/
│   ├── llmxcpg_query_finetune.py       ✅ Updated with HF push
│   ├── train_all_models.sh             ✅ Automated training
│   └── README.md
├── models/
│   ├── qwen2.5-7b-instruct-llmxcpg-query/    ← Generated
│   ├── qwen2.5-coder-14b-llmxcpg-query/      ← Generated
│   └── qwen2.5-coder-32b-llmxcpg-query/      ← Generated
├── inference/
│   ├── query_inference.py              ← Existing
│   └── quick_test.py                   ← Existing
├── setup.sh                             ✅ Setup wizard
├── QUICKSTART.md                        ✅ Quick guide
└── SETUP_SUMMARY.md                     ✅ This file
```

## 🔧 System Requirements

### Hardware
- ✅ NVIDIA A100 80GB GPU (as per your setup)
- ✅ 200GB+ disk space for models
- ✅ 32GB+ RAM

### Software
- ✅ Python 3.8+
- ✅ CUDA 11.8+
- ✅ Joern CPG tool

### Python Packages
```bash
pip install torch transformers unsloth pandas huggingface_hub
```

## 🎓 Model Configurations

### Qwen 2.5 7B Instruct
- Base: `unsloth/Qwen2.5-7B-Instruct`
- Batch size: 8, Gradient accumulation: 2
- LoRA rank: 128, alpha: 256
- Training time: ~2-4 hours
- Best for: Production deployment (fast + accurate)

### Qwen 2.5 Coder 14B
- Base: `Qwen/Qwen2.5-Coder-14B-Instruct`
- Batch size: 4, Gradient accumulation: 4
- LoRA rank: 128, alpha: 256
- Training time: ~4-6 hours
- Best for: Balanced performance

### Qwen 2.5 Coder 32B
- Base: `Qwen/Qwen2.5-Coder-32B-Instruct`
- Batch size: 2, Gradient accumulation: 8
- LoRA rank: 128, alpha: 256
- Training time: ~8-12 hours
- Best for: Research (highest accuracy)

## 📈 Benchmark Metrics Explained

### Generation Metrics
- **Generation Time**: Time to generate queries (lower = faster)
- **Query Count**: Number of queries generated per test
- **Valid JSON**: Whether output is parseable JSON

### Execution Metrics
- **Success Rate**: % of queries that execute without errors
- **Flow Count**: Number of vulnerability flows detected
- **Has reachableByFlows**: Whether queries use data flow analysis

### Quality Indicators
- Success rate >80%: Good model
- Flow count: Higher = better vulnerability detection
- reachableByFlows usage: Critical for vulnerability detection

## 🐛 Troubleshooting

### "Joern not found"
```bash
# Install Joern
wget https://github.com/joernio/joern/releases/latest/download/joern-install.sh
chmod +x joern-install.sh
./joern-install.sh

# Add to PATH
export PATH=$PATH:~/joern/joern-cli

# Or specify path explicitly
python run_benchmark.py --joern_path /path/to/joern ...
```

### "Out of memory during training"
```bash
# For 32B model, reduce batch size
# Edit train_all_models.sh line for 32B:
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16
```

### "HuggingFace push failed"
```bash
# Login to HuggingFace
huggingface-cli login

# Or set token
export HUGGING_FACE_TOKEN=hf_xxxxx

# Verify token
huggingface-cli whoami
```

### "Training stops at 1 iteration"
This is expected behavior when `max_steps=0` (trains full dataset).
If training actually stops early, check:
```python
# In llmxcpg_query_finetune.py
# Make sure max_steps is 0 or removed
trainer = SFTTrainer(
    ...
    # max_steps=0,  # Should be 0 or commented out
    ...
)
```

## 📚 Next Steps After Benchmarking

1. **Analyze Results**: Review HTML report and identify best model
2. **Deploy Best Model**: Use for production inference
3. **Fine-tune**: Adjust hyperparameters based on results
4. **Expand Tests**: Add more vulnerability types to benchmark
5. **Share Results**: Push models and report to HuggingFace

## 🤝 Contributing

To add test cases:
1. Edit `benchmark/benchmark_dataset.json`
2. Add entry with: id, name, CWE, code, ground_truth_queries
3. Test manually with Joern first
4. Run benchmark and verify results

## 📧 Support

- Benchmark issues: Check [benchmark/README.md](benchmark/README.md)
- Training issues: Check [training/README.md](training/README.md)
- Quick reference: See [QUICKSTART.md](QUICKSTART.md)

## ✅ Verification Checklist

Before starting, verify:
- [ ] Setup script run: `./setup.sh`
- [ ] HuggingFace token set
- [ ] Training dataset exists: `data/llmxcpg_query_train.json`
- [ ] Joern installed and in PATH
- [ ] A100 GPU available
- [ ] 200GB+ disk space
- [ ] Python dependencies installed

## 🎉 Ready to Go!

Everything is set up and ready. Run:

```bash
# Quick start
./setup.sh              # Configure
cd training             # Navigate
./train_all_models.sh   # Train (12-24 hours total)
cd ../benchmark         # Navigate
python run_benchmark.py --models ../models/qwen* --model_names 7B 14B 32B
open benchmark_results/benchmark_report.html
```

**Total pipeline time:** ~15-30 hours (training + benchmarking)

Good luck with your LLMxCPG benchmarking! 🚀
