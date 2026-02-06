# LLMxCPG Model Benchmarking System

Comprehensive benchmarking framework for evaluating multiple LLMxCPG models on vulnerability detection and Joern CPG query generation.

## 📋 Overview

This benchmarking system tests fine-tuned models on 10 diverse vulnerability patterns and validates generated queries using Joern CPG. It supports:

- ✅ Multiple model comparison (7B, 14B, 32B)
- ✅ Automated Joern query execution
- ✅ Comprehensive metrics collection
- ✅ HTML report generation
- ✅ Per-test and aggregate statistics

## 🎯 Benchmark Dataset

The `benchmark_dataset.json` contains 10 carefully selected vulnerability test cases covering:

1. **Buffer Overflow (CWE-120)** - `gets()` usage
2. **Use After Free (CWE-416)** - Memory use after deallocation
3. **SQL Injection (CWE-89)** - Direct query concatenation
4. **Integer Overflow (CWE-190)** - Arithmetic without bounds checking
5. **Format String (CWE-134)** - User-controlled format strings
6. **Path Traversal (CWE-22)** - Directory traversal attack
7. **Command Injection (CWE-78)** - Shell command injection
8. **Double Free (CWE-415)** - Multiple free() calls
9. **TOCTOU Race Condition (CWE-362)** - Time-of-check to time-of-use
10. **Null Pointer Dereference (CWE-476)** - Null pointer usage

Each test case includes:
- Vulnerable C code snippet
- Ground truth Joern CPG queries
- Expected vulnerability description
- CWE classification

## 🚀 Quick Start

### Prerequisites

1. **Joern Installation**
   ```bash
   # Download and install Joern
   wget https://github.com/joernio/joern/releases/latest/download/joern-install.sh
   chmod +x joern-install.sh
   ./joern-install.sh
   
   # Add to PATH
   export PATH=$PATH:~/joern/joern-cli
   ```

2. **Python Dependencies**
   ```bash
   pip install torch transformers unsloth pandas
   ```

### Training All Models

```bash
cd training

# Set HuggingFace token (optional, for pushing models)
export HUGGING_FACE_TOKEN=your_token_here

# Train all three models
chmod +x train_all_models.sh
./train_all_models.sh
```

This will train:
- Qwen 2.5 7B Instruct
- Qwen 2.5 Coder 14B
- Qwen 2.5 Coder 32B

And optionally push them to HuggingFace Hub.

### Running Benchmark

```bash
cd benchmark

# Option 1: Benchmark all trained models
python run_benchmark.py \
    --models ../models/qwen2.5-7b-instruct-llmxcpg-query \
             ../models/qwen2.5-coder-14b-llmxcpg-query \
             ../models/qwen2.5-coder-32b-llmxcpg-query \
    --model_names "7B-Instruct" "14B-Coder" "32B-Coder" \
    --output_dir ./benchmark_results

# Option 2: Benchmark specific models
python run_benchmark.py \
    --models path/to/your/model \
    --model_names "Your-Model" \
    --benchmark_data ./benchmark_dataset.json \
    --joern_path /path/to/joern  # Optional if in PATH
```

### View Results

```bash
# Open HTML report
open benchmark_results/benchmark_report.html

# View CSV summaries
cat benchmark_results/benchmark_summary.csv
cat benchmark_results/per_test_comparison.csv

# View detailed JSON results
cat benchmark_results/7B-Instruct_results.json
```

## 📊 Metrics Collected

For each model, the benchmark collects:

### Query Generation Metrics
- **Generation Time**: Time to generate queries (seconds)
- **Query Count**: Number of queries generated per test
- **JSON Validity**: Whether output is valid JSON format
- **Has reachableByFlows**: Whether queries use data flow analysis

### Query Execution Metrics
- **Success Rate**: Percentage of queries that execute without errors
- **Flow Count**: Number of vulnerability flows detected
- **Execution Time**: Time to execute each query
- **Error Details**: Detailed error messages for failed queries

### Aggregate Metrics
- **Average Generation Time**: Mean time across all tests
- **Average Success Rate**: Mean query execution success rate
- **Total Flows Found**: Sum of all detected vulnerability flows
- **Tests with reachableByFlows**: Count of tests using flow analysis

## 📁 Output Structure

```
benchmark_results/
├── benchmark_report.html          # Interactive HTML report
├── benchmark_summary.csv          # Model comparison summary
├── per_test_comparison.csv        # Per-test breakdown
├── 7B-Instruct_results.json       # Detailed results for 7B model
├── 14B-Coder_results.json         # Detailed results for 14B model
└── 32B-Coder_results.json         # Detailed results for 32B model
```

## 🔧 Advanced Usage

### Custom Benchmark Dataset

Create your own benchmark dataset:

```json
[
  {
    "id": "custom_01",
    "name": "Custom Vulnerability",
    "cwe": "CWE-XXX",
    "instruction": "Analyze this code for vulnerabilities...",
    "code": "// Your C code here",
    "ground_truth_queries": [
      "// Expected Joern query"
    ],
    "expected_vulnerability": "Description of vulnerability"
  }
]
```

Then run:
```bash
python run_benchmark.py \
    --benchmark_data ./custom_benchmark.json \
    --models your/model/path
```

### Benchmark Single Model

```bash
python run_benchmark.py \
    --models ./path/to/single/model \
    --model_names "MyModel" \
    --output_dir ./my_results
```

### Use Custom System Prompt

```bash
python run_benchmark.py \
    --models ./path/to/model \
    --system_prompt_path ./my_custom_prompt.txt
```

### Skip HuggingFace Push During Training

```bash
# Don't set HUGGING_FACE_TOKEN
./train_all_models.sh
```

## 🐛 Troubleshooting

### Joern Not Found

```bash
# Check if Joern is in PATH
which joern

# If not found, specify path explicitly
python run_benchmark.py \
    --models ./model \
    --joern_path /full/path/to/joern
```

### Out of Memory During Training

For the 32B model, reduce batch size:

```python
# In train_all_models.sh, modify:
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16
```

### Query Execution Timeouts

Increase timeout in `run_benchmark.py`:

```python
# Line ~252
result = subprocess.run(
    [...],
    timeout=120  # Increase from 60 to 120 seconds
)
```

## 📈 Understanding Results

### Success Rate
- **90-100%**: Excellent - queries are syntactically correct
- **70-90%**: Good - most queries execute successfully
- **50-70%**: Fair - some query syntax issues
- **<50%**: Poor - significant query generation problems

### Flow Count
- Higher is better - indicates more vulnerability paths detected
- Compare against ground truth queries
- Zero flows may indicate:
  - Incorrect query logic
  - Overly restrictive filters
  - CPG creation issues

### Generation Time
- Lower is better for production use
- Expected: 1-5 seconds per test case
- Larger models typically slower but more accurate

## 🔬 Research & Citation

This benchmarking system is based on the LLMxCPG methodology:

```
@inproceedings{llmxcpg2025,
  title={LLMxCPG: Large Language Models for Vulnerability Detection via Code Property Graphs},
  booktitle={USENIX Security},
  year={2025}
}
```

## 📝 Model Training Configuration

### Qwen 2.5 7B Instruct
- Batch size: 8
- Gradient accumulation: 2
- Effective batch: 16
- LoRA rank: 128
- Training time: ~2-4 hours on A100

### Qwen 2.5 Coder 14B
- Batch size: 4
- Gradient accumulation: 4
- Effective batch: 16
- LoRA rank: 128
- Training time: ~4-6 hours on A100

### Qwen 2.5 Coder 32B
- Batch size: 2
- Gradient accumulation: 8
- Effective batch: 16
- LoRA rank: 128
- Training time: ~8-12 hours on A100

All models use:
- Learning rate: 2e-4
- Max sequence length: 32768
- Precision: bfloat16
- Optimizer: AdamW 8-bit
- Epochs: 5

## 🤝 Contributing

To add new test cases to the benchmark:

1. Add entry to `benchmark_dataset.json`
2. Include ground truth queries
3. Test manually with Joern first
4. Submit PR with results

## 📧 Support

For issues or questions:
- Check troubleshooting section above
- Review Joern documentation
- Open GitHub issue with benchmark results

## ⚖️ License

Same license as parent LLMxCPG project.
