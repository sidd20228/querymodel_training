#!/usr/bin/env python3
"""
LLMxCPG Model Benchmarking Script
Tests multiple fine-tuned models on vulnerability detection and query generation.
Executes generated CPG queries using Joern and collects comprehensive metrics.
"""

import json
import subprocess
import tempfile
import shutil
import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import torch
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class QueryResult:
    """Results from executing a Joern query"""
    query: str
    success: bool
    output: str
    error: str
    execution_time: float
    flow_count: int  # Number of vulnerability flows found


@dataclass
class BenchmarkResult:
    """Complete benchmark result for one test case"""
    test_id: str
    model_name: str
    code_snippet: str
    generated_output: str
    parsed_queries: List[str]
    query_results: List[QueryResult]
    generation_time: float
    total_flows_found: int
    success_rate: float  # % of queries that executed successfully
    has_reachable_by_flows: bool
    is_valid_json: bool


def load_system_prompt(prompt_path: str = "../prompts/query_system_prompt.txt") -> str:
    """Load the system prompt from file."""
    prompt_file = Path(prompt_path)
    if not prompt_file.exists():
        return """You are a specialized assistant focused on generating Joern Code Property Graph Query Language (CPGQL) queries."""
    
    with open(prompt_file, 'r') as f:
        return f.read().strip()


def load_model(model_path: str, max_seq_length: int = 32768):
    """Load the fine-tuned model and tokenizer."""
    print(f"📥 Loading model from: {model_path}")
    
    if UNSLOTH_AVAILABLE:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=torch.bfloat16,
            load_in_4bit=False,
        )
        FastLanguageModel.for_inference(model)
        print("✅ Loaded with Unsloth (2x faster inference)")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()
        print("✅ Loaded with Transformers")
    
    return model, tokenizer


def generate_query(
    model,
    tokenizer,
    code_snippet: str,
    instruction: str,
    system_prompt: str,
    temperature: float = 0.1,
    max_new_tokens: int = 2048,
) -> Tuple[str, float]:
    """Generate Joern CPG query and return output with generation time."""
    
    start_time = time.time()
    
    # Format the prompt
    user_content = f"{system_prompt}\n{instruction}\n\n## Code snippet:\n```\n{code_snippet}\n```"
    
    # Create conversation format
    messages = [
        {"role": "user", "content": user_content}
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32768)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            repetition_penalty=1.1,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generation_time = time.time() - start_time
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    if "<|im_start|>assistant" in generated_text:
        response = generated_text.split("<|im_start|>assistant")[-1].strip()
    else:
        response = generated_text[len(prompt):].strip()
    
    return response, generation_time


def parse_cpg_queries(generated_text: str) -> Tuple[List[str], bool]:
    """Parse CPG queries from generated text."""
    import re
    
    try:
        # Try to extract JSON from markdown code block
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', generated_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            data = json.loads(json_str)
            if "queries" in data:
                return data["queries"], True
        
        # Try to find queries directly in text
        queries_match = re.search(r'"queries":\s*(\[.*?\])', generated_text, re.DOTALL)
        if queries_match:
            queries_str = queries_match.group(1)
            queries = json.loads(queries_str)
            return queries, True
        
        return [], False
    except Exception as e:
        return [], False


def setup_joern(joern_path: Optional[str] = None) -> str:
    """Setup Joern and return the path to joern executable."""
    if joern_path and Path(joern_path).exists():
        return joern_path
    
    # Check if joern is in PATH
    try:
        result = subprocess.run(['which', 'joern'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    
    # Check common installation paths
    common_paths = [
        '/usr/local/bin/joern',
        os.path.expanduser('~/joern/joern'),
        os.path.expanduser('~/bin/joern'),
    ]
    
    for path in common_paths:
        if Path(path).exists():
            return path
    
    raise FileNotFoundError("Joern not found. Please install Joern or provide path with --joern_path")


def create_cpg(joern_path: str, code_file: str) -> Tuple[bool, str, str]:
    """Create CPG from C code file using Joern."""
    try:
        # Create CPG
        result = subprocess.run(
            [joern_path + '-parse', code_file],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            return True, result.stdout, ""
        else:
            return False, result.stdout, result.stderr
            
    except subprocess.TimeoutExpired:
        return False, "", "Timeout creating CPG"
    except Exception as e:
        return False, "", str(e)


def execute_joern_query(joern_path: str, cpg_path: str, query: str) -> QueryResult:
    """Execute a single Joern query on the CPG."""
    start_time = time.time()
    
    try:
        # Write query to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sc', delete=False) as f:
            f.write(query)
            query_file = f.name
        
        # Execute query
        result = subprocess.run(
            [joern_path, '--script', query_file, '--cpg', cpg_path],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        execution_time = time.time() - start_time
        
        # Count flows (lines that look like flow results)
        flow_count = 0
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            # Count lines that contain "List(" or flow-like patterns
            flow_count = sum(1 for line in lines if 'List(' in line or '==>' in line)
        
        os.unlink(query_file)
        
        return QueryResult(
            query=query,
            success=result.returncode == 0,
            output=result.stdout,
            error=result.stderr,
            execution_time=execution_time,
            flow_count=flow_count
        )
        
    except subprocess.TimeoutExpired:
        return QueryResult(
            query=query,
            success=False,
            output="",
            error="Query execution timeout",
            execution_time=60.0,
            flow_count=0
        )
    except Exception as e:
        return QueryResult(
            query=query,
            success=False,
            output="",
            error=str(e),
            execution_time=time.time() - start_time,
            flow_count=0
        )


def run_benchmark_test(
    model,
    tokenizer,
    test_case: Dict,
    model_name: str,
    system_prompt: str,
    joern_path: str,
    temp_dir: str
) -> BenchmarkResult:
    """Run complete benchmark test for one code snippet."""
    
    test_id = test_case['id']
    code = test_case['code']
    instruction = test_case['instruction']
    
    print(f"\n  📝 Test: {test_id} - {test_case['name']}")
    
    # Generate queries
    print(f"     Generating queries...")
    generated_output, gen_time = generate_query(
        model, tokenizer, code, instruction, system_prompt
    )
    
    # Parse queries
    queries, is_json = parse_cpg_queries(generated_output)
    print(f"     ✓ Generated {len(queries)} queries in {gen_time:.2f}s")
    
    if not queries:
        print(f"     ⚠️  No queries parsed")
        return BenchmarkResult(
            test_id=test_id,
            model_name=model_name,
            code_snippet=code,
            generated_output=generated_output,
            parsed_queries=[],
            query_results=[],
            generation_time=gen_time,
            total_flows_found=0,
            success_rate=0.0,
            has_reachable_by_flows=False,
            is_valid_json=is_json
        )
    
    # Create CPG from code
    print(f"     Creating CPG...")
    code_file = os.path.join(temp_dir, f"{test_id}.c")
    with open(code_file, 'w') as f:
        f.write(code)
    
    cpg_success, cpg_out, cpg_err = create_cpg(joern_path, code_file)
    
    if not cpg_success:
        print(f"     ❌ CPG creation failed")
        return BenchmarkResult(
            test_id=test_id,
            model_name=model_name,
            code_snippet=code,
            generated_output=generated_output,
            parsed_queries=queries,
            query_results=[],
            generation_time=gen_time,
            total_flows_found=0,
            success_rate=0.0,
            has_reachable_by_flows=any('reachableByFlows' in q for q in queries),
            is_valid_json=is_json
        )
    
    cpg_path = code_file + ".bin"  # Joern creates CPG with .bin extension
    
    # Execute queries
    print(f"     Executing queries with Joern...")
    query_results = []
    for i, query in enumerate(queries, 1):
        print(f"       Query {i}/{len(queries)}...", end='')
        result = execute_joern_query(joern_path, cpg_path, query)
        query_results.append(result)
        status = "✓" if result.success else "✗"
        print(f" {status} ({result.execution_time:.2f}s, {result.flow_count} flows)")
    
    # Calculate metrics
    total_flows = sum(r.flow_count for r in query_results)
    success_count = sum(1 for r in query_results if r.success)
    success_rate = (success_count / len(queries) * 100) if queries else 0
    
    print(f"     ✅ Success rate: {success_rate:.1f}%, Total flows: {total_flows}")
    
    return BenchmarkResult(
        test_id=test_id,
        model_name=model_name,
        code_snippet=code,
        generated_output=generated_output,
        parsed_queries=queries,
        query_results=query_results,
        generation_time=gen_time,
        total_flows_found=total_flows,
        success_rate=success_rate,
        has_reachable_by_flows=any('reachableByFlows' in q for q in queries),
        is_valid_json=is_json
    )


def benchmark_model(
    model_path: str,
    model_name: str,
    benchmark_data: List[Dict],
    system_prompt: str,
    joern_path: str,
    output_dir: str
) -> List[BenchmarkResult]:
    """Benchmark a single model on all test cases."""
    
    print(f"\n{'='*80}")
    print(f"🤖 Benchmarking Model: {model_name}")
    print(f"{'='*80}")
    
    # Load model
    model, tokenizer = load_model(model_path)
    
    # Create temp directory for code files
    temp_dir = tempfile.mkdtemp(prefix=f"llmxcpg_bench_{model_name}_")
    
    results = []
    
    try:
        for test_case in benchmark_data:
            result = run_benchmark_test(
                model, tokenizer, test_case, model_name,
                system_prompt, joern_path, temp_dir
            )
            results.append(result)
        
        # Save detailed results
        model_output_file = os.path.join(output_dir, f"{model_name}_results.json")
        with open(model_output_file, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2, default=str)
        
        print(f"\n✅ Model benchmark complete. Results saved to {model_output_file}")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return results


def generate_summary_report(all_results: Dict[str, List[BenchmarkResult]], output_dir: str):
    """Generate comprehensive summary report comparing all models."""
    
    print(f"\n{'='*80}")
    print(f"📊 BENCHMARK SUMMARY")
    print(f"{'='*80}\n")
    
    # Summary statistics
    summary_data = []
    
    for model_name, results in all_results.items():
        avg_gen_time = sum(r.generation_time for r in results) / len(results)
        avg_success_rate = sum(r.success_rate for r in results) / len(results)
        total_flows = sum(r.total_flows_found for r in results)
        queries_with_reachable = sum(1 for r in results if r.has_reachable_by_flows)
        valid_json_count = sum(1 for r in results if r.is_valid_json)
        
        summary_data.append({
            'Model': model_name,
            'Avg Gen Time (s)': f"{avg_gen_time:.2f}",
            'Avg Success Rate (%)': f"{avg_success_rate:.1f}",
            'Total Flows Found': total_flows,
            'Tests w/ reachableByFlows': f"{queries_with_reachable}/{len(results)}",
            'Valid JSON Format': f"{valid_json_count}/{len(results)}"
        })
    
    # Create DataFrame
    df = pd.DataFrame(summary_data)
    
    print(df.to_string(index=False))
    print()
    
    # Save to CSV
    summary_file = os.path.join(output_dir, "benchmark_summary.csv")
    df.to_csv(summary_file, index=False)
    print(f"✅ Summary saved to {summary_file}")
    
    # Detailed comparison per test case
    print(f"\n{'='*80}")
    print(f"📈 PER-TEST COMPARISON")
    print(f"{'='*80}\n")
    
    # Get test IDs
    test_ids = [r.test_id for r in list(all_results.values())[0]]
    
    comparison_data = []
    for test_id in test_ids:
        row = {'Test ID': test_id}
        for model_name, results in all_results.items():
            result = next(r for r in results if r.test_id == test_id)
            row[f'{model_name} - Flows'] = result.total_flows_found
            row[f'{model_name} - Success %'] = f"{result.success_rate:.0f}%"
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    comparison_file = os.path.join(output_dir, "per_test_comparison.csv")
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\n✅ Comparison saved to {comparison_file}")
    
    # Generate HTML report
    html_report = generate_html_report(all_results, df, comparison_df)
    html_file = os.path.join(output_dir, "benchmark_report.html")
    with open(html_file, 'w') as f:
        f.write(html_report)
    print(f"✅ HTML report saved to {html_file}")


def generate_html_report(all_results: Dict, summary_df: pd.DataFrame, comparison_df: pd.DataFrame) -> str:
    """Generate HTML report with visualizations."""
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>LLMxCPG Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
        .section {{ background: white; padding: 20px; margin: 20px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #34495e; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .metric {{ display: inline-block; margin: 10px 20px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #3498db; }}
        .metric-label {{ color: #7f8c8d; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 LLMxCPG Model Benchmark Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>📊 Summary Statistics</h2>
        {summary_df.to_html(index=False)}
    </div>
    
    <div class="section">
        <h2>📈 Per-Test Comparison</h2>
        {comparison_df.to_html(index=False)}
    </div>
    
    <div class="section">
        <h2>📋 Detailed Results</h2>
"""
    
    for model_name, results in all_results.items():
        html += f"<h3>{model_name}</h3><ul>"
        for result in results:
            html += f"""
            <li><strong>{result.test_id}</strong>: 
                {len(result.parsed_queries)} queries, 
                {result.total_flows_found} flows, 
                {result.success_rate:.1f}% success rate
            </li>
            """
        html += "</ul>"
    
    html += """
    </div>
</body>
</html>
"""
    
    return html


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark LLMxCPG models")
    parser.add_argument("--models", nargs='+', required=True, help="Model paths or names")
    parser.add_argument("--model_names", nargs='+', help="Model display names (optional)")
    parser.add_argument("--benchmark_data", type=str, default="./benchmark_dataset.json")
    parser.add_argument("--joern_path", type=str, help="Path to Joern executable")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results")
    parser.add_argument("--system_prompt_path", type=str, default="../prompts/query_system_prompt.txt")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load benchmark data
    print(f"📖 Loading benchmark dataset: {args.benchmark_data}")
    with open(args.benchmark_data, 'r') as f:
        benchmark_data = json.load(f)
    
    print(f"✅ Loaded {len(benchmark_data)} test cases")
    
    # Load system prompt
    system_prompt = load_system_prompt(args.system_prompt_path)
    
    # Setup Joern
    print(f"\n🔧 Setting up Joern...")
    try:
        joern_path = setup_joern(args.joern_path)
        print(f"✅ Joern found at: {joern_path}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1
    
    # Generate model names if not provided
    if not args.model_names:
        model_names = [Path(m).name for m in args.models]
    else:
        model_names = args.model_names
    
    # Benchmark each model
    all_results = {}
    for model_path, model_name in zip(args.models, model_names):
        results = benchmark_model(
            model_path, model_name, benchmark_data,
            system_prompt, joern_path, args.output_dir
        )
        all_results[model_name] = results
    
    # Generate summary report
    generate_summary_report(all_results, args.output_dir)
    
    print(f"\n{'='*80}")
    print(f"🎉 BENCHMARK COMPLETE!")
    print(f"{'='*80}")
    print(f"\nResults saved to: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
