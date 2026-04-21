"""Quantization and edge deployment tools.

Supports:
- GGUF quantization (Q8, Q5, Q4) via llama.cpp
- Benchmarking on different hardware
- Pareto frontier analysis (accuracy vs latency vs power)
"""

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class QuantConfig:
    """Quantization configuration."""
    name: str
    bits: int
    gguf_type: str  # llama.cpp quantization type


QUANT_CONFIGS = {
    "q8": QuantConfig("Q8_0", 8, "q8_0"),
    "q5": QuantConfig("Q5_K_M", 5, "q5_k_m"),
    "q4": QuantConfig("Q4_K_M", 4, "q4_k_m"),
}


@click.group()
def cli():
    """Edge deployment tools."""
    pass


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--output", "-o", default="outputs/gguf", help="Output directory")
@click.option("--quant", "-q", multiple=True, default=["q8", "q5", "q4"], help="Quantization levels")
def quantize(model_path: str, output: str, quant: tuple):
    """Quantize model to GGUF format.
    
    Requires llama.cpp to be installed.
    """
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[bold]Quantizing {model_path}[/bold]")
    console.print(f"Levels: {', '.join(quant)}")
    
    # First convert to GGUF (F16)
    f16_path = output_dir / "model-f16.gguf"
    
    console.print("\n1. Converting to GGUF (F16)...")
    console.print("[yellow]Run manually:[/yellow]")
    console.print(f"  python llama.cpp/convert_hf_to_gguf.py {model_path} --outfile {f16_path}")
    
    # Then quantize to each level
    console.print("\n2. Quantizing to target levels...")
    for q in quant:
        if q not in QUANT_CONFIGS:
            console.print(f"[red]Unknown quantization: {q}[/red]")
            continue
        
        config = QUANT_CONFIGS[q]
        out_path = output_dir / f"model-{config.name}.gguf"
        
        console.print(f"\n[yellow]Run manually:[/yellow]")
        console.print(f"  llama.cpp/llama-quantize {f16_path} {out_path} {config.gguf_type}")
    
    console.print("\n[green]Quantization commands generated.[/green]")
    console.print("Install llama.cpp: git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make")


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--limit", "-n", default=50, help="Number of scenarios to benchmark")
@click.option("--device", default="cpu", help="Device (cpu, cuda, rpi5)")
def benchmark(model_path: str, data_path: str, limit: int, device: str):
    """Benchmark quantized model.
    
    Measures latency, throughput, and accuracy.
    """
    console.print(f"[bold]Benchmarking {model_path}[/bold]")
    console.print(f"Device: {device}")
    console.print(f"Scenarios: {limit}")
    
    # Load scenarios
    scenarios = []
    with open(data_path) as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            scenarios.append(json.loads(line))
    
    console.print(f"Loaded {len(scenarios)} scenarios")
    
    # Check if llama-cpp-python is available
    try:
        from llama_cpp import Llama
    except ImportError:
        console.print("[red]llama-cpp-python not installed[/red]")
        console.print("Install with: uv pip install llama-cpp-python")
        console.print("\n[yellow]Showing expected benchmark format:[/yellow]")
        
        # Show expected output format
        table = Table(title="Benchmark Results (placeholder)")
        table.add_column("Metric")
        table.add_column("Value")
        table.add_row("Model", model_path)
        table.add_row("Device", device)
        table.add_row("Scenarios", str(limit))
        table.add_row("Mean Latency", "TBD ms")
        table.add_row("P95 Latency", "TBD ms")
        table.add_row("Throughput", "TBD tok/s")
        table.add_row("Accuracy", "TBD %")
        console.print(table)
        return
    
    # Load model
    console.print("Loading model...")
    llm = Llama(
        model_path=model_path,
        n_ctx=512,
        n_threads=4 if device == "rpi5" else 8,
        verbose=False,
    )
    
    # Benchmark
    latencies = []
    correct = 0
    
    for item in scenarios:
        scenario = item["scenario"]
        ground_truth = item["decision"]
        
        # Format prompt
        vessels_text = []
        for i, v in enumerate(scenario["vessels"], 1):
            ais = "AIS" if v["ais_active"] else "NO_AIS"
            vessels_text.append(f"C{i}: {v['vessel_type']} {v['bearing']:.0f}°/{v['distance']:.1f}nm {v['speed']:.0f}kn {ais}")
        
        prompt = f"""Mission: {scenario['mission_type']}
Own: {scenario['own_heading']:.0f}°/{scenario['own_speed']:.0f}kn
Conditions: {scenario['weather']}, vis:{scenario['visibility']}, {scenario['time_of_day']}, comms:{scenario['comms_status']}
Contacts:
{chr(10).join(vessels_text)}

Respond with JSON: {{"threat_level": "...", "action": "...", "reasoning": "...", "confidence": 0.0}}"""
        
        start = time.perf_counter()
        output = llm(prompt, max_tokens=200, stop=["}"])
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)
        
        # Parse and check accuracy
        try:
            response = output["choices"][0]["text"] + "}"
            pred = json.loads(response)
            if pred.get("action") == ground_truth["action"]:
                correct += 1
        except:
            pass
    
    # Results
    latencies.sort()
    table = Table(title="Benchmark Results")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Model", model_path)
    table.add_row("Device", device)
    table.add_row("Scenarios", str(len(scenarios)))
    table.add_row("Mean Latency", f"{sum(latencies)/len(latencies):.1f} ms")
    table.add_row("P95 Latency", f"{latencies[int(0.95*len(latencies))]:.1f} ms")
    table.add_row("Accuracy", f"{correct/len(scenarios):.1%}")
    console.print(table)


@cli.command()
@click.argument("results_dir", type=click.Path(exists=True))
def pareto(results_dir: str):
    """Generate Pareto frontier analysis.
    
    Plots accuracy vs latency vs power for different quantization levels.
    """
    console.print("[yellow]Pareto analysis not yet implemented[/yellow]")
    console.print("Will generate: accuracy vs latency vs power tradeoff chart")
    console.print("\nExpected data points:")
    console.print("  - F16: highest accuracy, highest latency, highest power")
    console.print("  - Q8: ~same accuracy, lower latency, lower power")
    console.print("  - Q5: slight accuracy drop, much lower latency")
    console.print("  - Q4: more accuracy drop, lowest latency, fits on RPi 5")


if __name__ == "__main__":
    cli()
