"""HELM-inspired evaluation framework.

Evaluates models across multiple dimensions:
- Accuracy: correct threat level + action
- Consistency: same input → same output
- Calibration: confidence matches accuracy
- Robustness: handles input perturbation
- Efficiency: tokens/second, latency

Supports evaluation at each stage:
1. Baseline (rule-based)
2. Raw model (before fine-tuning)
3. Fine-tuned model
4. Quantized model (Q8, Q5, Q4)
5. Edge deployment (RPi 5)
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import click
from rich.console import Console
from rich.table import Table

from data.schema import Action, Decision, Scenario, ThreatLevel

console = Console()


class ModelInterface(Protocol):
    """Protocol for model inference."""
    def predict(self, scenario: Scenario) -> Decision: ...
    def name(self) -> str: ...


@dataclass
class EvalResult:
    """Results for a single scenario."""
    scenario_id: str
    ground_truth: Decision
    prediction: Decision
    latency_ms: float
    
    @property
    def threat_correct(self) -> bool:
        return self.prediction.threat_level == self.ground_truth.threat_level
    
    @property
    def action_correct(self) -> bool:
        return self.prediction.action == self.ground_truth.action
    
    @property
    def fully_correct(self) -> bool:
        return self.threat_correct and self.action_correct


@dataclass
class EvalMetrics:
    """Aggregated evaluation metrics."""
    model_name: str
    n_samples: int
    
    # Accuracy
    threat_accuracy: float = 0.0
    action_accuracy: float = 0.0
    full_accuracy: float = 0.0
    
    # Per-class accuracy
    threat_by_class: dict = field(default_factory=dict)
    action_by_class: dict = field(default_factory=dict)
    
    # Calibration
    expected_calibration_error: float = 0.0
    
    # Consistency (requires multiple runs)
    consistency_score: float = 0.0
    
    # Efficiency
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    tokens_per_second: float = 0.0
    
    # Failure analysis
    failure_modes: dict = field(default_factory=dict)


def compute_metrics(results: list[EvalResult], model_name: str) -> EvalMetrics:
    """Compute evaluation metrics from results."""
    n = len(results)
    
    metrics = EvalMetrics(model_name=model_name, n_samples=n)
    
    # Accuracy
    metrics.threat_accuracy = sum(r.threat_correct for r in results) / n
    metrics.action_accuracy = sum(r.action_correct for r in results) / n
    metrics.full_accuracy = sum(r.fully_correct for r in results) / n
    
    # Per-class accuracy
    for threat in ThreatLevel:
        class_results = [r for r in results if r.ground_truth.threat_level == threat]
        if class_results:
            metrics.threat_by_class[threat.value] = sum(r.threat_correct for r in class_results) / len(class_results)
    
    for action in Action:
        class_results = [r for r in results if r.ground_truth.action == action]
        if class_results:
            metrics.action_by_class[action.value] = sum(r.action_correct for r in class_results) / len(class_results)
    
    # Calibration (Expected Calibration Error)
    # Bin predictions by confidence and compare to actual accuracy
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ece = 0.0
    for i in range(len(bins) - 1):
        bin_results = [r for r in results if bins[i] <= r.prediction.confidence < bins[i+1]]
        if bin_results:
            bin_accuracy = sum(r.fully_correct for r in bin_results) / len(bin_results)
            bin_confidence = sum(r.prediction.confidence for r in bin_results) / len(bin_results)
            ece += len(bin_results) / n * abs(bin_accuracy - bin_confidence)
    metrics.expected_calibration_error = ece
    
    # Latency
    latencies = sorted([r.latency_ms for r in results])
    metrics.mean_latency_ms = sum(latencies) / n
    metrics.p95_latency_ms = latencies[int(0.95 * n)] if n > 0 else 0
    
    # Failure analysis
    failures = [r for r in results if not r.fully_correct]
    if failures:
        # Group by ground truth threat level
        for threat in ThreatLevel:
            threat_failures = [r for r in failures if r.ground_truth.threat_level == threat]
            if threat_failures:
                metrics.failure_modes[f"gt_{threat.value}"] = len(threat_failures)
    
    return metrics


def print_metrics(metrics: EvalMetrics):
    """Pretty print evaluation metrics."""
    console.print(f"\n[bold]Evaluation Results: {metrics.model_name}[/bold]")
    console.print(f"Samples: {metrics.n_samples}\n")
    
    # Accuracy table
    table = Table(title="Accuracy")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Threat Level", f"{metrics.threat_accuracy:.1%}")
    table.add_row("Action", f"{metrics.action_accuracy:.1%}")
    table.add_row("Full (both correct)", f"{metrics.full_accuracy:.1%}")
    console.print(table)
    
    # Per-class accuracy
    if metrics.threat_by_class:
        table = Table(title="Threat Level Accuracy by Class")
        table.add_column("Class", style="cyan")
        table.add_column("Accuracy", style="green")
        for cls, acc in sorted(metrics.threat_by_class.items()):
            table.add_row(cls, f"{acc:.1%}")
        console.print(table)
    
    if metrics.action_by_class:
        table = Table(title="Action Accuracy by Class")
        table.add_column("Class", style="cyan")
        table.add_column("Accuracy", style="green")
        for cls, acc in sorted(metrics.action_by_class.items()):
            table.add_row(cls, f"{acc:.1%}")
        console.print(table)
    
    # Calibration & Efficiency
    table = Table(title="Calibration & Efficiency")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Expected Calibration Error", f"{metrics.expected_calibration_error:.3f}")
    table.add_row("Mean Latency", f"{metrics.mean_latency_ms:.1f}ms")
    table.add_row("P95 Latency", f"{metrics.p95_latency_ms:.1f}ms")
    console.print(table)
    
    # Failure modes
    if metrics.failure_modes:
        table = Table(title="Failure Analysis (by ground truth)")
        table.add_column("Category", style="cyan")
        table.add_column("Count", style="red")
        for mode, count in sorted(metrics.failure_modes.items(), key=lambda x: -x[1]):
            table.add_row(mode, str(count))
        console.print(table)


def evaluate_model(
    model: ModelInterface,
    scenarios: list[tuple[Scenario, Decision]],
    n_runs: int = 1
) -> EvalMetrics:
    """Run evaluation on a model."""
    results = []
    
    for scenario, ground_truth in scenarios:
        start = time.perf_counter()
        prediction = model.predict(scenario)
        latency = (time.perf_counter() - start) * 1000
        
        results.append(EvalResult(
            scenario_id=scenario.id,
            ground_truth=ground_truth,
            prediction=prediction,
            latency_ms=latency
        ))
    
    return compute_metrics(results, model.name())


class RuleBasedModel:
    """Rule-based baseline model."""
    
    def predict(self, scenario: Scenario) -> Decision:
        from data.rule_label import label_scenario
        return label_scenario(scenario)
    
    def name(self) -> str:
        return "rule-based-baseline"


@click.group()
def cli():
    """HELM-inspired evaluation framework."""
    pass


@cli.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--model", type=click.Choice(["baseline", "raw", "finetuned", "quantized"]), default="baseline")
@click.option("--limit", type=int, default=None, help="Limit number of scenarios")
def run(data_path: str, model: str, limit: int | None):
    """Run evaluation on a model."""
    
    # Load data
    scenarios = []
    with open(data_path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            data = json.loads(line)
            scenario = Scenario(**data["scenario"])
            decision = Decision(**data["decision"])
            scenarios.append((scenario, decision))
    
    console.print(f"Loaded {len(scenarios)} scenarios")
    
    # Select model
    if model == "baseline":
        eval_model = RuleBasedModel()
    else:
        console.print(f"[red]Model '{model}' not yet implemented[/red]")
        return
    
    # Run evaluation
    metrics = evaluate_model(eval_model, scenarios)
    print_metrics(metrics)


@cli.command()
@click.argument("data_path", type=click.Path(exists=True))
def compare(data_path: str):
    """Compare all available models."""
    console.print("[yellow]Model comparison not yet implemented[/yellow]")
    console.print("Will compare: baseline, raw, finetuned, Q8, Q5, Q4")


if __name__ == "__main__":
    cli()
