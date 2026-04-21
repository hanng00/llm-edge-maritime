"""Batch label scenarios using Claude API."""

import json
from pathlib import Path

import click
from anthropic import Anthropic
from rich.console import Console
from rich.progress import Progress

from .schema import Action, Decision, Scenario, ThreatLevel

console = Console()
client = Anthropic()

LABEL_PROMPT = """You are an expert in maritime tactical decision-making for autonomous naval drones.

Given a scenario, provide a tactical assessment. Consider:
- Unknown vessels without AIS are higher risk
- High speed + direct approach = potential threat  
- Comms denied means autonomous decision required
- Close range (<2nm) requires immediate action
- Multiple vessels without AIS suggests coordinated activity
- Mission preservation vs threat avoidance tradeoff

Respond with ONLY valid JSON (no markdown, no explanation):
{
  "threat_level": "none|low|medium|high|critical",
  "action": "continue|monitor|evade|alert|abort",
  "reasoning": "1-2 sentences",
  "confidence": 0.0-1.0
}"""


def label_scenario(scenario: Scenario) -> Decision:
    """Use Claude to label a scenario."""
    scenario_text = scenario.model_dump_json(indent=2)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": f"{LABEL_PROMPT}\n\nScenario:\n{scenario_text}",
            }
        ],
    )

    content = response.content[0].text.strip()
    # Handle potential markdown wrapping
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    
    data = json.loads(content)

    return Decision(
        threat_level=ThreatLevel(data["threat_level"]),
        action=Action(data["action"]),
        reasoning=data["reasoning"],
        confidence=data["confidence"],
    )


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--start", default=0, help="Start index")
@click.option("--limit", default=None, type=int, help="Max scenarios to label")
def batch_label(input_path: str, output_path: str, start: int, limit: int | None):
    """Label scenarios from INPUT_PATH and write to OUTPUT_PATH."""
    
    # Load scenarios
    scenarios = []
    with open(input_path) as f:
        for line in f:
            data = json.loads(line)
            scenarios.append(Scenario(**data["scenario"]))
    
    # Slice if needed
    end = start + limit if limit else len(scenarios)
    scenarios = scenarios[start:end]
    
    console.print(f"Labeling {len(scenarios)} scenarios...")
    
    # Label and write
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output, "a") as f:
        with Progress() as progress:
            task = progress.add_task("Labeling...", total=len(scenarios))
            
            for scenario in scenarios:
                try:
                    decision = label_scenario(scenario)
                    item = {
                        "scenario": scenario.model_dump(),
                        "decision": decision.model_dump()
                    }
                    f.write(json.dumps(item) + "\n")
                    f.flush()
                    progress.update(task, advance=1)
                except Exception as e:
                    console.print(f"[red]Error on {scenario.id}: {e}[/red]")
                    progress.update(task, advance=1)
    
    console.print(f"[green]Done! Wrote to {output_path}[/green]")


if __name__ == "__main__":
    batch_label()
