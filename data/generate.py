"""Generate synthetic maritime scenarios and label with Claude."""

import json
import random
import uuid
from pathlib import Path

import click
from anthropic import Anthropic
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .schema import (
    Action,
    Decision,
    Scenario,
    ThreatLevel,
    Vessel,
    VesselType,
)

console = Console()
client = Anthropic()

LABEL_PROMPT = """You are an expert in maritime tactical decision-making for autonomous naval drones.

Given a scenario, provide a tactical assessment with:
1. threat_level: none, low, medium, high, critical
2. action: continue, monitor, evade, alert, abort
3. reasoning: 1-2 sentences explaining the decision
4. confidence: 0.0-1.0

Consider:
- Unknown vessels without AIS are higher risk
- High speed + direct approach = potential threat
- Comms denied means autonomous decision required
- Mission preservation vs threat avoidance tradeoff

Respond with valid JSON matching this schema:
{
  "threat_level": "low",
  "action": "monitor",
  "reasoning": "...",
  "confidence": 0.85
}"""


def generate_random_scenario() -> Scenario:
    n_vessels = random.choices([1, 2, 3, 4], weights=[0.4, 0.3, 0.2, 0.1])[0]

    vessels = []
    for _ in range(n_vessels):
        vessels.append(
            Vessel(
                bearing=random.uniform(0, 360),
                distance=random.uniform(0.5, 15),
                speed=random.uniform(0, 35),
                heading=random.uniform(0, 360),
                vessel_type=random.choice(list(VesselType)),
                ais_active=random.random() > 0.3,
                visual_description=random.choice(
                    [None, "small craft", "large vessel", "low profile"]
                ),
            )
        )

    return Scenario(
        id=str(uuid.uuid4())[:8],
        own_position=(random.uniform(57.0, 60.0), random.uniform(17.0, 20.0)),
        own_heading=random.uniform(0, 360),
        own_speed=random.uniform(5, 25),
        mission_type=random.choice(["patrol", "reconnaissance", "transit", "surveillance"]),
        vessels=vessels,
        weather=random.choice(["calm", "moderate", "rough"]),
        visibility=random.choice(["good", "moderate", "poor"]),
        time_of_day=random.choice(["day", "night", "dawn", "dusk"]),
        comms_status=random.choice(["full", "degraded", "denied"]),
    )


def label_scenario(scenario: Scenario) -> Decision:
    scenario_text = scenario.model_dump_json(indent=2)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": f"{LABEL_PROMPT}\n\nScenario:\n{scenario_text}",
            }
        ],
    )

    content = response.content[0].text
    data = json.loads(content)

    return Decision(
        threat_level=ThreatLevel(data["threat_level"]),
        action=Action(data["action"]),
        reasoning=data["reasoning"],
        confidence=data["confidence"],
    )


def render_scenario(scenario: Scenario, decision: Decision | None = None):
    table = Table(title=f"Contacts", show_header=True, header_style="bold")
    table.add_column("#")
    table.add_column("Type")
    table.add_column("Bearing")
    table.add_column("Dist (nm)")
    table.add_column("Speed (kn)")
    table.add_column("AIS")

    for i, v in enumerate(scenario.vessels, 1):
        ais = "[green]ON[/]" if v.ais_active else "[red]OFF[/]"
        table.add_row(
            str(i),
            v.vessel_type.value,
            f"{v.bearing:.0f}°",
            f"{v.distance:.1f}",
            f"{v.speed:.0f}",
            ais,
        )

    situation = (
        f"[bold]Mission:[/] {scenario.mission_type}  "
        f"[bold]Hdg:[/] {scenario.own_heading:.0f}°  "
        f"[bold]Spd:[/] {scenario.own_speed:.0f} kn\n"
        f"[bold]Weather:[/] {scenario.weather}  "
        f"[bold]Vis:[/] {scenario.visibility}  "
        f"[bold]Time:[/] {scenario.time_of_day}  "
        f"[bold]Comms:[/] {scenario.comms_status}"
    )

    console.print(Panel(situation, title=f"Scenario {scenario.id}", border_style="blue"))
    console.print(table)

    if decision:
        color = {
            ThreatLevel.NONE: "green",
            ThreatLevel.LOW: "green",
            ThreatLevel.MEDIUM: "yellow",
            ThreatLevel.HIGH: "red",
            ThreatLevel.CRITICAL: "bold red",
        }[decision.threat_level]

        console.print(
            Panel(
                f"[bold]Threat:[/] [{color}]{decision.threat_level.value.upper()}[/]  "
                f"[bold]Action:[/] {decision.action.value.upper()}\n"
                f"[bold]Reasoning:[/] {decision.reasoning}\n"
                f"[bold]Confidence:[/] {decision.confidence:.0%}",
                title="Decision",
                border_style="green" if decision.action == Action.CONTINUE else "yellow",
            )
        )
    console.print()


@click.group()
def cli():
    """Maritime scenario generator for edge LLM training."""
    pass


@cli.command()
@click.option("-n", "--count", default=5, help="Number of scenarios to preview")
def preview(count: int):
    """Generate and display scenarios without labeling."""
    for _ in range(count):
        scenario = generate_random_scenario()
        render_scenario(scenario)


@cli.command()
@click.option("-n", "--count", default=100, help="Number of scenarios to generate")
@click.option("-o", "--output", default="data/scenarios.jsonl", type=click.Path())
def generate(count: int, output: str):
    """Generate scenarios and label with Claude."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labeled = []
    with console.status("[bold green]Generating and labeling scenarios...") as status:
        for i in range(count):
            scenario = generate_random_scenario()
            decision = label_scenario(scenario)
            labeled.append({"scenario": scenario.model_dump(), "decision": decision.model_dump()})
            status.update(f"[bold green]Labeled {i + 1}/{count}: {scenario.id} → {decision.action.value}")

            if (i + 1) % 10 == 0:
                console.print(f"  {i + 1}/{count} complete")

    with open(output_path, "w") as f:
        for item in labeled:
            f.write(json.dumps(item) + "\n")

    console.print(f"\n[bold green]Saved {len(labeled)} scenarios to {output_path}")


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("-n", "--count", default=5, help="Number of scenarios to show")
def show(path: str, count: int):
    """Display labeled scenarios from a JSONL file."""
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= count:
                break
            data = json.loads(line)
            scenario = Scenario(**data["scenario"])
            decision = Decision(**data["decision"])
            render_scenario(scenario, decision)


if __name__ == "__main__":
    cli()
