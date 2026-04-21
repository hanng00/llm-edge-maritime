"""Visualize maritime scenarios as tactical plots."""

import json
import math
from pathlib import Path

import click
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle

from data.schema import Scenario, Decision, ThreatLevel, Action, VesselType


VESSEL_COLORS = {
    VesselType.CARGO: "#4A90D9",
    VesselType.TANKER: "#7B68EE",
    VesselType.FISHING: "#3CB371",
    VesselType.MILITARY: "#DC143C",
    VesselType.RECREATIONAL: "#FFD700",
    VesselType.UNKNOWN: "#FF6347",
}

THREAT_COLORS = {
    ThreatLevel.NONE: "#2ECC71",
    ThreatLevel.LOW: "#82E0AA",
    ThreatLevel.MEDIUM: "#F4D03F",
    ThreatLevel.HIGH: "#E67E22",
    ThreatLevel.CRITICAL: "#E74C3C",
}


def bearing_to_xy(bearing: float, distance: float) -> tuple[float, float]:
    """Convert bearing (degrees from north) and distance to x, y coordinates."""
    rad = math.radians(90 - bearing)
    return distance * math.cos(rad), distance * math.sin(rad)


def heading_to_vector(heading: float, speed: float, scale: float = 0.1) -> tuple[float, float]:
    """Convert heading and speed to a velocity vector."""
    rad = math.radians(90 - heading)
    length = speed * scale
    return length * math.cos(rad), length * math.sin(rad)


def plot_scenario(scenario: Scenario, decision: Decision | None = None, ax=None):
    """Plot a single scenario as a tactical display."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    else:
        fig = ax.figure

    ax.set_facecolor("#0f1419")
    ax.set_aspect("equal")

    # Determine plot bounds
    max_dist = max((v.distance for v in scenario.vessels), default=5)
    padding = max(2, max_dist * 0.3)
    plot_range = max_dist + padding
    ax.set_xlim(-plot_range, plot_range)
    ax.set_ylim(-plot_range, plot_range)

    # Draw range rings
    ring_interval = 1 if max_dist < 5 else 2 if max_dist < 10 else 5
    for r in range(ring_interval, int(plot_range) + 1, ring_interval):
        circle = Circle((0, 0), r, fill=False, color="#2a3a4a", linestyle="-", linewidth=0.8)
        ax.add_patch(circle)
        ax.text(0.15, r + 0.1, f"{r} nm", fontsize=9, color="#4a6a8a", fontweight="bold")

    # Draw bearing lines
    for bearing in range(0, 360, 30):
        x, y = bearing_to_xy(bearing, plot_range)
        ax.plot([0, x], [0, y], color="#1a2a3a", linewidth=0.5)

    # Cardinal labels
    for bearing, label in [(0, "N"), (90, "E"), (180, "S"), (270, "W")]:
        x, y = bearing_to_xy(bearing, plot_range * 0.95)
        ax.text(x, y, label, fontsize=14, color="#5a7a9a", ha="center", va="center", fontweight="bold")

    # === Own ship ===
    own_vx, own_vy = heading_to_vector(scenario.own_heading, scenario.own_speed)

    # Small circle for own ship
    own_ship = Circle((0, 0), 0.15, facecolor="#00FF88", edgecolor="white", linewidth=2, zorder=10)
    ax.add_patch(own_ship)

    # Velocity vector
    if scenario.own_speed > 0:
        ax.annotate("", xy=(own_vx, own_vy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#00FF88", lw=2.5),
                    zorder=9)

    # Own ship label
    ax.text(0, -0.5, f"OWN\n{scenario.own_heading:.0f}°/{scenario.own_speed:.0f}kn",
            fontsize=9, color="#00FF88", ha="center", va="top", fontweight="bold")

    # === Contacts ===
    for i, vessel in enumerate(scenario.vessels):
        x, y = bearing_to_xy(vessel.bearing, vessel.distance)
        color = VESSEL_COLORS.get(vessel.vessel_type, "#FF6347")

        # Vessel marker
        marker_size = 0.2
        if vessel.ais_active:
            # Filled circle for AIS
            marker = Circle((x, y), marker_size, facecolor=color, edgecolor="white", linewidth=2, zorder=8)
        else:
            # Hollow circle with X for no AIS
            marker = Circle((x, y), marker_size, facecolor="none", edgecolor=color, linewidth=3, zorder=8)
            ax.plot([x - marker_size*0.7, x + marker_size*0.7], [y - marker_size*0.7, y + marker_size*0.7],
                    color=color, linewidth=2, zorder=8)
            ax.plot([x - marker_size*0.7, x + marker_size*0.7], [y + marker_size*0.7, y - marker_size*0.7],
                    color=color, linewidth=2, zorder=8)
        ax.add_patch(marker)

        # Velocity vector
        if vessel.speed > 0:
            vx, vy = heading_to_vector(vessel.heading, vessel.speed)
            ax.annotate("", xy=(x + vx, y + vy), xytext=(x, y),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=2),
                        zorder=7)

        # Future position (dashed line showing 5-min projection)
        if vessel.speed > 0:
            future_dist = vessel.speed * (5/60)  # 5 minutes in hours * knots = nm
            fx, fy = heading_to_vector(vessel.heading, future_dist, scale=1)
            ax.plot([x, x + fx], [y, y + fy], color=color, linestyle=":", linewidth=1.5, alpha=0.5)
            ax.scatter([x + fx], [y + fy], color=color, s=20, alpha=0.5, marker="o")

        # Label - positioned away from center
        label_offset_x = 0.4 if x >= 0 else -0.4
        label_offset_y = 0.4 if y >= 0 else -0.4
        ha = "left" if x >= 0 else "right"

        ais_str = "AIS" if vessel.ais_active else "NO AIS"
        label_text = f"C{i+1} {vessel.vessel_type.value.upper()}\n{vessel.bearing:.0f}°/{vessel.distance:.1f}nm\n{vessel.speed:.0f}kn | {ais_str}"

        ax.text(x + label_offset_x, y + label_offset_y, label_text,
                fontsize=9, color="white", ha=ha, va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.85, edgecolor="white"))

    # === Title ===
    title_lines = [
        f"Scenario {scenario.id}  |  {scenario.mission_type.upper()}  |  {scenario.time_of_day.upper()}",
        f"Weather: {scenario.weather}  |  Visibility: {scenario.visibility}  |  Comms: {scenario.comms_status.upper()}"
    ]

    if decision:
        threat_color = THREAT_COLORS.get(decision.threat_level, "white")
        title_lines.append(f"ASSESSMENT: [{decision.threat_level.value.upper()}] → {decision.action.value.upper()}")
        title_bg = threat_color
    else:
        title_bg = "#2a3a4a"

    ax.set_title("\n".join(title_lines), fontsize=12, color="white", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor=title_bg, alpha=0.9), pad=15)

    # === Legend ===
    legend_elements = [
        mpatches.Patch(facecolor="#00FF88", edgecolor="white", label="Own Ship"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#4A90D9",
                   markersize=10, label="AIS Active", linestyle="None"),
        plt.Line2D([0], [0], marker="o", color="#FF6347", markerfacecolor="none",
                   markersize=10, markeredgewidth=2, label="No AIS", linestyle="None"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9,
              facecolor="#1a2a3a", edgecolor="#3a5a7a", labelcolor="white")

    # === Reasoning box ===
    if decision:
        reasoning_text = f"Reasoning: {decision.reasoning}\nConfidence: {decision.confidence:.0%}"
        ax.text(0.02, 0.02, reasoning_text, transform=ax.transAxes, fontsize=10,
                color="white", verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a2a3a", alpha=0.95, edgecolor="#3a5a7a"),
                wrap=True)

    ax.set_xlabel("East/West (nm)", color="#5a7a9a", fontsize=10)
    ax.set_ylabel("North/South (nm)", color="#5a7a9a", fontsize=10)
    ax.tick_params(colors="#5a7a9a", labelsize=9)

    for spine in ax.spines.values():
        spine.set_color("#2a3a4a")

    return fig, ax


def plot_scenarios_grid(scenarios: list[tuple[Scenario, Decision | None]], output: Path):
    """Plot multiple scenarios in a grid."""
    n = len(scenarios)
    cols = min(2, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12 * cols, 10 * rows))
    fig.patch.set_facecolor("#0a0e12")

    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, (scenario, decision) in enumerate(scenarios):
        plot_scenario(scenario, decision, ax=axes[i])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(pad=2)
    plt.savefig(output, dpi=150, facecolor="#0a0e12", edgecolor="none")
    plt.close()


@click.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("-n", "--count", default=6, help="Number of scenarios to visualize")
@click.option("-o", "--output", default="viz/plots/scenarios.png", help="Output image path")
@click.option("--single", is_flag=True, help="Output individual images instead of grid")
def visualize(path: str, count: int, output: str, single: bool):
    """Visualize scenarios from a JSONL file."""
    scenarios = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= count:
                break
            data = json.loads(line)
            scenario = Scenario(**data["scenario"])
            decision = Decision(**data["decision"]) if "decision" in data else None
            scenarios.append((scenario, decision))

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if single:
        for scenario, decision in scenarios:
            fig, ax = plot_scenario(scenario, decision)
            single_output = output_path.parent / f"scenario_{scenario.id}.png"
            plt.savefig(single_output, dpi=150, facecolor="#0a0e12", edgecolor="none",
                        bbox_inches="tight", pad_inches=0.3)
            plt.close()
            print(f"Saved {single_output}")
    else:
        plot_scenarios_grid(scenarios, output_path)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    visualize()
