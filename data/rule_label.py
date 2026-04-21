"""Rule-based scenario labeler for bootstrapping training data.

This implements the tactical decision logic explicitly, which:
1. Provides fast labeling without API calls
2. Creates a baseline to compare against the fine-tuned model
3. Makes the decision logic transparent and auditable

The rules encode domain knowledge about maritime threat assessment.
"""

import json
import math
from pathlib import Path

import click
from rich.console import Console

from .schema import Action, Decision, Scenario, ThreatLevel, VesselType

console = Console()


def assess_vessel_threat(vessel, own_heading: float, comms_status: str) -> tuple[float, str]:
    """Assess threat level for a single vessel. Returns (score, reason)."""
    score = 0.0
    reasons = []
    
    # No AIS is a major red flag
    if not vessel.ais_active:
        score += 3.0
        reasons.append("no AIS")
    
    # Unknown vessel type
    if vessel.vessel_type == VesselType.UNKNOWN:
        score += 1.5
        reasons.append("unknown type")
    
    # Military vessel
    if vessel.vessel_type == VesselType.MILITARY:
        if not vessel.ais_active:
            score += 2.0
            reasons.append("military without AIS")
        else:
            score += 0.5  # Could be friendly, but worth noting
            reasons.append("military")
    
    # Close range is dangerous
    if vessel.distance < 1.0:
        score += 4.0
        reasons.append(f"very close ({vessel.distance:.1f}nm)")
    elif vessel.distance < 2.0:
        score += 2.5
        reasons.append(f"close ({vessel.distance:.1f}nm)")
    elif vessel.distance < 5.0:
        score += 1.0
        reasons.append(f"moderate range ({vessel.distance:.1f}nm)")
    
    # High speed is concerning
    if vessel.speed > 30:
        score += 2.0
        reasons.append(f"very fast ({vessel.speed:.0f}kn)")
    elif vessel.speed > 20:
        score += 1.0
        reasons.append(f"fast ({vessel.speed:.0f}kn)")
    
    # Check if on intercept course (simplified)
    # Vessel heading toward us if their heading is roughly opposite to bearing from us
    bearing_to_vessel = vessel.bearing
    vessel_heading = vessel.heading
    # If vessel is heading toward our position (within 45 degrees of reciprocal bearing)
    reciprocal = (bearing_to_vessel + 180) % 360
    heading_diff = abs(vessel_heading - reciprocal)
    if heading_diff > 180:
        heading_diff = 360 - heading_diff
    
    if heading_diff < 30 and vessel.speed > 10:
        score += 2.0
        reasons.append("intercept course")
    elif heading_diff < 60 and vessel.speed > 15:
        score += 1.0
        reasons.append("converging")
    
    # Low profile description is suspicious
    if vessel.visual_description == "low profile":
        score += 0.5
        reasons.append("low profile")
    
    return score, ", ".join(reasons) if reasons else "normal traffic"


def label_scenario(scenario: Scenario) -> Decision:
    """Apply rule-based labeling to a scenario."""
    
    # Assess each vessel
    vessel_assessments = []
    total_threat = 0.0
    max_vessel_threat = 0.0
    threat_reasons = []
    
    for i, vessel in enumerate(scenario.vessels):
        score, reason = assess_vessel_threat(vessel, scenario.own_heading, scenario.comms_status)
        vessel_assessments.append((score, reason))
        total_threat += score
        max_vessel_threat = max(max_vessel_threat, score)
        if score > 1.0:
            threat_reasons.append(f"C{i+1}: {reason}")
    
    # Situational modifiers
    if scenario.comms_status == "denied":
        total_threat *= 1.3
        threat_reasons.append("comms denied")
    elif scenario.comms_status == "degraded":
        total_threat *= 1.1
    
    if scenario.visibility == "poor":
        total_threat *= 1.2
        if max_vessel_threat > 2:
            threat_reasons.append("poor visibility")
    
    if scenario.time_of_day == "night":
        total_threat *= 1.1
    
    # Multiple vessels without AIS suggests coordination
    no_ais_count = sum(1 for v in scenario.vessels if not v.ais_active)
    if no_ais_count >= 2:
        total_threat += 2.0
        threat_reasons.append(f"{no_ais_count} vessels without AIS")
    
    # Determine threat level
    if total_threat < 1.5:
        threat_level = ThreatLevel.NONE
    elif total_threat < 3.5:
        threat_level = ThreatLevel.LOW
    elif total_threat < 6.0:
        threat_level = ThreatLevel.MEDIUM
    elif total_threat < 9.0:
        threat_level = ThreatLevel.HIGH
    else:
        threat_level = ThreatLevel.CRITICAL
    
    # Determine action based on threat level and specifics
    if threat_level == ThreatLevel.NONE:
        action = Action.CONTINUE
        reasoning = "No significant threats detected. Normal maritime traffic."
    elif threat_level == ThreatLevel.LOW:
        action = Action.MONITOR if max_vessel_threat > 1.5 else Action.CONTINUE
        reasoning = f"Low threat. {'; '.join(threat_reasons[:2])}. Monitor situation."
    elif threat_level == ThreatLevel.MEDIUM:
        action = Action.MONITOR
        reasoning = f"Elevated threat. {'; '.join(threat_reasons[:2])}. Close monitoring required."
    elif threat_level == ThreatLevel.HIGH:
        # Check if we should evade or alert
        closest_threat = min((v.distance for v in scenario.vessels if not v.ais_active), default=999)
        if closest_threat < 3.0 or scenario.comms_status == "denied":
            action = Action.EVADE
            reasoning = f"High threat. {'; '.join(threat_reasons[:2])}. Evasive action recommended."
        else:
            action = Action.ALERT
            reasoning = f"High threat. {'; '.join(threat_reasons[:2])}. Alert command and prepare to evade."
    else:  # CRITICAL
        # Check if abort is warranted
        if max_vessel_threat > 6.0 and scenario.comms_status == "denied":
            action = Action.ABORT
            reasoning = f"Critical threat. {'; '.join(threat_reasons[:2])}. Mission abort required."
        else:
            action = Action.EVADE
            reasoning = f"Critical threat. {'; '.join(threat_reasons[:2])}. Immediate evasion required."
    
    # Confidence based on clarity of situation
    if len(scenario.vessels) == 1 and scenario.vessels[0].ais_active:
        confidence = 0.92
    elif no_ais_count == 0:
        confidence = 0.88
    elif len(scenario.vessels) > 3:
        confidence = 0.72
    elif scenario.comms_status == "denied" and no_ais_count > 0:
        confidence = 0.78
    else:
        confidence = 0.82
    
    # Add some variance
    import random
    confidence = min(0.98, max(0.65, confidence + random.uniform(-0.05, 0.05)))
    
    return Decision(
        threat_level=threat_level,
        action=action,
        reasoning=reasoning,
        confidence=round(confidence, 2)
    )


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
def rule_label(input_path: str, output_path: str):
    """Apply rule-based labeling to scenarios."""
    
    scenarios = []
    with open(input_path) as f:
        for line in f:
            data = json.loads(line)
            scenarios.append(Scenario(**data["scenario"]))
    
    console.print(f"Labeling {len(scenarios)} scenarios with rule-based system...")
    
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output, "w") as f:
        for scenario in scenarios:
            decision = label_scenario(scenario)
            item = {
                "scenario": scenario.model_dump(),
                "decision": decision.model_dump()
            }
            f.write(json.dumps(item) + "\n")
    
    console.print(f"[green]Done! Wrote {len(scenarios)} labeled scenarios to {output_path}[/green]")


if __name__ == "__main__":
    rule_label()
