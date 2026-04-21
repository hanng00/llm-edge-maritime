from pydantic import BaseModel
from enum import Enum


class VesselType(str, Enum):
    CARGO = "cargo"
    TANKER = "tanker"
    FISHING = "fishing"
    MILITARY = "military"
    RECREATIONAL = "recreational"
    UNKNOWN = "unknown"


class ThreatLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Action(str, Enum):
    CONTINUE = "continue"
    MONITOR = "monitor"
    EVADE = "evade"
    ALERT = "alert"
    ABORT = "abort"


class Vessel(BaseModel):
    bearing: float  # degrees, 0-360
    distance: float  # nautical miles
    speed: float  # knots
    heading: float  # degrees, 0-360
    vessel_type: VesselType
    ais_active: bool
    visual_description: str | None = None


class Scenario(BaseModel):
    id: str
    own_position: tuple[float, float]  # lat, lon
    own_heading: float
    own_speed: float
    mission_type: str
    vessels: list[Vessel]
    weather: str
    visibility: str  # good, moderate, poor
    time_of_day: str  # day, night, dawn, dusk
    comms_status: str  # full, degraded, denied


class Decision(BaseModel):
    threat_level: ThreatLevel
    action: Action
    reasoning: str
    confidence: float  # 0-1
