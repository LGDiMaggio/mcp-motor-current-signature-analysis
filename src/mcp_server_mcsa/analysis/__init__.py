"""MCSA analysis library — core signal processing and fault detection modules."""

from mcp_server_mcsa.analysis.bearing import (
    BearingDefectFrequencies,
    BearingGeometry,
    calculate_bearing_defect_frequencies,
)
from mcp_server_mcsa.analysis.motor import (
    MotorParameters,
    calculate_motor_parameters,
)

__all__ = [
    "MotorParameters",
    "calculate_motor_parameters",
    "BearingGeometry",
    "BearingDefectFrequencies",
    "calculate_bearing_defect_frequencies",
]
