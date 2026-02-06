"""Shared test fixtures for MCSA tests."""

from __future__ import annotations

import numpy as np
import pytest

from mcp_server_mcsa.analysis.motor import MotorParameters, calculate_motor_parameters
from mcp_server_mcsa.analysis.test_signal import generate_healthy_signal, inject_brb_fault


@pytest.fixture
def motor_params_50hz_4p() -> MotorParameters:
    """Standard 4-pole 50 Hz motor at 1470 RPM."""
    return calculate_motor_parameters(50.0, 4, 1470.0)


@pytest.fixture
def motor_params_60hz_2p() -> MotorParameters:
    """2-pole 60 Hz motor at 3540 RPM."""
    return calculate_motor_parameters(60.0, 2, 3540.0)


@pytest.fixture
def healthy_signal_50hz():
    """Healthy 50 Hz motor current signal — 10 s, 5 kHz sampling."""
    t, x = generate_healthy_signal(
        duration_s=10.0,
        fs_sample=5000.0,
        supply_freq_hz=50.0,
        amplitude=1.0,
        noise_std=0.005,
        harmonics=[(3, 0.03), (5, 0.015)],
    )
    return {"time": t, "signal": x, "fs": 5000.0, "supply_freq": 50.0}


@pytest.fixture
def brb_signal_50hz(healthy_signal_50hz):
    """50 Hz signal with injected broken rotor bar fault."""
    data = healthy_signal_50hz
    slip = (1500 - 1470) / 1500  # 4-pole, 50 Hz
    x_fault = inject_brb_fault(
        data["time"], data["signal"],
        supply_freq_hz=50.0,
        slip=slip,
        sideband_amplitude=0.05,
    )
    return {**data, "signal": x_fault, "slip": slip}
