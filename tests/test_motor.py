"""Tests for motor parameter calculations."""

import pytest

from mcp_server_mcsa.analysis.motor import (
    calculate_fault_frequencies,
    calculate_motor_parameters,
)


class TestCalculateMotorParameters:
    def test_basic_4pole_50hz(self):
        params = calculate_motor_parameters(50.0, 4, 1470.0)
        assert params.sync_speed_rpm == 1500.0
        assert params.slip == pytest.approx(0.02, abs=1e-6)
        assert params.rotor_freq_hz == pytest.approx(24.5, abs=1e-6)
        assert params.slip_freq_hz == pytest.approx(1.0, abs=1e-6)

    def test_basic_2pole_60hz(self):
        params = calculate_motor_parameters(60.0, 2, 3540.0)
        assert params.sync_speed_rpm == 3600.0
        assert params.slip == pytest.approx(1 / 60, abs=1e-6)

    def test_zero_speed(self):
        params = calculate_motor_parameters(50.0, 4, 0.0)
        assert params.slip == 1.0
        assert params.rotor_freq_hz == 0.0

    def test_invalid_poles_odd(self):
        with pytest.raises(ValueError, match="even"):
            calculate_motor_parameters(50.0, 3, 1470.0)

    def test_invalid_poles_zero(self):
        with pytest.raises(ValueError, match="even"):
            calculate_motor_parameters(50.0, 0, 1470.0)

    def test_invalid_supply_freq(self):
        with pytest.raises(ValueError, match="Supply frequency"):
            calculate_motor_parameters(-10.0, 4, 1470.0)

    def test_overspeed_raises(self):
        with pytest.raises(ValueError, match="exceeds"):
            calculate_motor_parameters(50.0, 4, 1600.0)

    def test_to_dict(self):
        params = calculate_motor_parameters(50.0, 4, 1470.0)
        d = params.to_dict()
        assert d["supply_freq_hz"] == 50.0
        assert d["poles"] == 4
        assert "slip" in d


class TestCalculateFaultFrequencies:
    def test_brb_frequencies(self, motor_params_50hz_4p):
        result = calculate_fault_frequencies(motor_params_50hz_4p, harmonics=2)
        brb = result["broken_rotor_bars"]
        # (1 - 2*0.02)*50 = 48 Hz for k=1
        assert brb["lower_sidebands_hz"][0] == pytest.approx(48.0, abs=0.1)
        # (1 + 2*0.02)*50 = 52 Hz for k=1
        assert brb["upper_sidebands_hz"][0] == pytest.approx(52.0, abs=0.1)

    def test_eccentricity_frequencies(self, motor_params_50hz_4p):
        result = calculate_fault_frequencies(motor_params_50hz_4p, harmonics=1)
        ecc = result["eccentricity"]
        # fs ± fr = 50 ± 24.5
        assert ecc["sidebands"][0]["lower"] == pytest.approx(25.5, abs=0.1)
        assert ecc["sidebands"][0]["upper"] == pytest.approx(74.5, abs=0.1)

    def test_stator_frequencies(self, motor_params_50hz_4p):
        result = calculate_fault_frequencies(motor_params_50hz_4p, harmonics=1)
        st = result["stator_faults"]
        # fs ± 2*fr = 50 ± 49
        assert st["sidebands"][0]["lower"] == pytest.approx(1.0, abs=0.1)
        assert st["sidebands"][0]["upper"] == pytest.approx(99.0, abs=0.1)

    def test_harmonics_count(self, motor_params_50hz_4p):
        result = calculate_fault_frequencies(motor_params_50hz_4p, harmonics=5)
        assert len(result["broken_rotor_bars"]["lower_sidebands_hz"]) == 5
