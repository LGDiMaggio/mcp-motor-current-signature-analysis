"""Tests for synthetic test signal generation."""

import numpy as np
import pytest

from mcp_server_mcsa.analysis.test_signal import (
    generate_healthy_signal,
    generate_test_signal,
    inject_bearing_fault,
    inject_brb_fault,
    inject_eccentricity_fault,
)


class TestGenerateHealthySignal:
    def test_basic_generation(self):
        t, x = generate_healthy_signal(1.0, 5000.0, 50.0)
        assert len(t) == 5000
        assert len(x) == 5000
        assert t[0] == 0.0

    def test_amplitude(self):
        t, x = generate_healthy_signal(1.0, 5000.0, 50.0, amplitude=2.0, noise_std=0)
        assert np.max(np.abs(x)) == pytest.approx(2.0, abs=0.1)

    def test_with_harmonics(self):
        t, x = generate_healthy_signal(
            1.0, 5000.0, 50.0,
            harmonics=[(3, 0.1), (5, 0.05)],
            noise_std=0,
        )
        # Should have content at 150 and 250 Hz
        freqs = np.fft.rfftfreq(len(x), 1.0 / 5000.0)
        amps = np.abs(np.fft.rfft(x))
        idx_150 = np.argmin(np.abs(freqs - 150))
        assert amps[idx_150] > 100  # relative to FFT scale


class TestInjectFaults:
    @pytest.fixture
    def base_signal(self):
        t, x = generate_healthy_signal(10.0, 5000.0, 50.0, noise_std=0.001)
        return t, x

    def test_brb_injection(self, base_signal):
        t, x = base_signal
        y = inject_brb_fault(t, x, 50.0, 0.02, 0.05)
        assert len(y) == len(x)
        # Signal should have changed
        assert not np.allclose(x, y)

    def test_eccentricity_injection(self, base_signal):
        t, x = base_signal
        y = inject_eccentricity_fault(t, x, 50.0, 24.5, 0.03)
        assert not np.allclose(x, y)

    def test_bearing_injection(self, base_signal):
        t, x = base_signal
        y = inject_bearing_fault(t, x, 50.0, 85.0, 0.02)
        assert not np.allclose(x, y)


class TestGenerateTestSignal:
    def test_healthy(self):
        result = generate_test_signal(duration_s=1.0, fs_sample=2000.0)
        assert result["n_samples"] == 2000
        assert result["faults_injected"] == []
        assert len(result["signal"]) == 2000
        assert len(result["time_s"]) == 2000

    def test_with_brb_fault(self):
        result = generate_test_signal(
            duration_s=1.0, fs_sample=2000.0,
            faults=["brb"], fault_severity=0.05,
        )
        assert "broken_rotor_bars" in result["faults_injected"]

    def test_with_multiple_faults(self):
        result = generate_test_signal(
            duration_s=1.0, fs_sample=2000.0,
            faults=["brb", "eccentricity", "bearing"],
            fault_severity=0.03,
        )
        assert len(result["faults_injected"]) == 3

    def test_motor_params_included(self):
        result = generate_test_signal(
            supply_freq_hz=60.0, poles=2, rotor_speed_rpm=3540.0,
        )
        assert result["motor_params"]["supply_freq_hz"] == 60.0
        assert result["motor_params"]["poles"] == 2
