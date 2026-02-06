"""Tests for fault detection functions."""

import numpy as np

from mcp_server_mcsa.analysis.fault_detection import (
    band_energy_index,
    bearing_fault_index,
    brb_fault_index,
    eccentricity_fault_index,
    envelope_statistical_indices,
    stator_fault_index,
)
from mcp_server_mcsa.analysis.motor import calculate_motor_parameters
from mcp_server_mcsa.analysis.spectral import compute_fft_spectrum
from mcp_server_mcsa.analysis.test_signal import (
    generate_healthy_signal,
    inject_eccentricity_fault,
)


class TestBRBFaultIndex:
    def test_healthy_signal_below_threshold(self, healthy_signal_50hz):
        data = healthy_signal_50hz
        params = calculate_motor_parameters(50.0, 4, 1470.0)
        freqs, amps = compute_fft_spectrum(data["signal"], data["fs"])
        result = brb_fault_index(freqs, amps, params)

        assert result["severity"] == "healthy"
        assert result["combined_index_db"] < -45

    def test_faulty_signal_detected(self, brb_signal_50hz):
        data = brb_signal_50hz
        params = calculate_motor_parameters(50.0, 4, 1470.0)
        freqs, amps = compute_fft_spectrum(data["signal"], data["fs"])
        result = brb_fault_index(freqs, amps, params)

        # Strong fault injection should be detected
        assert result["severity"] != "healthy"
        assert result["lower_sideband"]["found"]
        assert result["upper_sideband"]["found"]

    def test_fundamental_found(self, healthy_signal_50hz):
        data = healthy_signal_50hz
        params = calculate_motor_parameters(50.0, 4, 1470.0)
        freqs, amps = compute_fft_spectrum(data["signal"], data["fs"])
        result = brb_fault_index(freqs, amps, params)

        assert result["fundamental"]["found"]
        assert result["fundamental"]["amplitude"] > 0.5


class TestEccentricityFaultIndex:
    def test_healthy_signal(self, healthy_signal_50hz):
        data = healthy_signal_50hz
        params = calculate_motor_parameters(50.0, 4, 1470.0)
        freqs, amps = compute_fft_spectrum(data["signal"], data["fs"])
        result = eccentricity_fault_index(freqs, amps, params)

        assert result["severity"] == "healthy"

    def test_eccentricity_fault_detected(self):
        t, x = generate_healthy_signal(10.0, 5000.0, 50.0, noise_std=0.005)
        x_fault = inject_eccentricity_fault(t, x, 50.0, 24.5, 0.05)
        params = calculate_motor_parameters(50.0, 4, 1470.0)
        freqs, amps = compute_fft_spectrum(x_fault, 5000.0)
        result = eccentricity_fault_index(freqs, amps, params)

        assert result["severity"] != "healthy"


class TestStatorFaultIndex:
    def test_structure(self, healthy_signal_50hz):
        data = healthy_signal_50hz
        params = calculate_motor_parameters(50.0, 4, 1470.0)
        freqs, amps = compute_fft_spectrum(data["signal"], data["fs"])
        result = stator_fault_index(freqs, amps, params)

        assert "sidebands" in result
        assert "severity" in result
        assert result["fault_type"] == "stator_inter_turn"


class TestBearingFaultIndex:
    def test_structure(self, healthy_signal_50hz):
        data = healthy_signal_50hz
        freqs, amps = compute_fft_spectrum(data["signal"], data["fs"])
        result = bearing_fault_index(
            freqs, amps,
            supply_freq_hz=50.0,
            bearing_defect_freq_hz=85.0,
            defect_type="bpfo",
        )

        assert "sidebands" in result
        assert result["fault_type"] == "bearing_bpfo"
        assert "note" in result


class TestBandEnergyIndex:
    def test_nonzero_energy(self):
        freqs = np.arange(0, 500, 0.5)
        psd = np.ones_like(freqs) * 0.001
        # Add a peak at 50 Hz
        psd[np.abs(freqs - 50) < 2] = 1.0

        result = band_energy_index(freqs, psd, 50.0, bandwidth_hz=10.0)
        assert result["found"] is True
        assert result["band_energy"] > 0

    def test_empty_band(self):
        freqs = np.arange(0, 100, 1.0)
        psd = np.ones_like(freqs)
        result = band_energy_index(freqs, psd, 500.0, bandwidth_hz=5.0)
        assert result["found"] is False


class TestEnvelopeStatisticalIndices:
    def test_gaussian_kurtosis(self):
        rng = np.random.default_rng(42)
        env = rng.normal(0, 1, 10000)
        stats = envelope_statistical_indices(env)
        # Gaussian kurtosis (Fisher) should be near 0
        assert abs(stats["kurtosis"]) < 0.5
        assert abs(stats["skewness"]) < 0.2

    def test_impulsive_high_kurtosis(self):
        env = np.zeros(10000)
        env[::100] = 10.0  # periodic impulses
        stats = envelope_statistical_indices(env)
        assert stats["kurtosis"] > 5.0
