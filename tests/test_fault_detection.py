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


class TestBearingFaultIndexWithAdaptiveFFT:
    """End-to-end demonstration that ``compute_fft_spectrum(
    min_resolution_hz=...)`` lets ``bearing_fault_index`` recover
    sidebands that were below the resolution floor at the native bin
    width (issue #1 + the bench's plan-gate finding that motivated it).
    """

    def test_short_segment_with_min_resolution_recovers_bpfi_sideband(self):
        """A 0.2 s × 20 kHz current signal with a strong BPFI-style
        sideband at supply ± 132 Hz: at the native 5 Hz bin width the
        ``bearing_fault_index`` returns ``-inf`` because the +183 / -83
        sideband falls between bins. With ``min_resolution_hz=0.5`` the
        same signal yields a finite ``worst_sideband_db``."""
        fs = 20000.0
        duration = 0.2
        n = int(fs * duration)
        t = np.arange(n) / fs
        supply = 50.0
        bpfi_freq = 132.4  # ≈ 5.428 × 24.5 Hz (rotor freq at 100%Load)
        side_amp = 0.3
        x = (
            np.sin(2 * np.pi * supply * t)
            + side_amp * np.sin(2 * np.pi * (supply + bpfi_freq) * t)
            + side_amp * np.sin(2 * np.pi * (supply - bpfi_freq) * t)
        )

        # Scope to harmonics=1 to isolate the resolution effect cleanly:
        # the order-1 sideband at +182.4 Hz falls between 5 Hz bins at
        # both native and supply-folded locations; the order-2 sideband at
        # +314.8 Hz happens to land within 0.2 Hz of the 315 Hz native
        # bin and would pollute the test by carrying leakage from
        # order-1 even at native resolution.
        freqs_native, amps_native = compute_fft_spectrum(x, fs)
        res_native = bearing_fault_index(
            freqs_native,
            amps_native,
            supply_freq_hz=supply,
            bearing_defect_freq_hz=bpfi_freq,
            defect_type="bpfi",
            harmonics=1,
        )

        freqs_hi, amps_hi = compute_fft_spectrum(
            x, fs, min_resolution_hz=0.5
        )
        res_hi = bearing_fault_index(
            freqs_hi,
            amps_hi,
            supply_freq_hz=supply,
            bearing_defect_freq_hz=bpfi_freq,
            defect_type="bpfi",
            harmonics=1,
        )

        # Native bin width 5 Hz — order-1 sideband at +182.4 / -82.4 Hz
        # falls between bins (180/185 and 80/85); tolerance_hz=0.5
        # default cannot bridge → worst_sideband_db = -inf.
        assert res_native["worst_sideband_db"] == float("-inf")
        # Hi-res bin width ≤ 0.125 Hz — sideband recovered as a finite
        # value well above the noise floor.
        assert res_hi["worst_sideband_db"] > -60.0


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
