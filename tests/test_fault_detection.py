"""Tests for fault detection functions."""

import numpy as np
import pytest

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


class TestBearingDetectionStatus:
    """Structured ``detection_status`` block in ``bearing_fault_index``
    (issue #2). Reason priority: detected → frequency_out_of_range →
    frequency_resolution_insufficient → no_sideband_present."""

    def test_detected_when_strong_sideband_and_adequate_resolution(self):
        fs = 5000.0
        t = np.arange(0, 5.0, 1.0 / fs)  # 5 s → 0.2 Hz native bin width
        supply, bpfi = 50.0, 132.0
        x = (
            np.sin(2 * np.pi * supply * t)
            + 0.3 * np.sin(2 * np.pi * (supply + bpfi) * t)
        )
        freqs, amps = compute_fft_spectrum(x, fs)
        res = bearing_fault_index(
            freqs, amps,
            supply_freq_hz=supply,
            bearing_defect_freq_hz=bpfi,
            defect_type="bpfi",
            harmonics=1,
        )
        ds = res["detection_status"]
        assert ds["detected"] is True
        assert ds["reason"] == "detected"
        assert ds["fft_bin_width_hz"] <= 0.25
        assert ds["tolerance_hz"] == 0.5
        assert ds["min_bin_width_for_tolerance_hz"] == pytest.approx(0.125)
        # Legacy field preserved unchanged
        assert np.isfinite(res["worst_sideband_db"])

    def test_reason_frequency_resolution_insufficient_on_short_segment(self):
        fs = 20000.0
        n = int(fs * 0.2)
        t = np.arange(n) / fs
        x = np.sin(2 * np.pi * 50 * t)  # clean supply, no sidebands
        freqs, amps = compute_fft_spectrum(x, fs)
        res = bearing_fault_index(
            freqs, amps,
            supply_freq_hz=50.0,
            bearing_defect_freq_hz=132.4,
            defect_type="bpfi",
            harmonics=1,
        )
        ds = res["detection_status"]
        assert ds["detected"] is False
        assert ds["reason"] == "frequency_resolution_insufficient"
        assert ds["fft_bin_width_hz"] == pytest.approx(5.0)
        assert ds["min_bin_width_for_tolerance_hz"] == pytest.approx(0.125)
        # Legacy field preserved (no sideband detected → -inf)
        assert res["worst_sideband_db"] == float("-inf")

    def test_reason_no_sideband_present_on_clean_signal_adequate_resolution(self):
        """20 s × 5 kHz signal → 0.05 Hz native bin width (well under
        the 0.125 Hz safety threshold for tolerance 0.5). No sideband
        injected → reason is the well-formed ``no_sideband_present``,
        not ``frequency_resolution_insufficient``."""
        fs = 5000.0
        t = np.arange(0, 20.0, 1.0 / fs)
        x = np.sin(2 * np.pi * 50 * t)  # clean supply only
        freqs, amps = compute_fft_spectrum(x, fs)
        res = bearing_fault_index(
            freqs, amps,
            supply_freq_hz=50.0,
            bearing_defect_freq_hz=132.0,
            defect_type="bpfi",
            harmonics=1,
        )
        ds = res["detection_status"]
        assert ds["fft_bin_width_hz"] <= 0.125 + 1e-9
        assert ds["detected"] is False
        assert ds["reason"] == "no_sideband_present"

    def test_reason_frequency_out_of_range_when_all_targets_outside_spectrum(self):
        fs = 1000.0  # Nyquist = 500
        t = np.arange(0, 5.0, 1.0 / fs)
        x = np.sin(2 * np.pi * 50 * t)
        freqs, amps = compute_fft_spectrum(x, fs)
        # bpfi=2000: f_lo = 50-2000 = -1950, f_hi = 50+2000 = 2050 → both
        # outside [0, 500]. Resolution is adequate (0.2 Hz bin); the test
        # is conceptually impossible at this sampling rate.
        res = bearing_fault_index(
            freqs, amps,
            supply_freq_hz=50.0,
            bearing_defect_freq_hz=2000.0,
            defect_type="bpfi",
            harmonics=1,
        )
        ds = res["detection_status"]
        assert ds["detected"] is False
        assert ds["reason"] == "frequency_out_of_range"


class TestBRBDetectionStatus:
    """``brb_fault_index`` adds the BRB-specific
    ``sideband_inside_supply_main_lobe`` reason (issue #2)."""

    def test_reason_sideband_inside_supply_main_lobe_on_short_segment(self):
        """0.2 s × 20 kHz: bin width 5 Hz → main-lobe half-width estimate
        2*bw = 10 Hz. BRB slip-sideband distance = 2*0.02*50 = 2 Hz
        (inside main lobe). Reason takes priority over
        ``frequency_resolution_insufficient`` because no amount of
        zero-padding can resolve a sideband that's inside the
        time-domain window's own main lobe."""
        fs = 20000.0
        n = int(fs * 0.2)
        t = np.arange(n) / fs
        x = np.sin(2 * np.pi * 50 * t)  # clean supply
        freqs, amps = compute_fft_spectrum(x, fs)
        params = calculate_motor_parameters(50.0, 4, 1470.0)
        res = brb_fault_index(freqs, amps, params)
        ds = res["detection_status"]
        assert ds["detected"] is False
        assert ds["reason"] == "sideband_inside_supply_main_lobe"
        # Legacy field preserved
        assert res["combined_index_db"] == float("-inf")

    def test_signal_duration_s_prevents_p0_false_positive_on_zero_padded_spectrum(self):
        """**P0 regression guard** (code review 2026-05-28). Without
        ``signal_duration_s``, calling brb_fault_index on a 0.2 s clean
        signal with ``compute_fft_spectrum(min_resolution_hz=0.5)``
        falsely reports detected=True / severity='severe' because the
        bin-width-based main-lobe estimate shrinks to ~0.15 Hz (post-
        padding) while the physical main lobe stays at ~10 Hz.

        Passing ``signal_duration_s=0.2`` lets _build_detection_status
        compute the correct main-lobe half-width (2/0.2 = 10 Hz),
        recognise the BRB slip-sideband (2 Hz) is inside it, and
        OVERRIDE detected to False with reason
        ``sideband_inside_supply_main_lobe``."""
        fs = 20000.0
        signal_duration_s = 0.2
        n = int(fs * signal_duration_s)
        t = np.arange(n) / fs
        x = np.sin(2 * np.pi * 50 * t)  # clean supply, NO BRB fault
        freqs, amps = compute_fft_spectrum(x, fs, min_resolution_hz=0.5)
        params = calculate_motor_parameters(50.0, 4, 1470.0)

        # Without signal_duration_s: legacy behavior, false positive.
        res_legacy = brb_fault_index(freqs, amps, params)
        ds_legacy = res_legacy["detection_status"]
        # The legacy fallback is incorrect on zero-padded spectra (it
        # estimates main_lobe from bin_width which is now tiny). This
        # path reports detected=True on a HEALTHY signal — that's the
        # P0 bug. We pin its behavior here so the user is forced to
        # opt in to the fix by passing signal_duration_s.
        assert ds_legacy["detected"] is True  # WRONG, but legacy contract

        # With signal_duration_s: correct main-lobe check, no false
        # positive.
        res_fixed = brb_fault_index(
            freqs, amps, params, signal_duration_s=signal_duration_s
        )
        ds_fixed = res_fixed["detection_status"]
        assert ds_fixed["detected"] is False
        assert ds_fixed["reason"] == "sideband_inside_supply_main_lobe"

    def test_signal_duration_s_does_not_block_real_detection(self):
        """Sanity check: passing ``signal_duration_s`` must not break
        the detection of a real BRB sideband that IS resolvable. Use a
        long signal (5 s × 5 kHz) with an injected slip-sideband well
        outside the main lobe."""
        fs = 5000.0
        signal_duration_s = 5.0
        n = int(fs * signal_duration_s)
        t = np.arange(n) / fs
        # slip = 0.02 → sideband at 50 ± 2 Hz; main lobe half = 2/5 = 0.4 Hz
        # so 2 Hz is OUTSIDE the main lobe → should detect normally
        x = (
            np.sin(2 * np.pi * 50 * t)
            + 0.1 * np.sin(2 * np.pi * 48 * t)
            + 0.1 * np.sin(2 * np.pi * 52 * t)
        )
        freqs, amps = compute_fft_spectrum(x, fs)
        params = calculate_motor_parameters(50.0, 4, 1470.0)
        res = brb_fault_index(
            freqs, amps, params, signal_duration_s=signal_duration_s
        )
        ds = res["detection_status"]
        assert ds["detected"] is True
        assert ds["reason"] == "detected"


class TestEccentricityDetectionStatus:
    def test_detection_status_block_present_with_required_fields(self):
        fs = 5000.0
        t = np.arange(0, 5.0, 1.0 / fs)
        x = np.sin(2 * np.pi * 50 * t)
        freqs, amps = compute_fft_spectrum(x, fs)
        params = calculate_motor_parameters(50.0, 4, 1470.0)
        res = eccentricity_fault_index(freqs, amps, params)
        assert "detection_status" in res
        ds = res["detection_status"]
        for field in (
            "detected",
            "reason",
            "fft_bin_width_hz",
            "tolerance_hz",
            "min_bin_width_for_tolerance_hz",
        ):
            assert field in ds
        assert isinstance(ds["detected"], bool)
        # Pinned enum: the implementation emits exactly these five values.
        # The earlier draft also listed ``below_noise_floor`` (per the
        # original issue body), but the implementation collapses the
        # below-floor case into ``no_sideband_present`` via the
        # ``_DETECTION_NOISE_FLOOR_DB`` guard. Removed from the allow-set
        # per code-review P1 to keep doc and code in lockstep.
        assert ds["reason"] in {
            "detected",
            "frequency_resolution_insufficient",
            "sideband_inside_supply_main_lobe",
            "no_sideband_present",
            "frequency_out_of_range",
        }


class TestDetectionStatusBackwardCompat:
    """Adding ``detection_status`` is purely additive — all v0.2.2
    consumers reading the legacy fields must see the same values they
    saw before."""

    def test_bearing_legacy_fields_unchanged(self):
        fs = 5000.0
        t = np.arange(0, 5.0, 1.0 / fs)
        x = np.sin(2 * np.pi * 50 * t)
        freqs, amps = compute_fft_spectrum(x, fs)
        res = bearing_fault_index(
            freqs, amps,
            supply_freq_hz=50.0,
            bearing_defect_freq_hz=132.0,
            defect_type="bpfi",
        )
        # The full v0.2.2 key set must still be present and well-typed.
        for legacy_key in (
            "fault_type", "defect_frequency_hz", "fundamental",
            "sidebands", "worst_sideband_db", "note",
        ):
            assert legacy_key in res

    def test_brb_legacy_fields_unchanged(self):
        fs = 5000.0
        t = np.arange(0, 5.0, 1.0 / fs)
        x = np.sin(2 * np.pi * 50 * t)
        freqs, amps = compute_fft_spectrum(x, fs)
        params = calculate_motor_parameters(50.0, 4, 1470.0)
        res = brb_fault_index(freqs, amps, params)
        for legacy_key in (
            "fault_type", "fundamental", "lower_sideband",
            "upper_sideband", "combined_index_db", "severity",
            "thresholds_db",
        ):
            assert legacy_key in res

    def test_eccentricity_legacy_fields_unchanged(self):
        fs = 5000.0
        t = np.arange(0, 5.0, 1.0 / fs)
        x = np.sin(2 * np.pi * 50 * t)
        freqs, amps = compute_fft_spectrum(x, fs)
        params = calculate_motor_parameters(50.0, 4, 1470.0)
        res = eccentricity_fault_index(freqs, amps, params)
        for legacy_key in (
            "fault_type", "fundamental", "sidebands",
            "worst_sideband_db", "severity", "thresholds_db",
        ):
            assert legacy_key in res


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
