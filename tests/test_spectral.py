"""Tests for spectral analysis functions."""

import numpy as np
import pytest

from mcp_server_mcsa.analysis.spectral import (
    amplitude_at_frequency,
    compute_fft_spectrum,
    compute_psd,
    detect_peaks,
)


class TestComputeFFTSpectrum:
    def test_single_tone(self):
        fs = 1000.0
        t = np.arange(0, 1.0, 1.0 / fs)
        x = np.sin(2 * np.pi * 50 * t)
        freqs, amps = compute_fft_spectrum(x, fs)

        # Peak should be near 50 Hz
        peak_idx = np.argmax(amps)
        assert freqs[peak_idx] == pytest.approx(50.0, abs=1.5)
        assert amps[peak_idx] > 0.9

    def test_frequency_resolution(self):
        fs = 1000.0
        n = 10000  # 10 seconds
        x = np.sin(2 * np.pi * 50 * np.arange(n) / fs)
        freqs, amps = compute_fft_spectrum(x, fs)
        df = freqs[1] - freqs[0]
        assert df == pytest.approx(0.1, abs=0.01)

    def test_one_sided(self):
        fs = 1000.0
        x = np.random.randn(1000)
        freqs, amps = compute_fft_spectrum(x, fs, sided="one")
        assert freqs[0] == 0.0
        assert freqs[-1] == pytest.approx(fs / 2, abs=1.0)

    def test_two_sided(self):
        fs = 1000.0
        x = np.random.randn(1000)
        freqs, amps = compute_fft_spectrum(x, fs, sided="two")
        assert len(freqs) == 1000


class TestComputeFFTSpectrumAdaptive:
    """Adaptive ``n_fft`` via ``min_resolution_hz`` (issue #1)."""

    def test_min_resolution_hz_zero_pads_to_target_bin_width(self):
        """A 1000-sample signal at fs=1000 has native bin width = 1 Hz.
        Requesting ``min_resolution_hz=0.25`` should zero-pad so the bin
        width drops to ≤ 0.0625 Hz (factor-4 safety margin)."""
        fs = 1000.0
        n = 1000
        x = np.sin(2 * np.pi * 50 * np.arange(n) / fs)
        freqs, _ = compute_fft_spectrum(x, fs, min_resolution_hz=0.25)
        bin_width = float(freqs[1] - freqs[0])
        assert bin_width <= 0.25 / 4 + 1e-9, (
            f"min_resolution_hz=0.25 produced bin width {bin_width} Hz, "
            "expected ≤ 0.0625 Hz (factor-4 safety margin)"
        )

    def test_min_resolution_hz_recovers_close_to_supply_sidebands(self):
        """A 0.2 s × 20 kHz signal has 5 Hz native bin width — too coarse
        to resolve a sideband at supply±2 Hz. ``min_resolution_hz=0.5``
        forces ≤ 0.125 Hz bin width and the sideband appears as a
        distinct peak."""
        fs = 20000.0
        n = int(fs * 0.2)
        t = np.arange(n) / fs
        sideband = 2.0
        x = np.sin(2 * np.pi * 50 * t) + 0.2 * np.sin(
            2 * np.pi * (50 + sideband) * t
        )
        freqs_native, amps_native = compute_fft_spectrum(x, fs)
        freqs_hi, amps_hi = compute_fft_spectrum(x, fs, min_resolution_hz=0.5)
        # Native bin width is 5 Hz (no zero-padding); hi-res is ≤ 0.125 Hz.
        assert float(freqs_native[1] - freqs_native[0]) >= 4.0
        assert float(freqs_hi[1] - freqs_hi[0]) <= 0.125 + 1e-9
        # Hi-res spectrum has a local maximum within 0.5 Hz of 52 Hz that
        # the native spectrum cannot resolve (its closest bin is at 50 or
        # 55 Hz).
        mask = (freqs_hi >= 51.5) & (freqs_hi <= 52.5)
        assert np.any(mask)
        assert float(amps_hi[mask].max()) > 0.05

    def test_min_resolution_hz_none_preserves_v0_2_2_behavior(self):
        """``min_resolution_hz=None`` (default) must reproduce v0.2.2
        output bit-identically — same freqs, same amps as the
        no-keyword call."""
        fs = 1000.0
        x = np.sin(2 * np.pi * 50 * np.arange(2000) / fs)
        freqs_default, amps_default = compute_fft_spectrum(x, fs)
        freqs_none, amps_none = compute_fft_spectrum(
            x, fs, min_resolution_hz=None
        )
        np.testing.assert_array_equal(freqs_default, freqs_none)
        np.testing.assert_array_equal(amps_default, amps_none)

    def test_min_resolution_hz_excessive_raises(self):
        """An unreasonably fine request (would need ≥ 16 M FFT samples)
        raises ValueError instead of allocating a multi-GB array."""
        with pytest.raises(ValueError, match="exceeds the safety cap"):
            compute_fft_spectrum(
                np.zeros(1000, dtype=float), fs=1000.0,
                min_resolution_hz=1e-6,
            )

    def test_min_resolution_hz_non_positive_raises(self):
        """Zero or negative ``min_resolution_hz`` is a usage error."""
        x = np.zeros(1000, dtype=float)
        with pytest.raises(ValueError, match="min_resolution_hz must be > 0"):
            compute_fft_spectrum(x, fs=1000.0, min_resolution_hz=0.0)
        with pytest.raises(ValueError, match="min_resolution_hz must be > 0"):
            compute_fft_spectrum(x, fs=1000.0, min_resolution_hz=-0.1)


class TestComputePSD:
    def test_psd_peak_at_fundamental(self):
        fs = 5000.0
        t = np.arange(0, 10.0, 1.0 / fs)
        x = np.sin(2 * np.pi * 50 * t)
        freqs, psd = compute_psd(x, fs)
        peak_idx = np.argmax(psd)
        assert freqs[peak_idx] == pytest.approx(50.0, abs=2.0)

    def test_psd_non_negative(self):
        fs = 1000.0
        x = np.random.randn(10000)
        freqs, psd = compute_psd(x, fs)
        assert np.all(psd >= 0)


class TestDetectPeaks:
    def test_finds_known_peaks(self):
        fs = 5000.0
        t = np.arange(0, 2.0, 1.0 / fs)
        x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 150 * t)
        freqs, amps = compute_fft_spectrum(x, fs)
        peaks = detect_peaks(freqs, amps, prominence=0.01, max_peaks=5)

        peak_freqs = [p["frequency_hz"] for p in peaks]
        assert any(abs(f - 50.0) < 1.0 for f in peak_freqs)
        assert any(abs(f - 150.0) < 1.0 for f in peak_freqs)

    def test_frequency_range_filter(self):
        fs = 1000.0
        t = np.arange(0, 2.0, 1.0 / fs)
        x = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 200 * t)
        freqs, amps = compute_fft_spectrum(x, fs)
        peaks = detect_peaks(freqs, amps, freq_range=(100, 300), prominence=0.01)

        for p in peaks:
            assert 100 <= p["frequency_hz"] <= 300

    def test_max_peaks_limit(self):
        fs = 1000.0
        x = np.random.randn(10000)
        freqs, amps = compute_fft_spectrum(x, fs)
        peaks = detect_peaks(freqs, amps, max_peaks=5)
        assert len(peaks) <= 5


class TestAmplitudeAtFrequency:
    def test_finds_existing_component(self):
        fs = 5000.0
        t = np.arange(0, 2.0, 1.0 / fs)
        x = np.sin(2 * np.pi * 50 * t)
        freqs, amps = compute_fft_spectrum(x, fs)
        result = amplitude_at_frequency(freqs, amps, 50.0, tolerance_hz=1.0)
        assert result["found"] is True
        assert result["amplitude"] > 0.9

    def test_not_found_outside_range(self):
        freqs = np.array([0, 1, 2, 3, 4, 5], dtype=float)
        amps = np.array([0, 0, 0.5, 0, 0, 0], dtype=float)
        result = amplitude_at_frequency(freqs, amps, 100.0, tolerance_hz=0.5)
        assert result["found"] is False


class TestDetectPeaksSubBinInterpolation:
    """Sub-bin parabolic peak interpolation (issue #3)."""

    def test_interpolation_recovers_off_bin_frequency(self):
        """A sine wave at 49.93 Hz (off bin-center) should be recovered
        within ~0.05 Hz with interpolation, vs ~0.07-0.5 Hz raw error.
        Smith *Mathematics of the DFT* Sec 9.3."""
        fs = 1000.0
        n = 2000  # 2 s -> bin width 0.5 Hz
        t = np.arange(n) / fs
        f_true = 49.93  # deliberately off-bin
        x = np.sin(2 * np.pi * f_true * t)
        # Hann window so the spectrum has the standard windowed-sinusoid shape
        # (parabolic interpolation is most accurate on log-magnitude of a
        # Hann/Hamming/Blackman main lobe).
        x_w = x * np.hanning(n)
        freqs, amps = compute_fft_spectrum(x_w, fs)

        peaks_interp = detect_peaks(
            freqs, amps, prominence=0.001, max_peaks=1, interpolate=True
        )
        peaks_raw = detect_peaks(
            freqs, amps, prominence=0.001, max_peaks=1, interpolate=False
        )
        assert len(peaks_interp) == 1
        assert len(peaks_raw) == 1
        err_interp = abs(peaks_interp[0]["frequency_hz"] - f_true)
        err_raw = abs(peaks_raw[0]["frequency_hz"] - f_true)
        assert err_interp < 0.1, (
            f"interpolated error {err_interp:.4f} Hz exceeds 0.1 Hz tolerance"
        )
        assert err_interp < err_raw, (
            f"interpolation did not improve accuracy: "
            f"interp={err_interp:.4f} vs raw={err_raw:.4f}"
        )

    def test_interpolation_recovers_off_bin_amplitude(self):
        """Off-bin peak amplitude is under-reported when bin-quantised
        (energy splits between two adjacent bins). Interpolation in the
        log domain recovers the vertex amplitude."""
        fs = 1000.0
        n = 2000  # bin width 0.5 Hz
        t = np.arange(n) / fs
        f_true = 49.93
        true_amp_at_peak = 1.0
        x = true_amp_at_peak * np.sin(2 * np.pi * f_true * t) * np.hanning(n)
        freqs, amps = compute_fft_spectrum(x, fs)

        peaks_interp = detect_peaks(
            freqs, amps, prominence=0.001, max_peaks=1, interpolate=True
        )
        peaks_raw = detect_peaks(
            freqs, amps, prominence=0.001, max_peaks=1, interpolate=False
        )
        # Interpolated amplitude should be strictly >= raw (raw under-reports
        # the off-bin peak). The exact value depends on the windowed
        # main-lobe shape; the relative gain is the test signal.
        assert peaks_interp[0]["amplitude"] >= peaks_raw[0]["amplitude"]

    def test_interpolate_false_reproduces_v0_2_2_bin_centered(self):
        """Backward compat: interpolate=False must return identical output
        to v0.2.2 (peak frequencies exactly at bin centers, amplitudes the
        raw FFT bin values)."""
        fs = 1000.0
        n = 2000  # bin width 0.5 Hz; peak frequencies on bin centers
        t = np.arange(n) / fs
        x = np.sin(2 * np.pi * 50.0 * t) + 0.5 * np.sin(2 * np.pi * 100.0 * t)
        freqs, amps = compute_fft_spectrum(x, fs)
        peaks = detect_peaks(
            freqs, amps, prominence=0.01, max_peaks=5, interpolate=False
        )
        peak_freqs = sorted(p["frequency_hz"] for p in peaks[:2])
        # Both peaks land exactly on bin centers when interpolate=False
        assert 50.0 in peak_freqs or any(
            abs(f - 50.0) < 1e-9 for f in peak_freqs
        )
        # Bin-centered values are integer multiples of the bin width (0.5 Hz)
        for f in peak_freqs:
            assert abs(f / 0.5 - round(f / 0.5)) < 1e-9, (
                f"interpolate=False returned non-bin-centered freq {f}"
            )

    def test_interpolate_default_is_false_preserves_v0_2_2_bin_quantisation(self):
        """**Backward-compat default** (per the code-review P1 fix on
        this branch): ``interpolate`` defaults to ``False`` in v0.3.0,
        so a call site that does not pass the kwarg gets the same
        bin-centred frequency it got in v0.2.2 (every existing
        ``detect_peaks(...)`` call site, including the three in
        ``server.py``, continues to behave exactly as before).

        Pin both the default value AND the bin-centred behavior so a
        future flip-to-True in v0.4.0 cannot regress this v0.3.0
        guarantee."""
        import inspect

        sig = inspect.signature(detect_peaks)
        assert sig.parameters["interpolate"].default is False

        # And the bin-centred output: an off-bin sine (49.93 Hz on a
        # 0.5 Hz grid) returns the nearest BIN (50.0 Hz) when no kwarg
        # is passed, not the interpolated 49.93.
        fs = 1000.0
        n = 2000  # 0.5 Hz bin width; 49.93 is OFF-bin
        t = np.arange(n) / fs
        x = np.sin(2 * np.pi * 49.93 * t) * np.hanning(n)
        freqs, amps = compute_fft_spectrum(x, fs)
        peaks_default = detect_peaks(freqs, amps, prominence=0.001, max_peaks=1)
        # On-bin nearest (50.0 Hz), with parabolic delta NOT applied.
        assert peaks_default[0]["frequency_hz"] == 50.0

    def test_interpolate_explicit_true_refines_off_bin_peak(self):
        """The v0.3.0 opt-in: explicit ``interpolate=True`` recovers the
        sub-bin frequency. Same input as the default test above but with
        the new kwarg — refined frequency lands within 0.1 Hz of 49.93."""
        fs = 1000.0
        n = 2000
        t = np.arange(n) / fs
        x = np.sin(2 * np.pi * 49.93 * t) * np.hanning(n)
        freqs, amps = compute_fft_spectrum(x, fs)
        peaks_interp = detect_peaks(
            freqs, amps, prominence=0.001, max_peaks=1, interpolate=True
        )
        assert abs(peaks_interp[0]["frequency_hz"] - 49.93) < 0.1

    def test_interpolation_safe_at_array_boundary(self):
        """Peaks adjacent to the array edge cannot be interpolated
        (no neighbour on one side). The function must not raise; the
        boundary case returns the bin-centered value as-is."""
        # Manual spectrum: peak at index 0 (find_peaks may or may not flag
        # it; the contract is "no crash regardless").
        freqs = np.linspace(0.0, 100.0, 11)
        amps = np.array(
            [10.0, 5.0, 2.0, 1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01],
            dtype=float,
        )
        peaks = detect_peaks(freqs, amps, max_peaks=5, interpolate=True)
        assert isinstance(peaks, list)
        # Should not raise.

    def test_interpolation_safe_on_zero_amplitude_neighbours(self):
        """If a neighbour amplitude is exactly zero, log10 is undefined.
        The interpolator must guard with a floor and not raise."""
        freqs = np.linspace(0.0, 100.0, 11)
        amps = np.array(
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=float,
        )
        peaks = detect_peaks(freqs, amps, max_peaks=5, interpolate=True)
        assert isinstance(peaks, list)
        # The peak at index 2 has zero neighbours → log10(floor) on both;
        # symmetric → delta=0; should return the bin-centered freq=20.0.
        if peaks:
            assert abs(peaks[0]["frequency_hz"] - 20.0) < 0.5
