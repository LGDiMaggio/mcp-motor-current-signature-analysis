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
