"""Tests for signal preprocessing functions."""

import numpy as np
import pytest

from mcp_server_mcsa.analysis.preprocessing import (
    apply_window,
    bandpass_filter,
    lowpass_filter,
    notch_filter,
    normalize_signal,
    preprocess_pipeline,
    remove_dc_offset,
)


class TestRemoveDcOffset:
    def test_removes_mean(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = remove_dc_offset(x)
        assert np.mean(y) == pytest.approx(0.0, abs=1e-12)

    def test_zero_mean_unchanged(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        y = remove_dc_offset(x)
        np.testing.assert_array_almost_equal(y, x)


class TestNormalizeSignal:
    def test_rms_normalisation(self):
        x = np.sin(2 * np.pi * 50 * np.arange(1000) / 1000)
        y = normalize_signal(x)
        rms = np.sqrt(np.mean(y ** 2))
        assert rms == pytest.approx(1.0, abs=1e-6)

    def test_nominal_current_normalisation(self):
        x = np.array([10.0, 20.0, 30.0])
        y = normalize_signal(x, nominal_current=10.0)
        np.testing.assert_array_almost_equal(y, [1.0, 2.0, 3.0])

    def test_zero_signal(self):
        x = np.zeros(100)
        y = normalize_signal(x)
        np.testing.assert_array_equal(y, x)


class TestApplyWindow:
    def test_hann_reduces_edges(self):
        x = np.ones(100)
        y = apply_window(x, "hann")
        assert y[0] == pytest.approx(0.0, abs=1e-3)
        assert y[-1] < 0.01  # near-zero at edges
        assert y[50] > 0.9

    def test_rectangular_unchanged(self):
        x = np.ones(100)
        y = apply_window(x, "rectangular")
        np.testing.assert_array_equal(y, x)


class TestFilters:
    def test_notch_removes_frequency(self):
        fs = 5000.0
        t = np.arange(0, 1.0, 1.0 / fs)
        x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 100 * t)
        y = notch_filter(x, fs, 100.0, quality_factor=30)
        # 100 Hz component should be significantly reduced
        fft_orig = np.abs(np.fft.rfft(x))
        fft_filt = np.abs(np.fft.rfft(y))
        freqs = np.fft.rfftfreq(len(x), 1.0 / fs)
        idx_100 = np.argmin(np.abs(freqs - 100))
        assert fft_filt[idx_100] < fft_orig[idx_100] * 0.3

    def test_bandpass_limits_range(self):
        fs = 5000.0
        t = np.arange(0, 1.0, 1.0 / fs)
        x = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 500 * t)
        y = bandpass_filter(x, fs, 30.0, 100.0)
        # Should keep 50 Hz, reduce 500 Hz
        fft_y = np.abs(np.fft.rfft(y))
        freqs = np.fft.rfftfreq(len(y), 1.0 / fs)
        idx_50 = np.argmin(np.abs(freqs - 50))
        idx_500 = np.argmin(np.abs(freqs - 500))
        assert fft_y[idx_50] > fft_y[idx_500] * 10

    def test_bandpass_invalid_raises(self):
        x = np.zeros(1000)
        with pytest.raises(ValueError):
            bandpass_filter(x, 1000.0, 600.0, 100.0)  # low > high

    def test_lowpass(self):
        fs = 5000.0
        t = np.arange(0, 1.0, 1.0 / fs)
        x = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 1000 * t)
        y = lowpass_filter(x, fs, 200.0)
        fft_y = np.abs(np.fft.rfft(y))
        freqs = np.fft.rfftfreq(len(y), 1.0 / fs)
        idx_1000 = np.argmin(np.abs(freqs - 1000))
        idx_50 = np.argmin(np.abs(freqs - 50))
        assert fft_y[idx_50] > fft_y[idx_1000] * 10


class TestPreprocessPipeline:
    def test_basic_pipeline(self):
        fs = 5000.0
        t = np.arange(0, 1.0, 1.0 / fs)
        x = 5.0 + np.sin(2 * np.pi * 50 * t)  # DC offset = 5
        y = preprocess_pipeline(x, fs)
        # DC should be removed
        assert np.abs(np.mean(y)) < 1.0  # approximate due to windowing

    def test_pipeline_with_notch(self):
        fs = 5000.0
        t = np.arange(0, 1.0, 1.0 / fs)
        x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 150 * t)
        y = preprocess_pipeline(x, fs, notch_freqs=[150.0])
        assert len(y) == len(x)
