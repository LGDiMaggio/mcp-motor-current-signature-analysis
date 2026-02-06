"""Tests for envelope and time-frequency analysis."""

import numpy as np
import pytest

from mcp_server_mcsa.analysis.envelope import (
    envelope_spectrum,
    hilbert_envelope,
    instantaneous_frequency,
)
from mcp_server_mcsa.analysis.timefreq import (
    compute_spectrogram,
    compute_stft,
    track_frequency_over_time,
)


class TestHilbertEnvelope:
    def test_constant_envelope_for_sine(self):
        fs = 5000.0
        t = np.arange(0, 1.0, 1.0 / fs)
        x = np.sin(2 * np.pi * 50 * t)
        env = hilbert_envelope(x)
        # Envelope of a pure sine should be approximately constant ≈ 1
        assert np.std(env[100:-100]) < 0.05  # exclude edges

    def test_am_envelope_detection(self):
        fs = 5000.0
        t = np.arange(0, 2.0, 1.0 / fs)
        # AM signal: carrier 50 Hz, modulation 5 Hz
        mod = 1.0 + 0.5 * np.sin(2 * np.pi * 5 * t)
        x = mod * np.sin(2 * np.pi * 50 * t)
        env = hilbert_envelope(x)
        # Envelope should track the modulation
        assert np.max(env) > 1.2
        assert np.min(env[100:-100]) < 0.8


class TestInstantaneousFrequency:
    def test_constant_freq_for_sine(self):
        fs = 5000.0
        t = np.arange(0, 1.0, 1.0 / fs)
        x = np.sin(2 * np.pi * 100 * t)
        inst_f = instantaneous_frequency(x, fs)
        # Should be approximately 100 Hz
        assert np.mean(inst_f[100:-100]) == pytest.approx(100.0, abs=1.0)


class TestEnvelopeSpectrum:
    def test_detects_modulation_frequency(self):
        fs = 5000.0
        t = np.arange(0, 5.0, 1.0 / fs)
        mod = 1.0 + 0.3 * np.sin(2 * np.pi * 10 * t)
        x = mod * np.sin(2 * np.pi * 50 * t)
        freqs, amps = envelope_spectrum(x, fs)

        # Should find peak near 10 Hz in envelope spectrum
        idx_10 = np.argmin(np.abs(freqs - 10))
        nearby = amps[max(0, idx_10 - 2) : idx_10 + 3]
        assert np.max(nearby) > 0.05


class TestSTFT:
    def test_stft_output_shape(self):
        fs = 1000.0
        x = np.random.randn(5000)
        result = compute_stft(x, fs, nperseg=256)
        assert result["n_freq_bins"] > 0
        assert result["n_time_bins"] > 0
        assert result["magnitude"].shape == (
            result["n_freq_bins"],
            result["n_time_bins"],
        )

    def test_spectrogram_output(self):
        fs = 1000.0
        x = np.random.randn(5000)
        result = compute_spectrogram(x, fs, nperseg=256)
        assert "power" in result
        assert result["power"].shape[0] == result["n_freq_bins"]


class TestTrackFrequencyOverTime:
    def test_tracks_constant_tone(self):
        fs = 1000.0
        t = np.arange(0, 5.0, 1.0 / fs)
        x = np.sin(2 * np.pi * 50 * t)
        stft_result = compute_stft(x, fs, nperseg=256)
        tracking = track_frequency_over_time(stft_result, 50.0, tolerance_hz=5.0)

        assert tracking["found"] is True
        assert len(tracking["amplitude"]) == stft_result["n_time_bins"]
        # All amplitudes should be non-zero
        assert all(a > 0 for a in tracking["amplitude"])

    def test_not_found_outside_range(self):
        fs = 1000.0
        x = np.sin(2 * np.pi * 50 * np.arange(0, 5.0, 1.0 / fs))
        stft_result = compute_stft(x, fs, nperseg=256)
        tracking = track_frequency_over_time(stft_result, 400.0, tolerance_hz=1.0)
        # Frequency bin exists but amplitude should be negligible
        if tracking["found"]:
            assert all(a < 0.01 for a in tracking["amplitude"])
        else:
            assert tracking["found"] is False
