"""Tests for the file I/O module."""

from __future__ import annotations

import csv
import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from mcp_server_mcsa.analysis.file_io import (
    get_signal_file_info,
    load_csv,
    load_npy,
    load_signal,
    load_wav,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

FS = 8000.0
DURATION = 0.5
N = int(FS * DURATION)


@pytest.fixture()
def sine_signal() -> np.ndarray:
    """Simple 60 Hz sine at unit amplitude."""
    t = np.arange(N) / FS
    return np.sin(2 * np.pi * 60 * t)


@pytest.fixture()
def csv_file(tmp_path: Path, sine_signal: np.ndarray) -> Path:
    """CSV file with time + signal columns."""
    fp = tmp_path / "signal.csv"
    t = np.arange(N) / FS
    with open(fp, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "current"])
        for ti, xi in zip(t, sine_signal):
            writer.writerow([f"{ti:.8f}", f"{xi:.8f}"])
    return fp


@pytest.fixture()
def csv_file_no_time(tmp_path: Path, sine_signal: np.ndarray) -> Path:
    """CSV with only signal column (no time)."""
    fp = tmp_path / "signal_only.csv"
    with open(fp, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["current"])
        for xi in sine_signal:
            writer.writerow([f"{xi:.8f}"])
    return fp


@pytest.fixture()
def tsv_file(tmp_path: Path, sine_signal: np.ndarray) -> Path:
    """TSV file with time + signal columns."""
    fp = tmp_path / "signal.tsv"
    t = np.arange(N) / FS
    with open(fp, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["time", "current"])
        for ti, xi in zip(t, sine_signal):
            writer.writerow([f"{ti:.8f}", f"{xi:.8f}"])
    return fp


@pytest.fixture()
def wav_file(tmp_path: Path, sine_signal: np.ndarray) -> Path:
    """16-bit WAV file."""
    fp = tmp_path / "signal.wav"
    scaled = np.clip(sine_signal * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(fp), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(FS))
        wf.writeframes(scaled.tobytes())
    return fp


@pytest.fixture()
def wav_stereo(tmp_path: Path, sine_signal: np.ndarray) -> Path:
    """Stereo 16-bit WAV."""
    fp = tmp_path / "stereo.wav"
    scaled = np.clip(sine_signal * 32767, -32768, 32767).astype(np.int16)
    interleaved = np.column_stack([scaled, scaled]).flatten()
    with wave.open(str(fp), "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(int(FS))
        wf.writeframes(interleaved.tobytes())
    return fp


@pytest.fixture()
def npy_file(tmp_path: Path, sine_signal: np.ndarray) -> Path:
    """NumPy .npy file."""
    fp = tmp_path / "signal.npy"
    np.save(fp, sine_signal)
    return fp


# ---------------------------------------------------------------------------
# Tests – CSV
# ---------------------------------------------------------------------------


class TestLoadCSV:
    def test_csv_with_time(self, csv_file: Path, sine_signal: np.ndarray) -> None:
        result = load_csv(str(csv_file), signal_column=1, time_column=0)
        assert result["format"] == "csv"
        assert result["n_samples"] == N
        np.testing.assert_allclose(result["sampling_freq_hz"], FS, rtol=0.01)
        sig = np.array(result["signal"])
        np.testing.assert_allclose(sig, sine_signal, atol=1e-6)

    def test_csv_no_time(self, csv_file_no_time: Path) -> None:
        result = load_csv(str(csv_file_no_time), signal_column=0, time_column=None,
                          sampling_freq_hz=FS)
        assert result["n_samples"] == N
        assert result["sampling_freq_hz"] == FS

    def test_csv_no_time_no_fs_raises(self, csv_file_no_time: Path) -> None:
        with pytest.raises(ValueError, match="sampling_freq_hz"):
            load_csv(str(csv_file_no_time), signal_column=0, time_column=None)

    def test_tsv(self, tsv_file: Path, sine_signal: np.ndarray) -> None:
        result = load_csv(str(tsv_file), signal_column=1, time_column=0, delimiter="\t")
        assert result["format"] == "csv"
        assert result["n_samples"] == N

    def test_csv_by_column_name(self, csv_file: Path) -> None:
        result = load_csv(str(csv_file), signal_column="current", time_column="time")
        assert result["n_samples"] == N

    def test_csv_bad_column_raises(self, csv_file: Path) -> None:
        with pytest.raises((ValueError, KeyError)):
            load_csv(str(csv_file), signal_column="nonexistent")


# ---------------------------------------------------------------------------
# Tests – WAV
# ---------------------------------------------------------------------------


class TestLoadWAV:
    def test_wav_mono(self, wav_file: Path, sine_signal: np.ndarray) -> None:
        result = load_wav(str(wav_file))
        assert result["format"] == "wav"
        assert result["sampling_freq_hz"] == FS
        assert result["n_samples"] == N
        sig = np.array(result["signal"])
        # 16-bit quantisation error
        np.testing.assert_allclose(sig, sine_signal, atol=1e-3)

    def test_wav_stereo_channel0(self, wav_stereo: Path) -> None:
        result = load_wav(str(wav_stereo), channel=0)
        assert result["n_samples"] == N

    def test_wav_stereo_bad_channel(self, wav_stereo: Path) -> None:
        with pytest.raises(ValueError, match="Channel"):
            load_wav(str(wav_stereo), channel=5)


# ---------------------------------------------------------------------------
# Tests – NPY
# ---------------------------------------------------------------------------


class TestLoadNPY:
    def test_npy(self, npy_file: Path, sine_signal: np.ndarray) -> None:
        result = load_npy(str(npy_file), sampling_freq_hz=FS)
        assert result["format"] == "npy"
        assert result["n_samples"] == N
        np.testing.assert_allclose(result["signal"], sine_signal.tolist(), atol=1e-12)

    def test_npy_no_fs_raises(self, npy_file: Path) -> None:
        """load_npy requires sampling_freq_hz as a positional argument."""
        with pytest.raises(TypeError):
            load_npy(str(npy_file))  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Tests – auto-detect (load_signal)
# ---------------------------------------------------------------------------


class TestLoadSignal:
    def test_auto_csv(self, csv_file: Path) -> None:
        result = load_signal(str(csv_file))
        assert result["format"] == "csv"

    def test_auto_wav(self, wav_file: Path) -> None:
        result = load_signal(str(wav_file))
        assert result["format"] == "wav"

    def test_auto_npy(self, npy_file: Path) -> None:
        result = load_signal(str(npy_file), sampling_freq_hz=FS)
        assert result["format"] == "npy"

    def test_unsupported_format(self, tmp_path: Path) -> None:
        fp = tmp_path / "data.xyz"
        fp.write_text("garbage")
        with pytest.raises(ValueError, match="Unsupported"):
            load_signal(str(fp))


# ---------------------------------------------------------------------------
# Tests – file info
# ---------------------------------------------------------------------------


class TestGetSignalFileInfo:
    def test_csv_info(self, csv_file: Path) -> None:
        info = get_signal_file_info(str(csv_file))
        assert info["format"] == "csv"
        assert info["size_bytes"] > 0
        assert "header_line" in info

    def test_wav_info(self, wav_file: Path) -> None:
        info = get_signal_file_info(str(wav_file))
        assert info["format"] == "wav"
        assert info["channels"] == 1

    def test_npy_info(self, npy_file: Path) -> None:
        info = get_signal_file_info(str(npy_file))
        assert info["format"] == "npy"
        assert "shape" in info

    def test_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            get_signal_file_info("/nonexistent/file.csv")
