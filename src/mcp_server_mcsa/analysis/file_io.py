"""File I/O for loading real‑world motor‑current signals.

Supports CSV, WAV, NumPy binary (.npy), and TDMS formats commonly used
in industrial data‑acquisition systems.  Each loader returns a standardised
dictionary with ``signal``, ``sampling_freq_hz``, ``n_samples``, ``duration_s``
plus format‑specific metadata.
"""

from __future__ import annotations

import csv
import os
import struct
import wave
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def load_csv(
    file_path: str,
    signal_column: int | str = 1,
    time_column: int | str | None = 0,
    sampling_freq_hz: float | None = None,
    delimiter: str = ",",
    skip_header: int = 1,
    max_rows: int | None = None,
) -> dict:
    """Load a current signal from a CSV / TSV file.

    The CSV can contain a time column and one or more data columns.  The
    loader auto‑detects the sampling frequency from the time column unless
    ``sampling_freq_hz`` is provided explicitly.

    Args:
        file_path: Absolute or relative path to the CSV file.
        signal_column: Column index (0‑based int) or header name containing
            the current signal.
        time_column: Column index or header name for time (seconds).
            Set to ``None`` if the file has no time column (then
            ``sampling_freq_hz`` is required).
        sampling_freq_hz: Explicit sampling frequency.  If ``None``, it is
            inferred from the time column.
        delimiter: Column delimiter (default ``","``).
        skip_header: Number of header rows to skip (default 1).
        max_rows: Maximum number of data rows to read (``None`` → all).

    Returns:
        ``dict`` with keys: ``signal``, ``time_s`` (or ``None``),
        ``sampling_freq_hz``, ``n_samples``, ``duration_s``, ``file_path``,
        ``format``.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If sampling frequency cannot be determined.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # ---- Read raw rows ----
    rows: list[list[str]] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh, delimiter=delimiter)
        header_row: list[str] | None = None
        for i, row in enumerate(reader):
            if i < skip_header:
                header_row = row
                continue
            if max_rows is not None and len(rows) >= max_rows:
                break
            rows.append(row)

    if not rows:
        raise ValueError(f"No data rows found in {path}")

    # ---- Resolve column indices ----
    def _resolve_col(col: int | str | None, header: list[str] | None) -> int | None:
        if col is None:
            return None
        if isinstance(col, int):
            return col
        if header is not None:
            stripped = [h.strip() for h in header]
            if col in stripped:
                return stripped.index(col)
        raise ValueError(f"Column '{col}' not found in header: {header}")

    sig_idx = _resolve_col(signal_column, header_row)
    time_idx = _resolve_col(time_column, header_row)

    # ---- Parse numeric data ----
    signal_vals: list[float] = []
    time_vals: list[float] | None = [] if time_idx is not None else None

    for row in rows:
        try:
            signal_vals.append(float(row[sig_idx]))  # type: ignore[index]
            if time_vals is not None and time_idx is not None:
                time_vals.append(float(row[time_idx]))
        except (IndexError, ValueError):
            continue  # skip malformed rows

    signal = np.array(signal_vals, dtype=np.float64)

    # ---- Infer sampling frequency ----
    time_arr: NDArray[np.floating] | None = None
    if time_vals is not None and len(time_vals) > 1:
        time_arr = np.array(time_vals, dtype=np.float64)
        if sampling_freq_hz is None:
            dt = np.median(np.diff(time_arr))
            if dt <= 0:
                raise ValueError("Time column is not monotonically increasing")
            sampling_freq_hz = 1.0 / dt

    if sampling_freq_hz is None:
        raise ValueError(
            "Cannot determine sampling frequency — provide sampling_freq_hz "
            "or include a time column in the CSV."
        )

    n = len(signal)
    duration = n / sampling_freq_hz

    return {
        "signal": signal.tolist(),
        "time_s": time_arr.tolist() if time_arr is not None else None,
        "sampling_freq_hz": float(sampling_freq_hz),
        "n_samples": n,
        "duration_s": round(duration, 6),
        "file_path": str(path),
        "format": "csv",
    }


# ---------------------------------------------------------------------------
# WAV loader
# ---------------------------------------------------------------------------

def load_wav(
    file_path: str,
    channel: int = 0,
) -> dict:
    """Load a current signal from a WAV audio file.

    Many portable DAQ systems and low‑cost recorders save data as WAV.
    The signal is normalised to the full‑scale range of the bit depth.

    Args:
        file_path: Path to the WAV file.
        channel: Channel index for multi‑channel files (default 0).

    Returns:
        Standardised signal dict.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        fs = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    # Decode to numpy
    if sampwidth == 1:
        dtype = np.uint8
        max_val = 128.0
        offset = 128
    elif sampwidth == 2:
        dtype = np.int16
        max_val = 32768.0
        offset = 0
    elif sampwidth == 3:
        # 24-bit — unpack manually
        n_samples_total = len(raw) // 3
        unpacked = []
        for i in range(n_samples_total):
            b = raw[3 * i : 3 * i + 3]
            val = struct.unpack("<i", b + (b"\xff" if b[2] & 0x80 else b"\x00"))[0]
            unpacked.append(val)
        data = np.array(unpacked, dtype=np.float64)
        max_val = 8388608.0
        offset = 0
        dtype = None  # handled above
    elif sampwidth == 4:
        dtype = np.int32
        max_val = 2147483648.0
        offset = 0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth}")

    if dtype is not None:
        data = np.frombuffer(raw, dtype=dtype).astype(np.float64) - offset

    # De‑interleave channels
    if n_channels > 1:
        data = data.reshape(-1, n_channels)
        if channel >= n_channels:
            raise ValueError(f"Channel {channel} out of range (file has {n_channels})")
        data = data[:, channel]

    # Normalise to ±1
    signal = data / max_val

    n = len(signal)
    duration = n / fs

    return {
        "signal": signal.tolist(),
        "time_s": None,
        "sampling_freq_hz": float(fs),
        "n_samples": n,
        "duration_s": round(duration, 6),
        "file_path": str(path),
        "format": "wav",
        "metadata": {
            "channels": n_channels,
            "selected_channel": channel,
            "sample_width_bytes": sampwidth,
            "bit_depth": sampwidth * 8,
        },
    }


# ---------------------------------------------------------------------------
# NumPy binary loader
# ---------------------------------------------------------------------------

def load_npy(
    file_path: str,
    sampling_freq_hz: float,
    column: int = 0,
) -> dict:
    """Load a current signal from a NumPy ``.npy`` binary file.

    Args:
        file_path: Path to the ``.npy`` file.
        sampling_freq_hz: Sampling frequency (must be provided for raw arrays).
        column: If the array is 2‑D, select this column.

    Returns:
        Standardised signal dict.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    arr = np.load(str(path))

    if arr.ndim == 2:
        if column >= arr.shape[1]:
            raise ValueError(f"Column {column} out of range (array has {arr.shape[1]} cols)")
        signal = arr[:, column].astype(np.float64)
    elif arr.ndim == 1:
        signal = arr.astype(np.float64)
    else:
        raise ValueError(f"Expected 1‑D or 2‑D array, got {arr.ndim}‑D")

    n = len(signal)
    duration = n / sampling_freq_hz

    return {
        "signal": signal.tolist(),
        "time_s": None,
        "sampling_freq_hz": float(sampling_freq_hz),
        "n_samples": n,
        "duration_s": round(duration, 6),
        "file_path": str(path),
        "format": "npy",
        "metadata": {
            "original_shape": list(arr.shape),
            "selected_column": column if arr.ndim == 2 else None,
        },
    }


# ---------------------------------------------------------------------------
# Unified loader (auto‑detect by extension)
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".txt", ".wav", ".npy"}


def load_signal(
    file_path: str,
    sampling_freq_hz: float | None = None,
    signal_column: int | str = 1,
    time_column: int | str | None = 0,
    delimiter: str | None = None,
    channel: int = 0,
    skip_header: int = 1,
    max_rows: int | None = None,
) -> dict:
    """Auto‑detect file format and load a motor‑current signal.

    Dispatches to the appropriate specialised loader based on the file
    extension.

    Args:
        file_path: Path to the signal file.
        sampling_freq_hz: Sampling frequency (required for .npy; optional
            for CSV if a time column is present; auto‑detected for WAV).
        signal_column: CSV column (index or name) containing the signal.
        time_column: CSV column for time.  ``None`` → no time column.
        delimiter: CSV delimiter.  ``None`` → auto‑detect (``,`` for .csv,
            ``\\t`` for .tsv/.txt).
        channel: WAV channel index.
        skip_header: CSV header rows to skip.
        max_rows: Maximum rows to read from CSV.

    Returns:
        Standardised signal dictionary.
    """
    path = Path(file_path).resolve()
    ext = path.suffix.lower()

    if ext in (".csv", ".tsv", ".txt"):
        if delimiter is None:
            delimiter = "\t" if ext in (".tsv", ".txt") else ","
        return load_csv(
            str(path),
            signal_column=signal_column,
            time_column=time_column,
            sampling_freq_hz=sampling_freq_hz,
            delimiter=delimiter,
            skip_header=skip_header,
            max_rows=max_rows,
        )
    elif ext == ".wav":
        return load_wav(str(path), channel=channel)
    elif ext == ".npy":
        if sampling_freq_hz is None:
            raise ValueError("sampling_freq_hz is required for .npy files")
        return load_npy(str(path), sampling_freq_hz)
    else:
        raise ValueError(
            f"Unsupported file format: '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )


def get_signal_file_info(file_path: str) -> dict:
    """Return file metadata without fully loading the signal.

    Useful for inspecting large files before loading.

    Args:
        file_path: Path to the signal file.

    Returns:
        Dictionary with file size, format, estimated samples, etc.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    size_bytes = path.stat().st_size
    info: dict = {
        "file_path": str(path),
        "file_name": path.name,
        "extension": ext,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 2),
    }

    if ext == ".wav":
        try:
            with wave.open(str(path), "rb") as wf:
                info["format"] = "wav"
                info["channels"] = wf.getnchannels()
                info["sampling_freq_hz"] = wf.getframerate()
                info["n_samples"] = wf.getnframes()
                info["duration_s"] = round(wf.getnframes() / wf.getframerate(), 3)
                info["sample_width_bytes"] = wf.getsampwidth()
                info["bit_depth"] = wf.getsampwidth() * 8
        except Exception as e:
            info["error"] = str(e)
    elif ext == ".npy":
        try:
            # Read only the header — try public API first, fall back to private
            with open(path, "rb") as fh:
                version = np.lib.format.read_magic(fh)
                try:
                    shape, fortran, dtype = np.lib.format._read_array_header(fh, version)  # type: ignore[attr-defined]
                except AttributeError:
                    # NumPy >= 2.0 removed the private helper
                    fh.seek(0)
                    _ = np.lib.format.read_magic(fh)
                    header = np.lib.format.read_array_header_1_0(fh) if version[0] == 1 else np.lib.format.read_array_header_2_0(fh)
                    shape, fortran, dtype = header
            info["format"] = "npy"
            info["shape"] = list(shape)
            info["dtype"] = str(dtype)
            info["n_samples"] = shape[0] if len(shape) >= 1 else 0
        except Exception as e:
            info["error"] = str(e)
    elif ext in (".csv", ".tsv", ".txt"):
        try:
            # Count lines and peek at header
            with open(path, encoding="utf-8-sig") as fh:
                first_line = fh.readline().strip()
                second_line = fh.readline().strip()
                line_count = 2
                for _ in fh:
                    line_count += 1
            info["format"] = "csv"
            info["header_line"] = first_line
            info["sample_data_line"] = second_line
            info["total_lines"] = line_count
            info["estimated_data_rows"] = max(0, line_count - 1)
        except Exception as e:
            info["error"] = str(e)
    else:
        info["format"] = "unknown"

    return info
