"""MCP Server for Motor Current Signature Analysis (MCSA).

Provides tools for spectral analysis, fault frequency computation,
fault detection, and diagnostic assessment of electric‑motor stator
currents via the Model Context Protocol.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

import numpy as np
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from mcp_server_mcsa.analysis.bearing import (
    BearingGeometry,
    bearing_current_sidebands,
    calculate_bearing_defect_frequencies,
)
from mcp_server_mcsa.analysis.envelope import (
    envelope_spectrum,
    hilbert_envelope,
)
from mcp_server_mcsa.analysis.fault_detection import (
    band_energy_index,
    bearing_fault_index,
    brb_fault_index,
    eccentricity_fault_index,
    envelope_statistical_indices,
    stator_fault_index,
)
from mcp_server_mcsa.analysis.file_io import (
    get_signal_file_info,
    load_signal,
)
from mcp_server_mcsa.analysis.motor import (
    calculate_fault_frequencies,
    calculate_motor_parameters,
)
from mcp_server_mcsa.analysis.preprocessing import preprocess_pipeline
from mcp_server_mcsa.analysis.spectral import (
    compute_fft_spectrum,
    compute_psd,
    detect_peaks,
)
from mcp_server_mcsa.analysis.test_signal import generate_test_signal
from mcp_server_mcsa.analysis.timefreq import (
    compute_stft,
    track_frequency_over_time,
)

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "mcsa",
    instructions=(
        "Motor Current Signature Analysis (MCSA) server. "
        "Provides tools for spectral analysis of electric‑motor stator "
        "currents to detect rotor, stator, bearing, and load faults. "
        "IMPORTANT: Signals and spectra are persisted to disk (~/.mcsa_data/) "
        "and referenced by short IDs (e.g. sig_xxxx, spec_xxxx).  "
        "Data survives server restarts.  Producer tools (generate_test_current_signal, "
        "load_signal_from_file, preprocess_signal, compute_spectrum, compute_power_spectral_density) "
        "return a short ID.  Pass that ID to downstream tools "
        "instead of raw arrays.  Use list_stored_data to see what is stored "
        "and clear_stored_data to free disk space. "
        "Typical workflow: "
        "(1) load or generate a signal → get signal_id, "
        "(2) preprocess → get preprocessed signal_id, "
        "(3) compute spectrum → get spectrum_id, "
        "(4) detect faults passing spectrum_id, "
        "(5) or use run_full_diagnosis / diagnose_from_file for one-shot analysis. "
        "For real‑world signals use inspect_signal_file first to check "
        "the file format, then load_signal_from_file to read the data."
    ),
)


# ---------------------------------------------------------------------------
# Server-side data store — persisted to ~/.mcsa_data/ (configurable via
# MCSA_DATA_DIR env var).  Large arrays are stored as compressed .npz files
# so they survive server restarts.  A lightweight in-memory cache avoids
# redundant disk reads within a session.
# ---------------------------------------------------------------------------

_DATA_DIR = Path(os.environ.get("MCSA_DATA_DIR", Path.home() / ".mcsa_data"))
_SIGNALS_DIR = _DATA_DIR / "signals"
_SPECTRA_DIR = _DATA_DIR / "spectra"

# Ensure dirs exist at import time
_SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
_SPECTRA_DIR.mkdir(parents=True, exist_ok=True)

# In-memory cache: id → dict (same shape as before, but arrays loaded lazily)
_cache: dict[str, dict] = {}


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:8]}"


# --- Persist / load helpers ------------------------------------------------

def _signal_path(sid: str) -> Path:
    return _SIGNALS_DIR / f"{sid}.npz"


def _spectrum_path(sid: str) -> Path:
    return _SPECTRA_DIR / f"{sid}.npz"


def _store_signal(
    signal: np.ndarray | list,
    sampling_freq_hz: float,
    **metadata: object,
) -> str:
    """Store a signal to disk and cache, return a short reference ID."""
    sid = _new_id("sig")
    arr = np.asarray(signal, dtype=np.float64)
    # Serialisable metadata (no numpy arrays)
    meta_clean = {k: v for k, v in metadata.items() if not isinstance(v, np.ndarray)}
    np.savez_compressed(
        _signal_path(sid),
        signal=arr,
        sampling_freq_hz=np.float64(sampling_freq_hz),
        meta_json=json.dumps(meta_clean, default=str),
    )
    _cache[sid] = {
        "_type": "signal",
        "signal": arr,
        "sampling_freq_hz": float(sampling_freq_hz),
        "n_samples": len(arr),
        **meta_clean,
    }
    return sid


def _store_spectrum(
    frequencies_hz: np.ndarray | list,
    amplitudes: np.ndarray | list,
    **metadata: object,
) -> str:
    """Store a spectrum to disk and cache, return a short reference ID."""
    sid = _new_id("spec")
    freqs = np.asarray(frequencies_hz, dtype=np.float64)
    amps = np.asarray(amplitudes, dtype=np.float64)
    meta_clean = {k: v for k, v in metadata.items() if not isinstance(v, np.ndarray)}
    np.savez_compressed(
        _spectrum_path(sid),
        frequencies_hz=freqs,
        amplitudes=amps,
        meta_json=json.dumps(meta_clean, default=str),
    )
    _cache[sid] = {
        "_type": "spectrum",
        "frequencies_hz": freqs,
        "amplitudes": amps,
        "n_bins": len(freqs),
        **meta_clean,
    }
    return sid


def _load_signal_from_disk(sid: str) -> dict:
    """Load a signal .npz from disk into the cache."""
    path = _signal_path(sid)
    if not path.exists():
        raise FileNotFoundError(f"Signal file not found: {path}")
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta_json"])) if "meta_json" in data else {}
    arr = data["signal"]
    fs = float(data["sampling_freq_hz"])
    entry = {
        "_type": "signal",
        "signal": arr,
        "sampling_freq_hz": fs,
        "n_samples": len(arr),
        **meta,
    }
    _cache[sid] = entry
    return entry


def _load_spectrum_from_disk(sid: str) -> dict:
    """Load a spectrum .npz from disk into the cache."""
    path = _spectrum_path(sid)
    if not path.exists():
        raise FileNotFoundError(f"Spectrum file not found: {path}")
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta_json"])) if "meta_json" in data else {}
    freqs = data["frequencies_hz"]
    amps = data["amplitudes"]
    entry = {
        "_type": "spectrum",
        "frequencies_hz": freqs,
        "amplitudes": amps,
        "n_bins": len(freqs),
        **meta,
    }
    _cache[sid] = entry
    return entry


def _resolve_entry(data_id: str) -> dict:
    """Get an entry from cache or load from disk on demand."""
    if data_id in _cache:
        return _cache[data_id]
    if data_id.startswith("sig_"):
        return _load_signal_from_disk(data_id)
    if data_id.startswith("spec_"):
        return _load_spectrum_from_disk(data_id)
    raise ValueError(f"Unknown data ID: {data_id}")


def _get_signal(
    signal_id: str | None = None,
    signal: list[float] | None = None,
    sampling_freq_hz: float | None = None,
) -> tuple[np.ndarray, float]:
    """Resolve a signal from store ID or raw array."""
    if signal_id:
        entry = _resolve_entry(signal_id)
        return entry["signal"].copy(), entry["sampling_freq_hz"]
    if signal is not None:
        if sampling_freq_hz is None:
            raise ValueError("sampling_freq_hz is required when passing a raw signal array")
        return np.array(signal, dtype=np.float64), sampling_freq_hz
    raise ValueError(
        "Provide either signal_id (from generate_test_current_signal, "
        "load_signal_from_file, or preprocess_signal) or a raw signal "
        "array with sampling_freq_hz."
    )


def _get_spectrum(
    spectrum_id: str | None = None,
    frequencies_hz: list[float] | None = None,
    amplitudes: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a spectrum from store ID or raw arrays."""
    if spectrum_id:
        entry = _resolve_entry(spectrum_id)
        return entry["frequencies_hz"].copy(), entry["amplitudes"].copy()
    if frequencies_hz is not None and amplitudes is not None:
        return (
            np.array(frequencies_hz, dtype=np.float64),
            np.array(amplitudes, dtype=np.float64),
        )
    raise ValueError(
        "Provide either spectrum_id (from compute_spectrum or "
        "compute_power_spectral_density) or raw frequency/amplitude arrays."
    )


def _list_all_stored_ids() -> list[str]:
    """List all signal and spectrum IDs persisted on disk."""
    ids = []
    for f in _SIGNALS_DIR.glob("sig_*.npz"):
        ids.append(f.stem)
    for f in _SPECTRA_DIR.glob("spec_*.npz"):
        ids.append(f.stem)
    return sorted(ids)


def _signal_summary(arr: np.ndarray, fs: float) -> dict:
    """Compact statistical summary of a signal (no raw data)."""
    return {
        "n_samples": len(arr),
        "duration_s": round(len(arr) / fs, 3),
        "sampling_freq_hz": fs,
        "rms": round(float(np.sqrt(np.mean(arr**2))), 6),
        "peak_amplitude": round(float(np.max(np.abs(arr))), 6),
        "mean": round(float(np.mean(arr)), 6),
        "std": round(float(np.std(arr)), 6),
    }


def _spectrum_summary(freqs: np.ndarray, amps: np.ndarray) -> dict:
    """Compact summary of a spectrum (no raw data)."""
    top_idx = np.argsort(amps)[::-1][:5]
    top_peaks = [
        {"freq_hz": round(float(freqs[i]), 3), "amplitude": round(float(amps[i]), 6)}
        for i in top_idx if amps[i] > 0
    ]
    return {
        "n_bins": len(freqs),
        "freq_range_hz": [round(float(freqs[0]), 3), round(float(freqs[-1]), 3)],
        "freq_resolution_hz": round(float(freqs[1] - freqs[0]), 6) if len(freqs) > 1 else 0,
        "max_amplitude": round(float(np.max(amps)), 6),
        "top_5_peaks": top_peaks,
    }


# ===================================================================
# RESOURCE: Fault Signatures Reference
# ===================================================================

FAULT_SIGNATURES_REFERENCE = """# MCSA Fault Signature Reference

## Broken Rotor Bars (BRB)
- **Signature**: Sidebands at (1 ± 2s)·f_s around the supply fundamental
- **Index**: dB ratio of sideband amplitude to fundamental
- **Thresholds** (dB below fundamental):
  - Healthy: ≤ -50 dB
  - Incipient: -50 to -45 dB
  - Moderate: -45 to -40 dB
  - Severe: > -35 dB
- **Notes**: More visible at medium–high load; higher harmonics at (1 ± 2ks)·f_s

## Eccentricity (Static / Dynamic)
- **Signature**: Sidebands at f_s ± k·f_r (rotor frequency multiples)
- **Static eccentricity**: produces components at f_s ± f_r
- **Dynamic eccentricity**: produces components at f_s ± k·f_r, varying with load
- **Mixed eccentricity**: components at n·f_r (pure rotational harmonics)

## Stator Inter-Turn Short Circuit
- **Signature**: Sidebands at f_s ± 2k·f_r
- **Notes**: May also increase negative-sequence current component;
  distinguish from supply unbalance by checking load dependency

## Bearing Defects
- **Signature**: Sidebands at f_s ± k·f_defect where f_defect is BPFO/BPFI/BSF/FTF
- **Defect frequencies** (normalised to shaft speed):
  - BPFO = (n/2)·(1 - d/D·cos α)
  - BPFI = (n/2)·(1 + d/D·cos α)
  - BSF  = (D/2d)·(1 - (d/D·cos α)²)
  - FTF  = (1/2)·(1 - d/D·cos α)
- **Notes**: Weak in stator current; confirm with envelope analysis or vibration data

## Load Faults (Cavitation, Misalignment)
- **Signature**: Broadband energy increase around f_s ("foot" pattern in PSD)
- **Index**: Band energy integration around the supply frequency
"""


@mcp.resource("mcsa://fault-signatures")
def fault_signatures_resource() -> str:
    """Reference table of MCSA fault signatures, frequencies, and empirical thresholds."""
    return FAULT_SIGNATURES_REFERENCE


# ===================================================================
# TOOL 1: Calculate Motor Parameters
# ===================================================================

@mcp.tool()
def calculate_motor_params(
    supply_freq_hz: Annotated[float, Field(description="Supply (line) frequency in Hz, e.g. 50 or 60")],
    poles: Annotated[int, Field(description="Number of magnetic poles (even, ≥ 2)")],
    rotor_speed_rpm: Annotated[float, Field(description="Measured rotor speed in RPM")],
) -> str:
    """Calculate motor operating parameters from nameplate and measured data.

    Computes synchronous speed, slip, rotor frequency, and slip frequency.
    These parameters are required inputs for fault frequency calculations.
    """
    params = calculate_motor_parameters(supply_freq_hz, poles, rotor_speed_rpm)
    return json.dumps(params.to_dict(), indent=2)


# ===================================================================
# TOOL 2: Calculate Fault Frequencies
# ===================================================================

@mcp.tool()
def compute_fault_frequencies(
    supply_freq_hz: Annotated[float, Field(description="Supply frequency in Hz")],
    poles: Annotated[int, Field(description="Number of magnetic poles")],
    rotor_speed_rpm: Annotated[float, Field(description="Rotor speed in RPM")],
    harmonics: Annotated[int, Field(description="Number of harmonic orders to compute", default=3)] = 3,
) -> str:
    """Calculate expected fault frequencies for common induction-motor faults.

    Computes characteristic frequencies for broken rotor bars, eccentricity,
    stator faults, and mixed eccentricity based on motor operating parameters.
    Use these frequencies to know WHERE to look in the current spectrum.
    """
    params = calculate_motor_parameters(supply_freq_hz, poles, rotor_speed_rpm)
    result = calculate_fault_frequencies(params, harmonics)
    return json.dumps(result, indent=2, default=str)


# ===================================================================
# TOOL 3: Calculate Bearing Defect Frequencies
# ===================================================================

@mcp.tool()
def compute_bearing_frequencies(
    n_balls: Annotated[int, Field(description="Number of rolling elements in the bearing")],
    ball_dia_mm: Annotated[float, Field(description="Ball (roller) diameter in mm")],
    pitch_dia_mm: Annotated[float, Field(description="Pitch (cage) diameter in mm")],
    contact_angle_deg: Annotated[float, Field(description="Contact angle in degrees", default=0.0)] = 0.0,
    shaft_speed_rpm: Annotated[float, Field(description="Shaft speed in RPM (optional, for absolute Hz)", default=0.0)] = 0.0,
    supply_freq_hz: Annotated[float, Field(description="Supply frequency in Hz (optional, for current sidebands)", default=0.0)] = 0.0,
) -> str:
    """Calculate bearing characteristic defect frequencies (BPFO, BPFI, BSF, FTF).

    Returns normalised frequencies (multiples of shaft speed) and, if shaft
    speed is provided, absolute frequencies in Hz. If supply frequency is
    also given, computes expected stator-current sidebands.
    """
    geom = BearingGeometry(n_balls, ball_dia_mm, pitch_dia_mm, contact_angle_deg)
    defects = calculate_bearing_defect_frequencies(geom)

    result: dict = {
        "bearing_geometry": {
            "n_balls": n_balls,
            "ball_dia_mm": ball_dia_mm,
            "pitch_dia_mm": pitch_dia_mm,
            "contact_angle_deg": contact_angle_deg,
        },
        "normalised_to_shaft_speed": defects.to_dict(),
    }

    if shaft_speed_rpm > 0:
        shaft_freq = shaft_speed_rpm / 60.0
        result["absolute_frequencies_hz"] = defects.absolute(shaft_freq)

        if supply_freq_hz > 0:
            result["current_sidebands"] = bearing_current_sidebands(
                defects, shaft_freq, supply_freq_hz
            )

    return json.dumps(result, indent=2)


# ===================================================================
# TOOL 4: Preprocess Signal
# ===================================================================

@mcp.tool()
def preprocess_signal(
    signal: Annotated[list[float], Field(description="Raw current signal as a list of amplitude values")],
    sampling_freq_hz: Annotated[float, Field(description="Sampling frequency in Hz")],
    nominal_current: Annotated[float | None, Field(description="Nominal current for normalisation (A). Omit for RMS normalisation", default=None)] = None,
    window: Annotated[str, Field(description="Window function: hann, hamming, blackman, flattop, rectangular", default="hann")] = "hann",
    bandpass_low_hz: Annotated[float | None, Field(description="Lower bandpass cutoff in Hz (optional)", default=None)] = None,
    bandpass_high_hz: Annotated[float | None, Field(description="Upper bandpass cutoff in Hz (optional)", default=None)] = None,
    notch_freqs_hz: Annotated[list[float] | None, Field(description="Frequencies to notch-filter (optional)", default=None)] = None,
) -> str:
    """Preprocess a stator-current signal for spectral analysis.

    Applies (in order): DC offset removal → notch filtering → bandpass
    filtering → normalisation → windowing. Returns the preprocessed signal.
    """
    x = np.array(signal, dtype=np.float64)

    bandpass = None
    if bandpass_low_hz is not None and bandpass_high_hz is not None:
        bandpass = (bandpass_low_hz, bandpass_high_hz)

    y = preprocess_pipeline(
        x, sampling_freq_hz,
        nominal_current=nominal_current,
        window=window,  # type: ignore[arg-type]
        bandpass=bandpass,
        notch_freqs=notch_freqs_hz,
    )

    return json.dumps({
        "preprocessed_signal": y.tolist(),
        "n_samples": len(y),
        "sampling_freq_hz": sampling_freq_hz,
        "steps_applied": [
            "dc_offset_removal",
            *(["notch_filter"] if notch_freqs_hz else []),
            *(["bandpass_filter"] if bandpass else []),
            "normalisation",
            f"window_{window}",
        ],
    })


# ===================================================================
# TOOL 5: Compute FFT Spectrum
# ===================================================================

@mcp.tool()
def compute_spectrum(
    signal_id: Annotated[str | None, Field(description="ID of a stored signal. Preferred over raw array.", default=None)] = None,
    signal: Annotated[list[float] | None, Field(description="Time-domain signal array. Use signal_id instead for large signals.", default=None)] = None,
    sampling_freq_hz: Annotated[float | None, Field(description="Sampling frequency in Hz. Auto-resolved when using signal_id.", default=None)] = None,
    n_fft: Annotated[int | None, Field(description="FFT length (zero-padding). Omit for auto", default=None)] = None,
    max_freq_hz: Annotated[float | None, Field(description="Maximum frequency to return (Hz). Omit for full range", default=None)] = None,
) -> str:
    """Compute the single-sided amplitude spectrum (FFT) of a current signal.

    Returns a spectrum_id referencing the stored spectrum, plus a compact
    summary with top peaks. Use spectrum_id in downstream fault-detection tools.
    """
    x, fs = _get_signal(signal_id, signal, sampling_freq_hz)
    freqs, amps = compute_fft_spectrum(x, fs, n_fft=n_fft, sided="one")

    if max_freq_hz is not None:
        mask = freqs <= max_freq_hz
        freqs = freqs[mask]
        amps = amps[mask]

    spec_id = _store_spectrum(freqs, amps, source=signal_id or "raw_input", kind="fft")

    return json.dumps({
        "spectrum_id": spec_id,
        **_spectrum_summary(freqs, amps),
        "note": (
            f"Spectrum stored as '{spec_id}'. "
            "Pass this spectrum_id to detect_broken_rotor_bars, "
            "detect_eccentricity, find_spectrum_peaks, etc."
        ),
    }, indent=2)


# ===================================================================
# TOOL 6: Compute PSD
# ===================================================================

@mcp.tool()
def compute_power_spectral_density(
    signal_id: Annotated[str | None, Field(description="ID of a stored signal. Preferred over raw array.", default=None)] = None,
    signal: Annotated[list[float] | None, Field(description="Time-domain signal array. Use signal_id instead.", default=None)] = None,
    sampling_freq_hz: Annotated[float | None, Field(description="Sampling frequency in Hz. Auto-resolved when using signal_id.", default=None)] = None,
    nperseg: Annotated[int | None, Field(description="FFT segment length. Omit for auto", default=None)] = None,
    max_freq_hz: Annotated[float | None, Field(description="Maximum frequency to return (Hz)", default=None)] = None,
) -> str:
    """Compute Power Spectral Density using Welch's method.

    Better for noisy signals and trend analysis than a raw FFT.
    Returns a spectrum_id plus compact summary.
    """
    x, fs = _get_signal(signal_id, signal, sampling_freq_hz)
    freqs, psd = compute_psd(x, fs, nperseg=nperseg)

    if max_freq_hz is not None:
        mask = freqs <= max_freq_hz
        freqs = freqs[mask]
        psd = psd[mask]

    spec_id = _store_spectrum(freqs, psd, source=signal_id or "raw_input", kind="psd")

    return json.dumps({
        "spectrum_id": spec_id,
        **_spectrum_summary(freqs, psd),
        "note": (
            f"PSD stored as '{spec_id}'. "
            "Pass this spectrum_id to compute_band_energy, find_spectrum_peaks, etc."
        ),
    }, indent=2)


# ===================================================================
# TOOL 7: Detect Spectral Peaks
# ===================================================================

@mcp.tool()
def find_spectrum_peaks(
    spectrum_id: Annotated[str | None, Field(description="ID of a stored spectrum (from compute_spectrum or compute_power_spectral_density). Preferred over raw arrays.", default=None)] = None,
    frequencies_hz: Annotated[list[float] | None, Field(description="Frequency axis from spectrum/PSD. Use spectrum_id instead.", default=None)] = None,
    amplitudes: Annotated[list[float] | None, Field(description="Amplitude or PSD values. Use spectrum_id instead.", default=None)] = None,
    min_height: Annotated[float | None, Field(description="Minimum peak height", default=None)] = None,
    min_prominence: Annotated[float | None, Field(description="Minimum peak prominence", default=None)] = None,
    min_distance_hz: Annotated[float | None, Field(description="Min distance between peaks in Hz", default=None)] = None,
    freq_low_hz: Annotated[float | None, Field(description="Lower frequency bound for search", default=None)] = None,
    freq_high_hz: Annotated[float | None, Field(description="Upper frequency bound for search", default=None)] = None,
    max_peaks: Annotated[int, Field(description="Maximum number of peaks to return", default=20)] = 20,
) -> str:
    """Detect peaks in a frequency spectrum.

    Returns a list of peaks sorted by amplitude (highest first) with
    frequency, amplitude, and prominence values.
    """
    freqs, amps = _get_spectrum(spectrum_id, frequencies_hz, amplitudes)

    freq_range = None
    if freq_low_hz is not None and freq_high_hz is not None:
        freq_range = (freq_low_hz, freq_high_hz)

    peaks = detect_peaks(
        freqs, amps,
        height=min_height,
        prominence=min_prominence,
        distance_hz=min_distance_hz,
        freq_range=freq_range,
        max_peaks=max_peaks,
    )

    return json.dumps({"peaks": peaks, "n_peaks_found": len(peaks)}, indent=2)


# ===================================================================
# TOOL 8: Detect Broken Rotor Bars
# ===================================================================

@mcp.tool()
def detect_broken_rotor_bars(
    spectrum_id: Annotated[str | None, Field(description="ID of a stored spectrum (from compute_spectrum). Preferred over raw arrays.", default=None)] = None,
    frequencies_hz: Annotated[list[float] | None, Field(description="Spectrum frequency axis (Hz). Use spectrum_id instead.", default=None)] = None,
    amplitudes: Annotated[list[float] | None, Field(description="Spectrum amplitude values. Use spectrum_id instead.", default=None)] = None,
    supply_freq_hz: Annotated[float, Field(description="Supply frequency in Hz")] = 50.0,
    poles: Annotated[int, Field(description="Number of poles")] = 4,
    rotor_speed_rpm: Annotated[float, Field(description="Rotor speed in RPM")] = 1470.0,
    tolerance_hz: Annotated[float, Field(description="Frequency search tolerance in Hz", default=0.5)] = 0.5,
) -> str:
    """Detect broken rotor bar faults from current spectrum.

    Computes the BRB fault index by measuring sidebands at (1 ± 2s)·f_s
    relative to the fundamental. Returns severity classification:
    healthy / incipient / moderate / severe.
    """
    params = calculate_motor_parameters(supply_freq_hz, poles, rotor_speed_rpm)
    freqs, amps = _get_spectrum(spectrum_id, frequencies_hz, amplitudes)

    result = brb_fault_index(freqs, amps, params, tolerance_hz)
    return json.dumps(result, indent=2, default=str)


# ===================================================================
# TOOL 9: Detect Eccentricity
# ===================================================================

@mcp.tool()
def detect_eccentricity(
    spectrum_id: Annotated[str | None, Field(description="ID of a stored spectrum. Preferred over raw arrays.", default=None)] = None,
    frequencies_hz: Annotated[list[float] | None, Field(description="Spectrum frequency axis (Hz). Use spectrum_id instead.", default=None)] = None,
    amplitudes: Annotated[list[float] | None, Field(description="Spectrum amplitude values. Use spectrum_id instead.", default=None)] = None,
    supply_freq_hz: Annotated[float, Field(description="Supply frequency in Hz")] = 50.0,
    poles: Annotated[int, Field(description="Number of poles")] = 4,
    rotor_speed_rpm: Annotated[float, Field(description="Rotor speed in RPM")] = 1470.0,
    harmonics: Annotated[int, Field(description="Number of harmonic orders to check", default=3)] = 3,
    tolerance_hz: Annotated[float, Field(description="Frequency search tolerance in Hz", default=0.5)] = 0.5,
) -> str:
    """Detect air-gap eccentricity faults from current spectrum.

    Analyses sidebands at f_s ± k·f_r for static and dynamic eccentricity.
    Returns severity classification.
    """
    params = calculate_motor_parameters(supply_freq_hz, poles, rotor_speed_rpm)
    freqs, amps = _get_spectrum(spectrum_id, frequencies_hz, amplitudes)

    result = eccentricity_fault_index(freqs, amps, params, harmonics, tolerance_hz)
    return json.dumps(result, indent=2, default=str)


# ===================================================================
# TOOL 10: Detect Stator Faults
# ===================================================================

@mcp.tool()
def detect_stator_faults(
    spectrum_id: Annotated[str | None, Field(description="ID of a stored spectrum. Preferred over raw arrays.", default=None)] = None,
    frequencies_hz: Annotated[list[float] | None, Field(description="Spectrum frequency axis (Hz). Use spectrum_id instead.", default=None)] = None,
    amplitudes: Annotated[list[float] | None, Field(description="Spectrum amplitude values. Use spectrum_id instead.", default=None)] = None,
    supply_freq_hz: Annotated[float, Field(description="Supply frequency in Hz")] = 50.0,
    poles: Annotated[int, Field(description="Number of poles")] = 4,
    rotor_speed_rpm: Annotated[float, Field(description="Rotor speed in RPM")] = 1470.0,
    harmonics: Annotated[int, Field(description="Number of harmonic orders", default=3)] = 3,
    tolerance_hz: Annotated[float, Field(description="Frequency search tolerance in Hz", default=0.5)] = 0.5,
) -> str:
    """Detect stator inter-turn short circuit faults from current spectrum.

    Analyses sidebands at f_s ± 2k·f_r caused by stator winding asymmetry.
    """
    params = calculate_motor_parameters(supply_freq_hz, poles, rotor_speed_rpm)
    freqs, amps = _get_spectrum(spectrum_id, frequencies_hz, amplitudes)

    result = stator_fault_index(freqs, amps, params, harmonics, tolerance_hz)
    return json.dumps(result, indent=2, default=str)


# ===================================================================
# TOOL 11: Detect Bearing Faults
# ===================================================================

@mcp.tool()
def detect_bearing_faults(
    spectrum_id: Annotated[str | None, Field(description="ID of a stored spectrum. Preferred over raw arrays.", default=None)] = None,
    frequencies_hz: Annotated[list[float] | None, Field(description="Spectrum frequency axis (Hz). Use spectrum_id instead.", default=None)] = None,
    amplitudes: Annotated[list[float] | None, Field(description="Spectrum amplitude values. Use spectrum_id instead.", default=None)] = None,
    supply_freq_hz: Annotated[float, Field(description="Supply frequency in Hz")] = 50.0,
    bearing_defect_freq_hz: Annotated[float, Field(description="Bearing characteristic defect frequency in Hz (BPFO, BPFI, BSF, or FTF)")] = 0.0,
    defect_type: Annotated[str, Field(description="Defect type label: bpfo, bpfi, bsf, or ftf", default="bpfo")] = "bpfo",
    harmonics: Annotated[int, Field(description="Number of sideband orders", default=2)] = 2,
    tolerance_hz: Annotated[float, Field(description="Frequency search tolerance in Hz", default=0.5)] = 0.5,
) -> str:
    """Detect bearing defect signatures in the stator-current spectrum.

    Bearing faults modulate motor torque, creating sidebands at
    f_s ± k·f_defect. Note: bearing signatures in current are typically
    weak; envelope analysis or vibration data can improve detection.
    """
    freqs, amps = _get_spectrum(spectrum_id, frequencies_hz, amplitudes)

    result = bearing_fault_index(
        freqs, amps, supply_freq_hz,
        bearing_defect_freq_hz, defect_type, harmonics, tolerance_hz,
    )
    return json.dumps(result, indent=2, default=str)


# ===================================================================
# TOOL 12: Compute Envelope Spectrum
# ===================================================================

@mcp.tool()
def compute_envelope_spectrum(
    signal_id: Annotated[str | None, Field(description="ID of a stored signal. Preferred over raw array.", default=None)] = None,
    signal: Annotated[list[float] | None, Field(description="Time-domain current signal. Use signal_id instead.", default=None)] = None,
    sampling_freq_hz: Annotated[float | None, Field(description="Sampling frequency in Hz. Auto-resolved when using signal_id.", default=None)] = None,
    bandpass_low_hz: Annotated[float | None, Field(description="Lower bandpass cutoff before envelope (optional)", default=None)] = None,
    bandpass_high_hz: Annotated[float | None, Field(description="Upper bandpass cutoff before envelope (optional)", default=None)] = None,
    max_freq_hz: Annotated[float | None, Field(description="Max frequency to return (Hz)", default=None)] = None,
) -> str:
    """Compute the envelope spectrum of a current signal.

    Uses the Hilbert transform to extract the amplitude envelope, then
    computes its FFT. Useful for detecting bearing and mechanical faults
    that modulate the current at low frequencies.
    """
    x, fs = _get_signal(signal_id, signal, sampling_freq_hz)

    bp = None
    if bandpass_low_hz is not None and bandpass_high_hz is not None:
        bp = (bandpass_low_hz, bandpass_high_hz)

    freqs, amps = envelope_spectrum(x, fs, bandpass=bp)

    if max_freq_hz is not None:
        mask = freqs <= max_freq_hz
        freqs = freqs[mask]
        amps = amps[mask]

    # Also compute statistical indices of the envelope
    env = hilbert_envelope(x)
    env = env - np.mean(env)
    stats = envelope_statistical_indices(env)

    return json.dumps({
        "frequencies_hz": freqs.tolist(),
        "amplitudes": amps.tolist(),
        "n_bins": len(freqs),
        "envelope_statistics": stats,
    })


# ===================================================================
# TOOL 13: Compute Band Energy
# ===================================================================

@mcp.tool()
def compute_band_energy(
    centre_freq_hz: Annotated[float, Field(description="Centre of the frequency band (Hz)")],
    spectrum_id: Annotated[str | None, Field(description="ID of a stored PSD spectrum. Preferred over raw arrays.", default=None)] = None,
    frequencies_hz: Annotated[list[float] | None, Field(description="PSD frequency axis (Hz). Use spectrum_id instead.", default=None)] = None,
    psd_values: Annotated[list[float] | None, Field(description="PSD values. Use spectrum_id instead.", default=None)] = None,
    bandwidth_hz: Annotated[float, Field(description="Total bandwidth for energy integration (Hz)", default=5.0)] = 5.0,
) -> str:
    """Compute the integrated spectral energy in a frequency band.

    Useful as a generic fault/cavitation indicator — measures the energy
    concentration around a characteristic frequency in the PSD.
    """
    freqs, psd = _get_spectrum(spectrum_id, frequencies_hz, psd_values)

    result = band_energy_index(freqs, psd, centre_freq_hz, bandwidth_hz)
    return json.dumps(result, indent=2)


# ===================================================================
# TOOL 14: Compute STFT
# ===================================================================

@mcp.tool()
def compute_time_frequency(
    signal_id: Annotated[str | None, Field(description="ID of a stored signal. Preferred over raw array.", default=None)] = None,
    signal: Annotated[list[float] | None, Field(description="Time-domain current signal. Use signal_id instead.", default=None)] = None,
    sampling_freq_hz: Annotated[float | None, Field(description="Sampling frequency in Hz. Auto-resolved when using signal_id.", default=None)] = None,
    nperseg: Annotated[int, Field(description="Window length per segment (samples)", default=256)] = 256,
    target_freq_hz: Annotated[float | None, Field(description="Frequency to track over time (Hz). If provided, returns amplitude vs time for that frequency", default=None)] = None,
    tolerance_hz: Annotated[float, Field(description="Tolerance for frequency tracking (Hz)", default=2.0)] = 2.0,
) -> str:
    """Compute Short-Time Fourier Transform (STFT) for time-frequency analysis.

    For non-stationary conditions (variable speed/load, start-up transients).
    If target_freq_hz is provided, also tracks that frequency's amplitude over time.
    Returns a summary (not the full 2D matrix) to keep output manageable.
    """
    x, fs = _get_signal(signal_id, signal, sampling_freq_hz)
    stft_result = compute_stft(x, fs, nperseg=nperseg)

    output: dict = {
        "n_freq_bins": stft_result["n_freq_bins"],
        "n_time_bins": stft_result["n_time_bins"],
        "freq_range_hz": [
            float(stft_result["frequencies_hz"][0]),
            float(stft_result["frequencies_hz"][-1]),
        ],
        "time_range_s": [
            float(stft_result["times_s"][0]),
            float(stft_result["times_s"][-1]),
        ],
        "freq_resolution_hz": round(
            float(stft_result["frequencies_hz"][1] - stft_result["frequencies_hz"][0]), 6
        ) if stft_result["n_freq_bins"] > 1 else 0,
    }

    # Average spectrum over time
    avg_spectrum = np.mean(stft_result["magnitude"], axis=1)
    output["average_spectrum"] = {
        "frequencies_hz": stft_result["frequencies_hz"].tolist(),
        "amplitudes": avg_spectrum.tolist(),
    }

    if target_freq_hz is not None:
        tracking = track_frequency_over_time(stft_result, target_freq_hz, tolerance_hz)
        output["frequency_tracking"] = tracking

    return json.dumps(output, indent=2, default=str)


# ===================================================================
# TOOL 15: Inspect Signal File
# ===================================================================

@mcp.tool()
def inspect_signal_file(
    file_path: Annotated[str, Field(description="Absolute path to the signal file (CSV, WAV, or NPY)")],
) -> str:
    """Inspect a signal file without fully loading it.

    Returns file metadata: size, format details, estimated number of
    samples, sampling frequency (for WAV), column headers (for CSV),
    and array shape (for NPY).  Use this before load_signal_from_file
    to verify the file format and plan the loading parameters.
    """
    info = get_signal_file_info(file_path)
    return json.dumps(info, indent=2)


# ===================================================================
# TOOL 16: Load Signal from File
# ===================================================================

@mcp.tool()
def load_signal_from_file(
    file_path: Annotated[str, Field(description="Absolute path to the signal file (CSV, WAV, or NPY)")],
    sampling_freq_hz: Annotated[float | None, Field(description="Sampling frequency in Hz. Required for NPY; optional for CSV if a time column exists; auto-detected for WAV", default=None)] = None,
    signal_column: Annotated[int | str, Field(description="CSV column containing the current signal (0-based index or header name)", default=1)] = 1,
    time_column: Annotated[int | str | None, Field(description="CSV column for time (index or name). Set to null if no time column", default=0)] = 0,
    delimiter: Annotated[str | None, Field(description="CSV delimiter. Auto-detected if null (comma for .csv, tab for .tsv/.txt)", default=None)] = None,
    channel: Annotated[int, Field(description="WAV channel index (0-based) for multi-channel files", default=0)] = 0,
    skip_header: Annotated[int, Field(description="Number of CSV header rows to skip", default=1)] = 1,
    max_rows: Annotated[int | None, Field(description="Max data rows to read from CSV (null = all)", default=None)] = None,
) -> str:
    """Load a motor-current signal from a file (CSV, WAV, or NumPy NPY).

    Supports the most common formats used by industrial DAQ systems:
    - **CSV/TSV/TXT**: Columnar data with optional time column.
      The sampling frequency is inferred from the time column or
      must be provided explicitly.
    - **WAV**: Audio files from portable recorders or DAQ. Sampling
      frequency is read from the WAV header.
    - **NPY**: NumPy binary arrays. Sampling frequency must be provided.

    Returns the signal, sampling frequency, number of samples, duration,
    and file metadata.  The returned signal can then be passed to
    preprocess_signal, compute_spectrum, or run_full_diagnosis.
    """
    result = load_signal(
        file_path,
        sampling_freq_hz=sampling_freq_hz,
        signal_column=signal_column,
        time_column=time_column,
        delimiter=delimiter,
        channel=channel,
        skip_header=skip_header,
        max_rows=max_rows,
    )

    # Store server-side — return only ID + summary
    sig_id = _store_signal(
        result["signal"], result["sampling_freq_hz"],
        file_path=result["file_path"],
        file_format=result["format"],
    )
    arr = np.array(result["signal"], dtype=np.float64)

    return json.dumps({
        "signal_id": sig_id,
        **_signal_summary(arr, result["sampling_freq_hz"]),
        "file_path": result["file_path"],
        "format": result["format"],
        "metadata": result.get("metadata"),
        "note": (
            f"Signal stored server-side as '{sig_id}'. "
            "Pass this signal_id to preprocess_signal, compute_spectrum, "
            "run_full_diagnosis, or other analysis tools."
        ),
    }, indent=2)


# ===================================================================
# TOOL 17: Generate Test Signal
# ===================================================================

@mcp.tool()
def generate_test_current_signal(
    duration_s: Annotated[float, Field(description="Signal duration in seconds", default=10.0)] = 10.0,
    sampling_freq_hz: Annotated[float, Field(description="Sampling frequency in Hz", default=5000.0)] = 5000.0,
    supply_freq_hz: Annotated[float, Field(description="Supply frequency in Hz", default=50.0)] = 50.0,
    poles: Annotated[int, Field(description="Number of poles", default=4)] = 4,
    rotor_speed_rpm: Annotated[float, Field(description="Rotor speed in RPM", default=1470.0)] = 1470.0,
    noise_level: Annotated[float, Field(description="Noise standard deviation (0-1 relative)", default=0.01)] = 0.01,
    faults: Annotated[list[str] | None, Field(description="Faults to inject: 'brb', 'eccentricity', 'bearing'. Omit for healthy signal", default=None)] = None,
    fault_severity: Annotated[float, Field(description="Fault component amplitude (0-1 relative)", default=0.02)] = 0.02,
) -> str:
    """Generate a synthetic motor-current test signal.

    Creates a simulated stator-current waveform with the fundamental,
    supply harmonics, noise, and optional fault signatures. Useful for
    testing, validation, and demonstration of MCSA analysis tools.
    """
    result = generate_test_signal(
        duration_s=duration_s,
        fs_sample=sampling_freq_hz,
        supply_freq_hz=supply_freq_hz,
        poles=poles,
        rotor_speed_rpm=rotor_speed_rpm,
        noise_std=noise_level,
        faults=faults,
        fault_severity=fault_severity,
    )

    # Store server-side — return only ID + summary
    sig_id = _store_signal(
        result["signal"], result["sampling_freq_hz"],
        motor_params=result["motor_params"],
        faults_injected=result["faults_injected"],
    )
    arr = np.array(result["signal"], dtype=np.float64)

    return json.dumps({
        "signal_id": sig_id,
        **_signal_summary(arr, result["sampling_freq_hz"]),
        "motor_params": result["motor_params"],
        "faults_injected": result["faults_injected"],
        "fault_severity": result["fault_severity"],
        "note": (
            f"Signal stored server-side as '{sig_id}'. "
            "Pass this signal_id to preprocess_signal, compute_spectrum, "
            "run_full_diagnosis, or other analysis tools."
        ),
    }, indent=2)


# ===================================================================
# TOOL 18: Full Diagnostic Report
# ===================================================================

@mcp.tool()
def run_full_diagnosis(
    signal_id: Annotated[str | None, Field(description="ID of a stored signal (from generate_test_current_signal or load_signal_from_file). Preferred over raw array.", default=None)] = None,
    signal: Annotated[list[float] | None, Field(description="Raw time-domain current signal. Use signal_id instead for large signals.", default=None)] = None,
    sampling_freq_hz: Annotated[float | None, Field(description="Sampling frequency in Hz. Auto-resolved when using signal_id.", default=None)] = None,
    supply_freq_hz: Annotated[float, Field(description="Supply frequency in Hz")] = 50.0,
    poles: Annotated[int, Field(description="Number of poles")] = 4,
    rotor_speed_rpm: Annotated[float, Field(description="Rotor speed in RPM")] = 1470.0,
    bearing_defect_freq_hz: Annotated[float | None, Field(description="Bearing defect frequency in Hz (optional, for bearing analysis)", default=None)] = None,
    tolerance_hz: Annotated[float, Field(description="Frequency search tolerance in Hz", default=0.5)] = 0.5,
) -> str:
    """Run a comprehensive MCSA diagnostic analysis on a current signal.

    Performs the full pipeline: preprocessing → spectrum → fault detection
    for broken rotor bars, eccentricity, stator faults, and optionally
    bearing defects. Returns a complete diagnostic report.
    """
    x, fs = _get_signal(signal_id, signal, sampling_freq_hz)

    # Motor parameters
    params = calculate_motor_parameters(supply_freq_hz, poles, rotor_speed_rpm)

    # Preprocess
    x_proc = preprocess_pipeline(x, fs, window="hann")

    # Spectrum
    freqs, amps = compute_fft_spectrum(x_proc, fs, sided="one")

    # PSD for band energy
    freqs_psd, psd_vals = compute_psd(x, fs)

    # Fault detection
    brb = brb_fault_index(freqs, amps, params, tolerance_hz)
    ecc = eccentricity_fault_index(freqs, amps, params, tolerance_hz=tolerance_hz)
    stator = stator_fault_index(freqs, amps, params, tolerance_hz=tolerance_hz)

    # Band energy around fundamental
    be = band_energy_index(freqs_psd, psd_vals, supply_freq_hz, bandwidth_hz=10.0)

    # Envelope statistics
    env = hilbert_envelope(x)
    env_dc_removed = env - np.mean(env)
    env_stats = envelope_statistical_indices(env_dc_removed)

    # Bearing (if geometry provided)
    bearing_result = None
    if bearing_defect_freq_hz is not None:
        bearing_result = bearing_fault_index(
            freqs, amps, supply_freq_hz,
            bearing_defect_freq_hz, "bearing", tolerance_hz=tolerance_hz,
        )

    # Top peaks
    peaks = detect_peaks(freqs, amps, prominence=0.001, max_peaks=10)

    # Assemble report
    report = {
        "motor_parameters": params.to_dict(),
        "signal_info": {
            "n_samples": len(x),
            "sampling_freq_hz": fs,
            "duration_s": round(len(x) / fs, 3),
            "freq_resolution_hz": round(float(freqs[1] - freqs[0]), 6) if len(freqs) > 1 else 0,
        },
        "top_spectral_peaks": peaks,
        "fault_analysis": {
            "broken_rotor_bars": brb,
            "eccentricity": ecc,
            "stator_inter_turn": stator,
            "bearing": bearing_result,
        },
        "band_energy_around_fundamental": be,
        "envelope_statistics": env_stats,
        "summary": {
            "brb_severity": brb["severity"],
            "eccentricity_severity": ecc["severity"],
            "stator_severity": stator["severity"],
            "envelope_kurtosis": env_stats["kurtosis"],
            "overall_assessment": _overall_assessment(brb, ecc, stator, env_stats),
        },
    }

    return json.dumps(report, indent=2, default=str)


def _overall_assessment(brb: dict, ecc: dict, stator: dict, env_stats: dict) -> str:
    """Generate a brief overall assessment string."""
    severities = [brb["severity"], ecc["severity"], stator["severity"]]

    if "severe" in severities:
        return "CRITICAL — One or more fault indicators at severe level. Immediate inspection recommended."
    if "moderate" in severities:
        return "WARNING — Moderate fault indication detected. Schedule inspection."
    if "incipient" in severities:
        return "WATCH — Incipient fault signatures detected. Increase monitoring frequency."
    if env_stats["kurtosis"] > 6.0:
        return "WATCH — Elevated envelope kurtosis may indicate mechanical impulsiveness."
    return "NORMAL — No significant fault indicators detected."


# ===================================================================
# TOOL 19: Diagnose from File (one-shot)
# ===================================================================

@mcp.tool()
def diagnose_from_file(
    file_path: Annotated[str, Field(description="Absolute path to the signal file (CSV, WAV, or NPY)")],
    supply_freq_hz: Annotated[float, Field(description="Supply frequency in Hz")],
    poles: Annotated[int, Field(description="Number of poles")],
    rotor_speed_rpm: Annotated[float, Field(description="Rotor speed in RPM")],
    sampling_freq_hz: Annotated[float | None, Field(description="Sampling frequency in Hz (required for NPY, optional for CSV with time column, auto-detected for WAV)", default=None)] = None,
    signal_column: Annotated[int | str, Field(description="CSV column for the current signal", default=1)] = 1,
    time_column: Annotated[int | str | None, Field(description="CSV column for time (null if absent)", default=0)] = 0,
    channel: Annotated[int, Field(description="WAV channel index", default=0)] = 0,
    bearing_defect_freq_hz: Annotated[float | None, Field(description="Bearing defect frequency in Hz (optional)", default=None)] = None,
    tolerance_hz: Annotated[float, Field(description="Frequency search tolerance in Hz", default=0.5)] = 0.5,
) -> str:
    """Load a signal from file and run the full MCSA diagnostic pipeline.

    One-shot tool: reads the signal file, preprocesses, computes the
    spectrum, runs all fault detectors, and returns a complete diagnostic
    report.  Ideal for batch or automated condition-monitoring workflows.
    """
    # Load signal
    loaded = load_signal(
        file_path,
        sampling_freq_hz=sampling_freq_hz,
        signal_column=signal_column,
        time_column=time_column,
        channel=channel,
    )

    x = np.array(loaded["signal"], dtype=np.float64)
    fs_sample = loaded["sampling_freq_hz"]

    # Store signal for potential follow-up analysis
    sig_id = _store_signal(x, fs_sample, source_file=loaded["file_path"])

    # Motor parameters
    params = calculate_motor_parameters(supply_freq_hz, poles, rotor_speed_rpm)

    # Preprocess
    x_proc = preprocess_pipeline(x, fs_sample, window="hann")

    # Spectrum
    freqs, amps = compute_fft_spectrum(x_proc, fs_sample, sided="one")

    # PSD
    freqs_psd, psd_vals = compute_psd(x, fs_sample)

    # Fault detection
    brb = brb_fault_index(freqs, amps, params, tolerance_hz)
    ecc = eccentricity_fault_index(freqs, amps, params, tolerance_hz=tolerance_hz)
    stator = stator_fault_index(freqs, amps, params, tolerance_hz=tolerance_hz)

    # Band energy
    be = band_energy_index(freqs_psd, psd_vals, supply_freq_hz, bandwidth_hz=10.0)

    # Envelope
    env = hilbert_envelope(x)
    env_dc = env - np.mean(env)
    env_stats = envelope_statistical_indices(env_dc)

    # Bearing
    bearing_result = None
    if bearing_defect_freq_hz is not None:
        bearing_result = bearing_fault_index(
            freqs, amps, supply_freq_hz,
            bearing_defect_freq_hz, "bearing", tolerance_hz=tolerance_hz,
        )

    # Peaks
    peaks = detect_peaks(freqs, amps, prominence=0.001, max_peaks=10)

    report = {
        "signal_id": sig_id,
        "source_file": loaded["file_path"],
        "file_format": loaded["format"],
        "motor_parameters": params.to_dict(),
        "signal_info": {
            "n_samples": loaded["n_samples"],
            "sampling_freq_hz": fs_sample,
            "duration_s": loaded["duration_s"],
            "freq_resolution_hz": round(float(freqs[1] - freqs[0]), 6) if len(freqs) > 1 else 0,
        },
        "top_spectral_peaks": peaks,
        "fault_analysis": {
            "broken_rotor_bars": brb,
            "eccentricity": ecc,
            "stator_inter_turn": stator,
            "bearing": bearing_result,
        },
        "band_energy_around_fundamental": be,
        "envelope_statistics": env_stats,
        "summary": {
            "brb_severity": brb["severity"],
            "eccentricity_severity": ecc["severity"],
            "stator_severity": stator["severity"],
            "envelope_kurtosis": env_stats["kurtosis"],
            "overall_assessment": _overall_assessment(brb, ecc, stator, env_stats),
        },
    }

    return json.dumps(report, indent=2, default=str)


# ===================================================================
# UTILITY TOOLS: Data store management
# ===================================================================

@mcp.tool()
def list_stored_data() -> str:
    """List all signals and spectra currently stored on disk.

    Returns a compact summary of each stored item (ID, type, size,
    and key metadata) without returning the raw data arrays.
    The data is persisted in the MCSA data directory and survives
    server restarts.
    """
    all_ids = _list_all_stored_ids()
    if not all_ids:
        return json.dumps({
            "stored_items": [],
            "data_directory": str(_DATA_DIR),
            "note": "No data stored yet. Use generate_test_current_signal or load_signal_from_file to create signals.",
        })

    items = []
    for sid in all_ids:
        try:
            entry = _resolve_entry(sid)
        except Exception:
            items.append({"id": sid, "type": "unknown", "error": "could not load"})
            continue

        dtype = entry.get("_type", "unknown")
        info: dict = {"id": sid, "type": dtype}
        if dtype == "signal":
            arr = entry.get("signal")
            fs = entry.get("sampling_freq_hz", 0)
            info["n_samples"] = entry.get("n_samples", len(arr) if arr is not None else 0)
            info["sampling_freq_hz"] = fs
            if arr is not None and fs > 0:
                info["duration_s"] = round(len(arr) / fs, 3)
        elif dtype == "spectrum":
            info["n_bins"] = entry.get("n_bins", 0)
            freqs = entry.get("frequencies_hz")
            if freqs is not None and len(freqs) > 0:
                info["freq_range_hz"] = [round(float(freqs[0]), 2), round(float(freqs[-1]), 2)]
        items.append(info)

    return json.dumps({
        "stored_items": items,
        "total": len(items),
        "data_directory": str(_DATA_DIR),
    }, indent=2, default=str)


@mcp.tool()
def clear_stored_data(
    data_id: Annotated[str | None, Field(description="ID of a specific item to remove, or omit to clear everything", default=None)] = None,
) -> str:
    """Delete stored signals and spectra from disk and memory.

    Pass a specific data_id to remove one item, or omit to clear all.
    """
    if data_id is not None:
        # Delete one item
        removed = False
        path = _signal_path(data_id) if data_id.startswith("sig_") else _spectrum_path(data_id)
        if path.exists():
            path.unlink()
            removed = True
        _cache.pop(data_id, None)
        if removed:
            return json.dumps({"cleared": data_id, "remaining": len(_list_all_stored_ids())})
        return json.dumps({"error": f"ID '{data_id}' not found in store."})

    # Delete everything
    count = 0
    for f in _SIGNALS_DIR.glob("sig_*.npz"):
        f.unlink()
        count += 1
    for f in _SPECTRA_DIR.glob("spec_*.npz"):
        f.unlink()
        count += 1
    _cache.clear()
    return json.dumps({"cleared": "all", "items_removed": count})


# ===================================================================
# PROMPT: Guided analysis
# ===================================================================

@mcp.prompt()
def analyze_motor_current(
    motor_type: str = "induction",
    supply_freq_hz: str = "50",
    poles: str = "4",
    rotor_speed_rpm: str = "1470",
) -> str:
    """Step-by-step guided prompt for MCSA analysis of a motor current signal."""
    return f"""You are performing Motor Current Signature Analysis (MCSA) on a {motor_type} motor.

Motor parameters:
- Supply frequency: {supply_freq_hz} Hz
- Poles: {poles}
- Rotor speed: {rotor_speed_rpm} RPM

**IMPORTANT — Server-side data store:**
Signals and spectra are stored on the server and referenced by short IDs
(e.g. sig_0001, spec_0001). NEVER pass raw sample arrays in conversation.
Use `list_stored_data` to see what is available; `clear_stored_data` to free memory.

Follow this diagnostic workflow:

1. **Load the current signal**:
   - From a file: use `inspect_signal_file` to check the format, then `load_signal_from_file`
   - Supported formats: CSV, TSV, WAV, NumPy NPY
   - Or generate a synthetic signal with `generate_test_current_signal`
   - Both return a **signal_id** (e.g. `sig_0001`) and a compact summary.

2. **Calculate motor parameters** with `calculate_motor_params` to get slip, synchronous speed, and rotor frequency.

3. **Compute expected fault frequencies** with `compute_fault_frequencies` to know where to look in the spectrum.

4. **Preprocess** the signal with `preprocess_signal(signal_id=...)` — returns a **new signal_id** for the cleaned signal.

5. **Compute the spectrum** with `compute_spectrum(signal_id=...)` or `compute_power_spectral_density(signal_id=...)` — returns a **spectrum_id** and a summary with top peaks.

6. **Detect faults** (pass the spectrum_id):
   - `detect_broken_rotor_bars(spectrum_id=...)` — checks (1 ± 2s)·f_s sidebands
   - `detect_eccentricity(spectrum_id=...)` — checks f_s ± k·f_r sidebands
   - `detect_stator_faults(spectrum_id=...)` — checks f_s ± 2k·f_r sidebands
   - `detect_bearing_faults(spectrum_id=...)` — checks f_s ± k·f_defect (needs bearing geometry)

7. **Envelope analysis** with `compute_envelope_spectrum(signal_id=...)` for mechanical/bearing signatures.

8. Or use **one-shot shortcuts**:
   - `run_full_diagnosis(signal_id=...)` — full pipeline from a stored signal
   - `diagnose_from_file` — full pipeline directly from a file path

Report findings with severity levels and actionable recommendations.
"""


# ---------------------------------------------------------------------------
# Server entry
# ---------------------------------------------------------------------------

def serve(transport: Literal["stdio", "sse", "streamable-http"] = "stdio") -> None:
    """Start the MCSA MCP server."""
    mcp.run(transport=transport)
