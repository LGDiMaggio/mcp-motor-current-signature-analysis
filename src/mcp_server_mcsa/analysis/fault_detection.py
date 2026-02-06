"""Fault detection and severity assessment for MCSA.

Computes standardised fault indices from current spectra and provides
severity classification based on configurable thresholds.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mcp_server_mcsa.analysis.motor import MotorParameters
from mcp_server_mcsa.analysis.spectral import amplitude_at_frequency


# ---------------------------------------------------------------------------
# Severity thresholds (dB below fundamental)
# ---------------------------------------------------------------------------
# These are widely‑used empirical guidelines for induction motors.
# They should be adapted to the specific motor/application.
BRB_THRESHOLDS = {
    "healthy": -50.0,       # dB — sideband ≤ -50 dB relative to fundamental
    "incipient": -45.0,     # dB — early-stage fault
    "moderate": -40.0,      # dB — developing fault
    "severe": -35.0,        # dB — immediate action recommended
}

ECCENTRICITY_THRESHOLDS = {
    "healthy": -50.0,
    "incipient": -44.0,
    "moderate": -38.0,
    "severe": -30.0,
}


def _db_ratio(a: float, ref: float) -> float:
    """Compute 20·log10(a / ref), safe for zero values."""
    if ref <= 0 or a <= 0:
        return -np.inf
    return 20.0 * np.log10(a / ref)


def _classify_severity(db_value: float, thresholds: dict[str, float]) -> str:
    """Classify severity from dB value and ordered thresholds."""
    if db_value <= thresholds["healthy"]:
        return "healthy"
    elif db_value <= thresholds["incipient"]:
        return "incipient"
    elif db_value <= thresholds["moderate"]:
        return "moderate"
    else:
        return "severe"


# ---------------------------------------------------------------------------
# Broken Rotor Bars
# ---------------------------------------------------------------------------

def brb_fault_index(
    freqs: NDArray[np.floating],
    amps: NDArray[np.floating],
    params: MotorParameters,
    tolerance_hz: float = 0.5,
) -> dict:
    """Compute the Broken Rotor Bar (BRB) fault index.

    The index is the ratio of the lower and upper sideband amplitudes
    at (1 ± 2s)·f_s to the fundamental amplitude, expressed in dB.

    Args:
        freqs: Frequency axis of the spectrum.
        amps: Amplitude values of the spectrum.
        params: Motor parameters (for slip and supply frequency).
        tolerance_hz: Frequency search tolerance.

    Returns:
        Dictionary with frequencies found, amplitudes, dB indices,
        and severity classification.
    """
    fs = params.supply_freq_hz
    s = params.slip

    f_lower = (1 - 2 * s) * fs
    f_upper = (1 + 2 * s) * fs

    fundamental = amplitude_at_frequency(freqs, amps, fs, tolerance_hz)
    lower_sb = amplitude_at_frequency(freqs, amps, f_lower, tolerance_hz)
    upper_sb = amplitude_at_frequency(freqs, amps, f_upper, tolerance_hz)

    a_fund = fundamental["amplitude"]
    a_lower = lower_sb["amplitude"]
    a_upper = upper_sb["amplitude"]

    db_lower = _db_ratio(a_lower, a_fund)
    db_upper = _db_ratio(a_upper, a_fund)
    db_combined = _db_ratio((a_lower + a_upper) / 2.0, a_fund) if a_fund > 0 else -np.inf

    severity = _classify_severity(max(db_lower, db_upper), BRB_THRESHOLDS)

    return {
        "fault_type": "broken_rotor_bars",
        "fundamental": {
            "expected_hz": fs,
            **fundamental,
        },
        "lower_sideband": {
            "expected_hz": round(f_lower, 4),
            **lower_sb,
            "db_relative": round(float(db_lower), 2),
        },
        "upper_sideband": {
            "expected_hz": round(f_upper, 4),
            **upper_sb,
            "db_relative": round(float(db_upper), 2),
        },
        "combined_index_db": round(float(db_combined), 2),
        "severity": severity,
        "thresholds_db": BRB_THRESHOLDS,
    }


# ---------------------------------------------------------------------------
# Eccentricity
# ---------------------------------------------------------------------------

def eccentricity_fault_index(
    freqs: NDArray[np.floating],
    amps: NDArray[np.floating],
    params: MotorParameters,
    harmonics: int = 3,
    tolerance_hz: float = 0.5,
) -> dict:
    """Compute eccentricity fault indices.

    Searches for sidebands at f_s ± k·f_r (k = 1 … harmonics).

    Args:
        freqs: Frequency axis.
        amps: Amplitude values.
        params: Motor parameters.
        harmonics: Number of harmonic orders.
        tolerance_hz: Frequency tolerance.

    Returns:
        Dictionary with sideband amplitudes, dB indices, severity.
    """
    fs = params.supply_freq_hz
    fr = params.rotor_freq_hz
    fund = amplitude_at_frequency(freqs, amps, fs, tolerance_hz)
    a_fund = fund["amplitude"]

    sidebands = []
    worst_db = -np.inf

    for k in range(1, harmonics + 1):
        f_lo = fs - k * fr
        f_hi = fs + k * fr
        sb_lo = amplitude_at_frequency(freqs, amps, f_lo, tolerance_hz)
        sb_hi = amplitude_at_frequency(freqs, amps, f_hi, tolerance_hz)

        db_lo = _db_ratio(sb_lo["amplitude"], a_fund)
        db_hi = _db_ratio(sb_hi["amplitude"], a_fund)

        worst_db = max(worst_db, db_lo, db_hi)

        sidebands.append({
            "harmonic_order": k,
            "lower": {
                "expected_hz": round(f_lo, 4),
                **sb_lo,
                "db_relative": round(float(db_lo), 2),
            },
            "upper": {
                "expected_hz": round(f_hi, 4),
                **sb_hi,
                "db_relative": round(float(db_hi), 2),
            },
        })

    severity = _classify_severity(float(worst_db), ECCENTRICITY_THRESHOLDS)

    return {
        "fault_type": "eccentricity",
        "fundamental": {
            "expected_hz": fs,
            **fund,
        },
        "sidebands": sidebands,
        "worst_sideband_db": round(float(worst_db), 2),
        "severity": severity,
        "thresholds_db": ECCENTRICITY_THRESHOLDS,
    }


# ---------------------------------------------------------------------------
# Stator inter-turn short circuit
# ---------------------------------------------------------------------------

def stator_fault_index(
    freqs: NDArray[np.floating],
    amps: NDArray[np.floating],
    params: MotorParameters,
    harmonics: int = 3,
    tolerance_hz: float = 0.5,
) -> dict:
    """Compute stator inter‑turn fault indices.

    Looks for sidebands at f_s ± 2k·f_r.

    Args:
        freqs: Frequency axis.
        amps: Amplitude values.
        params: Motor parameters.
        harmonics: Number of harmonic orders.
        tolerance_hz: Frequency tolerance.

    Returns:
        Dictionary with sideband analysis and severity.
    """
    fs = params.supply_freq_hz
    fr = params.rotor_freq_hz
    fund = amplitude_at_frequency(freqs, amps, fs, tolerance_hz)
    a_fund = fund["amplitude"]

    sidebands = []
    worst_db = -np.inf

    for k in range(1, harmonics + 1):
        f_lo = fs - 2 * k * fr
        f_hi = fs + 2 * k * fr
        sb_lo = amplitude_at_frequency(freqs, amps, f_lo, tolerance_hz)
        sb_hi = amplitude_at_frequency(freqs, amps, f_hi, tolerance_hz)

        db_lo = _db_ratio(sb_lo["amplitude"], a_fund)
        db_hi = _db_ratio(sb_hi["amplitude"], a_fund)
        worst_db = max(worst_db, db_lo, db_hi)

        sidebands.append({
            "harmonic_order": k,
            "lower": {
                "expected_hz": round(f_lo, 4),
                **sb_lo,
                "db_relative": round(float(db_lo), 2),
            },
            "upper": {
                "expected_hz": round(f_hi, 4),
                **sb_hi,
                "db_relative": round(float(db_hi), 2),
            },
        })

    severity = _classify_severity(float(worst_db), ECCENTRICITY_THRESHOLDS)

    return {
        "fault_type": "stator_inter_turn",
        "fundamental": {
            "expected_hz": fs,
            **fund,
        },
        "sidebands": sidebands,
        "worst_sideband_db": round(float(worst_db), 2),
        "severity": severity,
        "thresholds_db": ECCENTRICITY_THRESHOLDS,
    }


# ---------------------------------------------------------------------------
# Bearing faults (via stator current)
# ---------------------------------------------------------------------------

def bearing_fault_index(
    freqs: NDArray[np.floating],
    amps: NDArray[np.floating],
    supply_freq_hz: float,
    bearing_defect_freq_hz: float,
    defect_type: str = "bpfo",
    harmonics: int = 2,
    tolerance_hz: float = 0.5,
) -> dict:
    """Compute bearing fault indices from stator‑current spectrum.

    Bearing defects produce torque oscillations that modulate the current,
    creating sidebands at f_s ± k · f_defect.

    Args:
        freqs: Frequency axis.
        amps: Amplitude values.
        supply_freq_hz: Supply frequency in Hz.
        bearing_defect_freq_hz: Characteristic defect frequency in Hz
            (BPFO, BPFI, BSF, or FTF).
        defect_type: Label for the defect type.
        harmonics: Number of sideband orders.
        tolerance_hz: Frequency tolerance.

    Returns:
        Dictionary with sideband analysis.
    """
    fs = supply_freq_hz
    fd = bearing_defect_freq_hz
    fund = amplitude_at_frequency(freqs, amps, fs, tolerance_hz)
    a_fund = fund["amplitude"]

    sidebands = []
    worst_db = -np.inf

    for k in range(1, harmonics + 1):
        f_lo = fs - k * fd
        f_hi = fs + k * fd
        sb_lo = amplitude_at_frequency(freqs, amps, f_lo, tolerance_hz)
        sb_hi = amplitude_at_frequency(freqs, amps, f_hi, tolerance_hz)

        db_lo = _db_ratio(sb_lo["amplitude"], a_fund)
        db_hi = _db_ratio(sb_hi["amplitude"], a_fund)
        worst_db = max(worst_db, db_lo, db_hi)

        sidebands.append({
            "order": k,
            "lower": {
                "expected_hz": round(f_lo, 4),
                **sb_lo,
                "db_relative": round(float(db_lo), 2),
            },
            "upper": {
                "expected_hz": round(f_hi, 4),
                **sb_hi,
                "db_relative": round(float(db_hi), 2),
            },
        })

    return {
        "fault_type": f"bearing_{defect_type}",
        "defect_frequency_hz": round(fd, 4),
        "fundamental": {
            "expected_hz": fs,
            **fund,
        },
        "sidebands": sidebands,
        "worst_sideband_db": round(float(worst_db), 2),
        "note": (
            "Bearing signatures in stator current are typically weak. "
            "Confirm with envelope analysis or vibration measurements."
        ),
    }


# ---------------------------------------------------------------------------
# Band energy index (cavitation, load faults)
# ---------------------------------------------------------------------------

def band_energy_index(
    freqs: NDArray[np.floating],
    psd: NDArray[np.floating],
    centre_freq_hz: float,
    bandwidth_hz: float = 5.0,
) -> dict:
    """Compute the integrated spectral energy in a frequency band.

    Useful as generic fault/cavitation indicator: the energy in a band
    around the supply frequency or other characteristic region.

    Args:
        freqs: Frequency axis (from PSD).
        psd: PSD values.
        centre_freq_hz: Centre of the integration band.
        bandwidth_hz: Total bandwidth for integration.

    Returns:
        Dictionary with band energy, limits used.
    """
    low = centre_freq_hz - bandwidth_hz / 2.0
    high = centre_freq_hz + bandwidth_hz / 2.0

    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return {
            "centre_freq_hz": centre_freq_hz,
            "bandwidth_hz": bandwidth_hz,
            "band_energy": 0.0,
            "found": False,
        }

    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
    energy = float(np.sum(psd[mask]) * df)

    return {
        "centre_freq_hz": centre_freq_hz,
        "bandwidth_hz": bandwidth_hz,
        "band_low_hz": round(low, 4),
        "band_high_hz": round(high, 4),
        "band_energy": energy,
        "found": True,
    }


# ---------------------------------------------------------------------------
# Statistical indices on envelope
# ---------------------------------------------------------------------------

def envelope_statistical_indices(
    envelope: NDArray[np.floating],
) -> dict:
    """Compute statistical indices of the envelope signal.

    Kurtosis, skewness, crest factor, and RMS — indicators of impulsive
    content from bearing or gear faults.

    Args:
        envelope: Amplitude envelope of the current signal.

    Returns:
        Dictionary of statistical indices.
    """
    from scipy.stats import kurtosis, skew

    rms = float(np.sqrt(np.mean(envelope ** 2)))
    peak = float(np.max(np.abs(envelope)))
    crest = peak / rms if rms > 0 else 0.0

    return {
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "crest_factor": round(crest, 4),
        "kurtosis": round(float(kurtosis(envelope, fisher=True)), 4),
        "skewness": round(float(skew(envelope)), 4),
    }
