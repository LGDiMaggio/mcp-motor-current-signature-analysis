"""Bearing defect frequency calculations.

Computes characteristic defect frequencies (BPFO, BPFI, BSF, FTF) from
bearing geometry and shaft speed, plus their expected sidebands in the
stator‑current spectrum.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BearingGeometry:
    """Physical geometry of a rolling‑element bearing.

    Attributes:
        n_balls: Number of rolling elements.
        ball_dia_mm: Ball (roller) diameter in mm.
        pitch_dia_mm: Pitch (cage) diameter in mm.
        contact_angle_deg: Contact angle in degrees.
    """

    n_balls: int
    ball_dia_mm: float
    pitch_dia_mm: float
    contact_angle_deg: float = 0.0


@dataclass(frozen=True)
class BearingDefectFrequencies:
    """Characteristic defect frequencies normalised to shaft speed.

    All values are multiples of the shaft rotational frequency f_r.
    Multiply by f_r (Hz) to get absolute frequencies.

    Attributes:
        bpfo: Ball Pass Frequency — Outer race.
        bpfi: Ball Pass Frequency — Inner race.
        bsf: Ball Spin Frequency.
        ftf: Fundamental Train (cage) Frequency.
    """

    bpfo: float
    bpfi: float
    bsf: float
    ftf: float

    def to_dict(self) -> dict:
        return asdict(self)

    def absolute(self, shaft_freq_hz: float) -> dict:
        """Return absolute frequencies in Hz given shaft speed."""
        return {
            "bpfo_hz": self.bpfo * shaft_freq_hz,
            "bpfi_hz": self.bpfi * shaft_freq_hz,
            "bsf_hz": self.bsf * shaft_freq_hz,
            "ftf_hz": self.ftf * shaft_freq_hz,
        }


def calculate_bearing_defect_frequencies(
    geometry: BearingGeometry,
) -> BearingDefectFrequencies:
    """Compute bearing defect frequencies (normalised to shaft speed).

    Standard kinematic equations for rolling‑element bearings.

    Args:
        geometry: Bearing physical dimensions.

    Returns:
        Normalised defect frequencies (multiply by f_rotor to get Hz).

    Raises:
        ValueError: If geometry parameters are physically invalid.
    """
    n = geometry.n_balls
    d = geometry.ball_dia_mm
    D = geometry.pitch_dia_mm
    alpha = math.radians(geometry.contact_angle_deg)

    if n < 1:
        raise ValueError(f"Number of balls must be ≥ 1, got {n}")
    if d <= 0 or D <= 0:
        raise ValueError("Ball and pitch diameters must be > 0")
    if d >= D:
        raise ValueError("Ball diameter must be < pitch diameter")

    cos_alpha = math.cos(alpha)
    ratio = d / D

    # Fundamental Train Frequency (cage speed / shaft speed)
    ftf = 0.5 * (1.0 - ratio * cos_alpha)

    # Ball Pass Frequency — Outer race
    bpfo = n * ftf

    # Ball Pass Frequency — Inner race
    bpfi = 0.5 * n * (1.0 + ratio * cos_alpha)

    # Ball Spin Frequency
    bsf = (D / (2.0 * d)) * (1.0 - (ratio * cos_alpha) ** 2)

    return BearingDefectFrequencies(bpfo=bpfo, bpfi=bpfi, bsf=bsf, ftf=ftf)


def bearing_current_sidebands(
    defect_freqs: BearingDefectFrequencies,
    shaft_freq_hz: float,
    supply_freq_hz: float,
    harmonics: int = 2,
) -> dict:
    """Compute expected stator‑current sidebands from bearing defects.

    Bearing defects modulate motor torque, producing sidebands in the
    stator current at f_supply ± k · f_defect.

    Args:
        defect_freqs: Normalised bearing defect frequencies.
        shaft_freq_hz: Shaft rotational frequency in Hz.
        supply_freq_hz: Supply (line) frequency in Hz.
        harmonics: Number of sideband orders.

    Returns:
        Dictionary mapping defect type → list of sideband frequencies.
    """
    abs_freqs = defect_freqs.absolute(shaft_freq_hz)
    fs = supply_freq_hz

    result = {}
    for name, fdef in abs_freqs.items():
        label = name.replace("_hz", "")
        sidebands = []
        for k in range(1, harmonics + 1):
            sidebands.append({
                "order": k,
                "lower_hz": fs - k * fdef,
                "upper_hz": fs + k * fdef,
            })
        result[label] = {
            "defect_frequency_hz": fdef,
            "current_sidebands": sidebands,
        }

    return result
