"""Tests for bearing defect frequency calculations."""

import math

import pytest

from mcp_server_mcsa.analysis.bearing import (
    BearingGeometry,
    bearing_current_sidebands,
    calculate_bearing_defect_frequencies,
)


class TestBearingDefectFrequencies:
    """Test bearing defect frequency computations against known values."""

    @pytest.fixture
    def typical_bearing(self):
        """6205 bearing approximate geometry."""
        return BearingGeometry(
            n_balls=9,
            ball_dia_mm=7.94,
            pitch_dia_mm=38.5,
            contact_angle_deg=0.0,
        )

    def test_ftf_range(self, typical_bearing):
        defects = calculate_bearing_defect_frequencies(typical_bearing)
        # FTF is always < 0.5 × shaft speed
        assert 0 < defects.ftf < 0.5

    def test_bpfo_greater_than_bpfi_for_zero_angle(self, typical_bearing):
        # For zero contact angle, BPFO < BPFI typically
        defects = calculate_bearing_defect_frequencies(typical_bearing)
        # BPFI should be > BPFO for most bearings
        assert defects.bpfi > defects.bpfo

    def test_bpfo_equals_n_times_ftf(self, typical_bearing):
        defects = calculate_bearing_defect_frequencies(typical_bearing)
        assert defects.bpfo == pytest.approx(
            typical_bearing.n_balls * defects.ftf, rel=1e-10
        )

    def test_absolute_frequencies(self, typical_bearing):
        defects = calculate_bearing_defect_frequencies(typical_bearing)
        shaft_freq = 24.5  # 1470 RPM
        abs_freqs = defects.absolute(shaft_freq)
        assert abs_freqs["bpfo_hz"] == pytest.approx(defects.bpfo * shaft_freq, rel=1e-10)
        assert abs_freqs["ftf_hz"] == pytest.approx(defects.ftf * shaft_freq, rel=1e-10)

    def test_invalid_geometry(self):
        with pytest.raises(ValueError):
            calculate_bearing_defect_frequencies(
                BearingGeometry(n_balls=0, ball_dia_mm=5, pitch_dia_mm=30)
            )

    def test_ball_larger_than_pitch_raises(self):
        with pytest.raises(ValueError):
            calculate_bearing_defect_frequencies(
                BearingGeometry(n_balls=8, ball_dia_mm=50, pitch_dia_mm=30)
            )

    def test_contact_angle_effect(self):
        geom_0 = BearingGeometry(n_balls=9, ball_dia_mm=8, pitch_dia_mm=40, contact_angle_deg=0)
        geom_30 = BearingGeometry(n_balls=9, ball_dia_mm=8, pitch_dia_mm=40, contact_angle_deg=30)
        d0 = calculate_bearing_defect_frequencies(geom_0)
        d30 = calculate_bearing_defect_frequencies(geom_30)
        # Contact angle reduces the effective ratio → changes frequencies
        assert d0.bpfo != d30.bpfo


class TestBearingCurrentSidebands:
    def test_sideband_structure(self):
        geom = BearingGeometry(n_balls=9, ball_dia_mm=7.94, pitch_dia_mm=38.5)
        defects = calculate_bearing_defect_frequencies(geom)
        result = bearing_current_sidebands(defects, shaft_freq_hz=24.5, supply_freq_hz=50.0)

        assert "bpfo" in result
        assert "bpfi" in result
        assert len(result["bpfo"]["current_sidebands"]) == 2  # default harmonics=2

    def test_sidebands_symmetric(self):
        geom = BearingGeometry(n_balls=9, ball_dia_mm=7.94, pitch_dia_mm=38.5)
        defects = calculate_bearing_defect_frequencies(geom)
        result = bearing_current_sidebands(defects, shaft_freq_hz=24.5, supply_freq_hz=50.0)

        for defect_data in result.values():
            fd = defect_data["defect_frequency_hz"]
            for sb in defect_data["current_sidebands"]:
                assert sb["upper_hz"] - 50.0 == pytest.approx(
                    50.0 - sb["lower_hz"], abs=0.01
                )
