"""
test_pdf_parser.py
==================
Unit tests for the Fine Needle Aspirate (FNA) PDF pathology report parser.
Tests extraction accuracy, 30-feature completeness, rejection of incompatible medical reports,
and error handling.
"""

import os
import pytest
from src.pdf_parser import parse_fna_pdf_report, parse_morphology_from_text
from src.data_loader import FEATURE_NAMES


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "sample_reports")


def test_parse_fna_malignant_pdf():
    """Verify parser extracts all 30 features from authentic malignant FNA biopsy PDF."""
    pdf_path = os.path.join(SAMPLES_DIR, "fna_biopsy_sample_malignant.pdf")
    assert os.path.exists(pdf_path), "Sample malignant PDF does not exist"

    with open(pdf_path, "rb") as f:
        res = parse_fna_pdf_report(f)

    assert res["success"] is True
    assert res["is_complete"] is True
    assert res["features_found_count"] == 30
    assert len(res["missing_features"]) == 0

    feats = res["extracted_features"]
    assert feats["radius_mean"] == pytest.approx(17.99, rel=1e-3)
    assert feats["texture_mean"] == pytest.approx(10.38, rel=1e-3)
    assert feats["area_worst"] == pytest.approx(2019.0, rel=1e-3)


def test_parse_fna_benign_pdf():
    """Verify parser extracts all 30 features from authentic benign FNA biopsy PDF."""
    pdf_path = os.path.join(SAMPLES_DIR, "fna_biopsy_sample_benign.pdf")
    assert os.path.exists(pdf_path), "Sample benign PDF does not exist"

    with open(pdf_path, "rb") as f:
        res = parse_fna_pdf_report(f)

    assert res["success"] is True
    assert res["is_complete"] is True
    assert res["features_found_count"] == 30
    assert len(res["missing_features"]) == 0

    feats = res["extracted_features"]
    assert feats["radius_mean"] == pytest.approx(13.54, rel=1e-3)
    assert feats["texture_mean"] == pytest.approx(14.36, rel=1e-3)
    assert feats["area_worst"] == pytest.approx(711.2, rel=1e-3)


def test_reject_unrelated_medical_report():
    """Verify parser detects and rejects non-morphology reports (e.g. CBC blood test)."""
    pdf_path = os.path.join(SAMPLES_DIR, "unrelated_blood_test.pdf")
    assert os.path.exists(pdf_path), "Sample blood test PDF does not exist"

    with open(pdf_path, "rb") as f:
        res = parse_fna_pdf_report(f)

    assert res["success"] is False
    assert "Incompatible medical report detected" in res["error"]
    assert res["is_complete"] is False


def test_incomplete_text_reports_missing_features():
    """Verify parser identifies missing features when only partial measurements are present."""
    partial_text = """
    CYTOPATHOLOGY REPORT
    Radius Mean: 15.2
    Texture Mean: 18.4
    Area Mean: 650.0
    """
    res = parse_morphology_from_text(partial_text)
    assert res["is_complete"] is False
    assert res["features_found_count"] == 3
    assert len(res["missing_features"]) == 27
    assert "radius_mean" in res["extracted_features"]
    assert "perimeter_worst" in res["missing_features"]
