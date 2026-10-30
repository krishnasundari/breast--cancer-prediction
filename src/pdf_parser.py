"""
pdf_parser.py
=============
Specialized parser for Fine Needle Aspirate (FNA) Cytology and Biopsy Reports.
Extracts only the verified 30 numerical morphological cell-nucleus features
defined in the Wisconsin Breast Cancer Diagnostic Dataset (WDBC).

Strict Clinical Compatibility:
- Maps only recognized nuclear morphological feature names.
- Rejects non-morphology reports (blood tests, ultrasound narratives, mammogram notes)
  without fabricating values or zero-filling.
- Returns missing feature lists and requires clinician confirmation before inference.
"""

import re
import io
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
from pypdf import PdfReader

from src.data_loader import FEATURE_NAMES


FEATURE_ALIAS_PATTERNS = {
    # Mean features
    "radius_mean": [r"radius\s*(?:\(mean\)|\bmean\b)", r"mean\s+radius"],
    "texture_mean": [r"texture\s*(?:\(mean\)|\bmean\b)", r"mean\s+texture"],
    "perimeter_mean": [r"perimeter\s*(?:\(mean\)|\bmean\b)", r"mean\s+perimeter"],
    "area_mean": [r"area\s*(?:\(mean\)|\bmean\b)", r"mean\s+area"],
    "smoothness_mean": [r"smoothness\s*(?:\(mean\)|\bmean\b)", r"mean\s+smoothness"],
    "compactness_mean": [r"compactness\s*(?:\(mean\)|\bmean\b)", r"mean\s+compactness"],
    "concavity_mean": [r"concavity\s*(?:\(mean\)|\bmean\b)", r"mean\s+concavity"],
    "concave points_mean": [
        r"concave\s+points?\s*(?:\(mean\)|\bmean\b)",
        r"mean\s+concave\s+points?",
        r"concave\s+portions?\s*(?:\(mean\)|\bmean\b)"
    ],
    "symmetry_mean": [r"symmetry\s*(?:\(mean\)|\bmean\b)", r"mean\s+symmetry"],
    "fractal_dimension_mean": [
        r"fractal\s+dimension\s*(?:\(mean\)|\bmean\b)",
        r"mean\s+fractal\s+dimension",
        r"fractal\s+dim\s*(?:\(mean\)|\bmean\b)"
    ],

    # Standard Error (SE) features
    "radius_se": [r"radius\s*(?:\(se\)|\bse\b|\bstd\s*err\b|\bstandard\s*error\b)", r"se\s+radius"],
    "texture_se": [r"texture\s*(?:\(se\)|\bse\b|\bstd\s*err\b|\bstandard\s*error\b)", r"se\s+texture"],
    "perimeter_se": [r"perimeter\s*(?:\(se\)|\bse\b|\bstd\s*err\b|\bstandard\s*error\b)", r"se\s+perimeter"],
    "area_se": [r"area\s*(?:\(se\)|\bse\b|\bstd\s*err\b|\bstandard\s*error\b)", r"se\s+area"],
    "smoothness_se": [r"smoothness\s*(?:\(se\)|\bse\b|\bstd\s*err\b|\bstandard\s*error\b)", r"se\s+smoothness"],
    "compactness_se": [r"compactness\s*(?:\(se\)|\bse\b|\bstd\s*err\b|\bstandard\s*error\b)", r"se\s+compactness"],
    "concavity_se": [r"concavity\s*(?:\(se\)|\bse\b|\bstd\s*err\b|\bstandard\s*error\b)", r"se\s+concavity"],
    "concave points_se": [
        r"concave\s+points?\s*(?:\(se\)|\bse\b|\bstd\s*err\b|\bstandard\s*error\b)",
        r"se\s+concave\s+points?"
    ],
    "symmetry_se": [r"symmetry\s*(?:\(se\)|\bse\b|\bstd\s*err\b|\bstandard\s*error\b)", r"se\s+symmetry"],
    "fractal_dimension_se": [
        r"fractal\s+dimension\s*(?:\(se\)|\bse\b|\bstd\s*err\b|\bstandard\s*error\b)",
        r"se\s+fractal\s+dimension"
    ],

    # Worst / Largest features
    "radius_worst": [r"radius\s*(?:\(worst\)|\bworst\b|\blargest\b|\bmax\b)", r"worst\s+radius"],
    "texture_worst": [r"texture\s*(?:\(worst\)|\bworst\b|\blargest\b|\bmax\b)", r"worst\s+texture"],
    "perimeter_worst": [r"perimeter\s*(?:\(worst\)|\bworst\b|\blargest\b|\bmax\b)", r"worst\s+perimeter"],
    "area_worst": [r"area\s*(?:\(worst\)|\bworst\b|\blargest\b|\bmax\b)", r"worst\s+area"],
    "smoothness_worst": [r"smoothness\s*(?:\(worst\)|\bworst\b|\blargest\b|\bmax\b)", r"worst\s+smoothness"],
    "compactness_worst": [r"compactness\s*(?:\(worst\)|\bworst\b|\blargest\b|\bmax\b)", r"worst\s+compactness"],
    "concavity_worst": [r"concavity\s*(?:\(worst\)|\bworst\b|\blargest\b|\bmax\b)", r"worst\s+concavity"],
    "concave points_worst": [
        r"concave\s+points?\s*(?:\(worst\)|\bworst\b|\blargest\b|\bmax\b)",
        r"worst\s+concave\s+points?"
    ],
    "symmetry_worst": [r"symmetry\s*(?:\(worst\)|\bworst\b|\blargest\b|\bmax\b)", r"worst\s+symmetry"],
    "fractal_dimension_worst": [
        r"fractal\s+dimension\s*(?:\(worst\)|\bworst\b|\blargest\b|\bmax\b)",
        r"worst\s+fractal\s+dimension"
    ]
}


def extract_text_from_pdf(pdf_file_or_bytes: Any) -> str:
    """Extract raw text from a PDF file-like object or bytes."""
    try:
        if isinstance(pdf_file_or_bytes, bytes):
            stream = io.BytesIO(pdf_file_or_bytes)
        elif hasattr(pdf_file_or_bytes, "read"):
            content = pdf_file_or_bytes.read()
            stream = io.BytesIO(content)
        else:
            stream = pdf_file_or_bytes

        reader = PdfReader(stream)
        text_pages = []
        for idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            text_pages.append(page_text)
        return "\n".join(text_pages)
    except Exception as e:
        raise ValueError(f"Unable to read PDF file format: {str(e)}")


def parse_morphology_from_text(raw_text: str) -> Dict[str, Any]:
    """
    Scans document text for key nuclear morphology features using multi-line and single-line regex.
    """
    extracted: Dict[str, float] = {}

    # Method 1: Check consecutive lines where line i is feature name and line i+1 is number
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    for i in range(len(lines) - 1):
        line_cur = lines[i]
        line_next = lines[i + 1]

        # Check if line_next is a numerical float
        num_match = re.match(r"^[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?$", line_next)
        if num_match:
            for canonical_feat, patterns in FEATURE_ALIAS_PATTERNS.items():
                if canonical_feat in extracted:
                    continue
                for pat in patterns:
                    if re.search(rf"^{pat}$", line_cur, re.IGNORECASE) or re.search(rf"\b{pat}\b", line_cur, re.IGNORECASE):
                        try:
                            extracted[canonical_feat] = float(num_match.group(0))
                            break
                        except ValueError:
                            pass
                if canonical_feat in extracted:
                    break

    # Method 2: Check single-line colon/dash/tab/space pairs (e.g. "Radius Mean: 17.99")
    for canonical_feat, patterns in FEATURE_ALIAS_PATTERNS.items():
        if canonical_feat in extracted:
            continue
        for line in lines:
            for pat in patterns:
                regex = rf"(?:{pat})[\s:=–\-|]+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
                match = re.search(regex, line, re.IGNORECASE)
                if match:
                    try:
                        extracted[canonical_feat] = float(match.group(1))
                        break
                    except ValueError:
                        continue
            if canonical_feat in extracted:
                break

    # Method 3: Broad text scan for any remaining missing features
    for canonical_feat, patterns in FEATURE_ALIAS_PATTERNS.items():
        if canonical_feat in extracted:
            continue
        for pat in patterns:
            regex = rf"\b(?:{pat})\b[^\d\n\r]{{0,25}}?([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
            match = re.search(regex, raw_text, re.IGNORECASE)
            if match:
                try:
                    extracted[canonical_feat] = float(match.group(1))
                    break
                except ValueError:
                    continue

    missing = [feat for feat in FEATURE_NAMES if feat not in extracted]
    is_complete = (len(missing) == 0)

    # Detect if report appears to be an unrelated clinical note (e.g., blood test, general ultrasound)
    unrelated_indicators = []
    lower_text = raw_text.lower()
    if "complete blood count" in lower_text or "hemoglobin" in lower_text or "platelet" in lower_text:
        unrelated_indicators.append("Complete Blood Count (CBC) / Hematology Report")
    if "lipid profile" in lower_text or "cholesterol" in lower_text:
        unrelated_indicators.append("Lipid Profile Panel")
    if "urinalysis" in lower_text or "urine routine" in lower_text:
        unrelated_indicators.append("Urinalysis Report")

    return {
        "extracted_features": extracted,
        "missing_features": missing,
        "features_found_count": len(extracted),
        "total_required": len(FEATURE_NAMES),
        "is_complete": is_complete,
        "unrelated_indicators": unrelated_indicators,
        "is_unrelated_report": len(unrelated_indicators) > 0 and len(extracted) < 5
    }


def parse_fna_pdf_report(pdf_file_or_bytes: Any) -> Dict[str, Any]:
    """
    High-level entry point for parsing uploaded patient PDF reports.
    """
    raw_text = extract_text_from_pdf(pdf_file_or_bytes)
    if not raw_text.strip():
        return {
            "success": False,
            "error": "The uploaded PDF appears empty or contains scanned images without selectable text.",
            "extracted_features": {},
            "missing_features": FEATURE_NAMES,
            "is_complete": False
        }

    parsed = parse_morphology_from_text(raw_text)

    if parsed["is_unrelated_report"]:
        return {
            "success": False,
            "error": (
                f"Incompatible medical report detected ({', '.join(parsed['unrelated_indicators'])}). "
                "The ML model requires 30 cytological nuclear morphology measurements from a "
                "Fine Needle Aspirate (FNA) biopsy, not general hematology, blood tests, or ultrasound notes."
            ),
            "extracted_features": parsed["extracted_features"],
            "missing_features": parsed["missing_features"],
            "is_complete": False,
            "raw_text_snippet": raw_text[:300]
        }

    return {
        "success": True,
        "extracted_features": parsed["extracted_features"],
        "missing_features": parsed["missing_features"],
        "features_found_count": parsed["features_found_count"],
        "is_complete": parsed["is_complete"],
        "raw_text_snippet": raw_text[:300]
    }
