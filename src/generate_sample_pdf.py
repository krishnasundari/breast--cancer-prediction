"""
generate_sample_pdf.py
======================
Generates genuine, authentic-looking Wisconsin FNA Biopsy and CBC test PDFs
for testing and demonstrating the PDF extraction pipeline.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def create_fna_sample_pdf(output_path: str, patient_id: str = "PT-842302", is_malignant: bool = True):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    doc = SimpleDocTemplate(output_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'], fontSize=16, leading=20, textColor=colors.HexColor('#1A237E')
    )
    subtitle_style = ParagraphStyle(
        'DocSub', parent=styles['Normal'], fontSize=9, textColor=colors.HexColor('#455A64')
    )
    header_style = ParagraphStyle(
        'SecHeader', parent=styles['Heading2'], fontSize=12, leading=16, textColor=colors.HexColor('#C2185B'), spaceBefore=8, spaceAfter=4
    )
    body_style = ParagraphStyle(
        'Body', parent=styles['Normal'], fontSize=9, leading=12
    )

    story.append(Paragraph("ONCOLOGY PATHOLOGY LABORATORY SERVICES", title_style))
    story.append(Paragraph("CYTOPATHOLOGY EVALUATION REPORT — FINE NEEDLE ASPIRATE (FNA) BIOPSY", subtitle_style))
    story.append(Spacer(1, 10))

    patient_meta = [
        [Paragraph(f"<b>Patient Reference ID:</b> {patient_id}", body_style), Paragraph("<b>Specimen:</b> Right Breast Upper Outer Quadrant", body_style)],
        [Paragraph("<b>Clinical Indication:</b> Palpable discrete breast nodule", body_style), Paragraph("<b>Pathologist:</b> Dr. E. Bennett, MD, FCAP", body_style)]
    ]
    meta_table = Table(patient_meta, colWidths=[270, 270])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8F9FA')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CFD8DC')),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("COMPUTER-ASSISTED NUCLEAR MORPHOMETRY (WISCONSIN CRITERIA)", header_style))
    story.append(Paragraph("Digital image analysis of nuclear morphology from FNA aspirates. Thirty quantitative geometric metrics computed from digitized cell boundaries.", body_style))
    story.append(Spacer(1, 8))

    if is_malignant:
        data = [
            ["Morphological Feature", "Mean Value", "Standard Error (SE)", "Worst / Maximum Value"],
            ["Radius", "17.9900", "1.0950", "25.3800"],
            ["Texture", "10.3800", "0.9053", "17.3300"],
            ["Perimeter", "122.8000", "8.5890", "184.6000"],
            ["Area", "1001.0000", "153.4000", "2019.0000"],
            ["Smoothness", "0.1184", "0.0064", "0.1622"],
            ["Compactness", "0.2776", "0.0490", "0.6656"],
            ["Concavity", "0.3001", "0.0537", "0.7119"],
            ["Concave Points", "0.1471", "0.0159", "0.2654"],
            ["Symmetry", "0.2419", "0.0300", "0.4601"],
            ["Fractal Dimension", "0.0787", "0.0062", "0.1189"]
        ]
    else:
        data = [
            ["Morphological Feature", "Mean Value", "Standard Error (SE)", "Worst / Maximum Value"],
            ["Radius", "13.5400", "0.2699", "15.1100"],
            ["Texture", "14.3600", "0.7886", "19.2600"],
            ["Perimeter", "87.4600", "2.0580", "99.7000"],
            ["Area", "566.3000", "23.5600", "711.2000"],
            ["Smoothness", "0.0978", "0.0085", "0.1440"],
            ["Compactness", "0.0813", "0.0146", "0.1773"],
            ["Concavity", "0.0666", "0.0239", "0.2390"],
            ["Concave Points", "0.0478", "0.0132", "0.1288"],
            ["Symmetry", "0.1885", "0.0198", "0.2977"],
            ["Fractal Dimension", "0.0577", "0.0023", "0.0726"]
        ]

    # Convert to explicit 30 feature key-value table lines for parsing fidelity
    kv_data = [["Nuclear Feature Metric", "Clinical Quantitation"]]
    if is_malignant:
        feats = [
            ("radius_mean", "17.99"), ("texture_mean", "10.38"), ("perimeter_mean", "122.8"), ("area_mean", "1001.0"),
            ("smoothness_mean", "0.1184"), ("compactness_mean", "0.2776"), ("concavity_mean", "0.3001"),
            ("concave points_mean", "0.1471"), ("symmetry_mean", "0.2419"), ("fractal_dimension_mean", "0.07871"),
            ("radius_se", "1.095"), ("texture_se", "0.9053"), ("perimeter_se", "8.589"), ("area_se", "153.4"),
            ("smoothness_se", "0.006399"), ("compactness_se", "0.04904"), ("concavity_se", "0.05373"),
            ("concave points_se", "0.01587"), ("symmetry_se", "0.03003"), ("fractal_dimension_se", "0.006193"),
            ("radius_worst", "25.38"), ("texture_worst", "17.33"), ("perimeter_worst", "184.6"), ("area_worst", "2019.0"),
            ("smoothness_worst", "0.1622"), ("compactness_worst", "0.6656"), ("concavity_worst", "0.7119"),
            ("concave points_worst", "0.2654"), ("symmetry_worst", "0.4601"), ("fractal_dimension_worst", "0.1189")
        ]
    else:
        feats = [
            ("radius_mean", "13.54"), ("texture_mean", "14.36"), ("perimeter_mean", "87.46"), ("area_mean", "566.3"),
            ("smoothness_mean", "0.09779"), ("compactness_mean", "0.08129"), ("concavity_mean", "0.06664"),
            ("concave points_mean", "0.04781"), ("symmetry_mean", "0.1885"), ("fractal_dimension_mean", "0.05766"),
            ("radius_se", "0.2699"), ("texture_se", "0.7886"), ("perimeter_se", "2.058"), ("area_se", "23.56"),
            ("smoothness_se", "0.008462"), ("compactness_se", "0.0146"), ("concavity_se", "0.02387"),
            ("concave points_se", "0.01315"), ("symmetry_se", "0.0198"), ("fractal_dimension_se", "0.0023"),
            ("radius_worst", "15.11"), ("texture_worst", "19.26"), ("perimeter_worst", "99.7"), ("area_worst", "711.2"),
            ("smoothness_worst", "0.144"), ("compactness_worst", "0.1773"), ("concavity_worst", "0.239"),
            ("concave points_worst", "0.1288"), ("symmetry_worst", "0.2977"), ("fractal_dimension_worst", "0.07259")
        ]

    for f_name, f_val in feats:
        kv_data.append([f_name.replace('_', ' ').title(), f_val])

    # Display in 2 columns
    col1 = kv_data[:16]
    col2 = [["Nuclear Feature Metric", "Clinical Quantitation"]] + kv_data[16:]
    merged_rows = []
    for r1, r2 in zip(col1, col2):
        merged_rows.append([r1[0], r1[1], r2[0], r2[1]])

    morph_table = Table(merged_rows, colWidths=[170, 100, 170, 100])
    morph_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#ECEFF1')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#263238')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CFD8DC')),
    ]))
    story.append(morph_table)
    story.append(Spacer(1, 14))

    story.append(Paragraph("DIAGNOSTIC PATHOLOGY IMPRESSION", header_style))
    if is_malignant:
        story.append(Paragraph("<b>Impression:</b> Hyperchromatic, enlarged pleomorphic ductal epithelial cells exhibiting irregular nuclear contours and deep membrane invaginations. Morphometric dimensions substantiate high cytologic risk.", body_style))
    else:
        story.append(Paragraph("<b>Impression:</b> Cohesive sheets of uniform ductal epithelial cells exhibiting bland nuclei, smooth contours, and absent nuclear atypia. Cytologic dimensions consistent with benign fibroadenoma.", body_style))

    doc.build(story)
    return output_path


def create_incomplete_fna_sample_pdf(output_path: str):
    """Creates a sample PDF missing 6 features (e.g. fractal dimensions & symmetry worst)."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    doc = SimpleDocTemplate(output_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'], fontSize=16, leading=20, textColor=colors.HexColor('#1A237E')
    )
    header_style = ParagraphStyle(
        'SecHeader', parent=styles['Heading2'], fontSize=12, leading=16, textColor=colors.HexColor('#C2185B'), spaceBefore=8, spaceAfter=4
    )
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=9, leading=12)

    story.append(Paragraph("ONCOLOGY PATHOLOGY LABORATORY SERVICES", title_style))
    story.append(Paragraph("PARTIAL PRELIMINARY CYTOLOGY EVALUATION REPORT", styles['Normal']))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Notice:</b> Preliminary report. Fractal dimension metrics pending secondary scan confirmation.", body_style))
    story.append(Spacer(1, 10))

    # Features excluding fractal dimensions and symmetry worst
    data = [
        ["Morphological Feature", "Mean Value", "Standard Error (SE)", "Worst / Maximum Value"],
        ["Radius", "17.9900", "1.0950", "25.3800"],
        ["Texture", "10.3800", "0.9053", "17.3300"],
        ["Perimeter", "122.8000", "8.5890", "184.6000"],
        ["Area", "1001.0000", "153.4000", "2019.0000"],
        ["Smoothness", "0.1184", "0.0064", "0.1622"],
        ["Compactness", "0.2776", "0.0490", "0.6656"],
        ["Concavity", "0.3001", "0.0537", "0.7119"],
        ["Concave Points", "0.1471", "0.0159", "0.2654"],
    ]
    t = Table(data, colWidths=[150, 110, 120, 130])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E1BEE7')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CE93D8')),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t)
    doc.build(story)
    return output_path


def create_unrelated_cbc_pdf(output_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    doc = SimpleDocTemplate(output_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=16, leading=20, textColor=colors.HexColor('#004D40'))
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=9, leading=12)

    story.append(Paragraph("GENERAL CLINICAL HEMATOLOGY LABORATORY", title_style))
    story.append(Paragraph("COMPLETE BLOOD COUNT (CBC) & METABOLIC PANEL", styles['Normal']))
    story.append(Spacer(1, 12))

    data = [
        ["Hematology Test", "Result", "Reference Range", "Units"],
        ["Hemoglobin", "13.8", "12.0 - 15.5", "g/dL"],
        ["Platelet Count", "250", "150 - 450", "x10^3/uL"],
        ["White Blood Cells (WBC)", "6.8", "4.5 - 11.0", "x10^3/uL"],
        ["Hematocrit", "41.2", "37.0 - 48.0", "%"],
        ["Fasting Blood Glucose", "95", "70 - 99", "mg/dL"],
        ["Total Cholesterol", "180", "< 200", "mg/dL"]
    ]
    t = Table(data, colWidths=[180, 90, 130, 90])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E0F2F1')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#B2DFDB')),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))
    story.append(Paragraph("Note: This hematology report contains routine blood values and lacks cytologic nuclear morphology.", body_style))

    doc.build(story)
    return output_path


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    samples_dir = os.path.join(base_dir, "data", "sample_reports")
    p1 = os.path.join(samples_dir, "fna_biopsy_sample_malignant.pdf")
    p2 = os.path.join(samples_dir, "fna_biopsy_sample_benign.pdf")
    p3 = os.path.join(samples_dir, "unrelated_blood_test.pdf")

    create_fna_sample_pdf(p1, patient_id="PT-842302", is_malignant=True)
    create_fna_sample_pdf(p2, patient_id="PT-8510426", is_malignant=False)
    create_unrelated_cbc_pdf(p3)
    print("Created sample PDF reports in:", samples_dir)
