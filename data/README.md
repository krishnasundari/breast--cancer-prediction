# Wisconsin Breast Cancer Diagnostic Dataset

## Dataset Overview
- **Instances:** 569
- **Features:** 30 continuous real-valued features computed from fine needle aspirate (FNA) images of breast masses
- **Target Variable:** `diagnosis`
  - `M` = Malignant (212 samples, ~37.3%)
  - `B` = Benign (357 samples, ~62.7%)
- **Source:** University of Wisconsin Clinical Sciences Center (Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian) / UCI Machine Learning Repository / Kaggle.

---

## Feature Definitions
Ten real-valued characteristics are computed for each cell nucleus:
1. **Radius:** Mean of distances from center to points on the perimeter
2. **Texture:** Standard deviation of gray-scale values
3. **Perimeter:** Total perimeter length of the cell nucleus
4. **Area:** Total area enclosed by the nuclear contour
5. **Smoothness:** Local variation in radius lengths
6. **Compactness:** $\frac{\text{perimeter}^2}{\text{area} - 1.0}$
7. **Concavity:** Severity of concave portions of the contour
8. **Concave Points:** Number of concave portions of the contour
9. **Symmetry:** Relative symmetry of nuclear geometry
10. **Fractal Dimension:** "Coastline approximation" - 1.0

For each image, the **Mean**, **Standard Error (SE)**, and **"Worst" (mean of the three largest values)** were computed for each characteristic, yielding **$3 \times 10 = 30$ features**.

---

## Missing Values
There are **zero missing attribute values** in the 30 continuous measurement columns.
The redundant `id` column and trailing empty column `Unnamed: 32` are dropped during pipeline preprocessing.
