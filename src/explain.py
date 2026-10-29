"""
explain.py
==========
Explainable AI (XAI) module using SHAP (SHapley Additive exPlanations).
Provides model-specific explainers (TreeExplainer, LinearExplainer, KernelExplainer),
local individual instance explanations (waterfall plots, contribution rankings),
and global feature importance analysis (summary plots, reports).
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from src.data_loader import (
    get_train_test_split,
    FEATURE_NAMES
)
from src.predict import load_trained_pipeline


def get_training_background(
    X_train: Optional[pd.DataFrame] = None,
    n_samples: int = 30,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Returns a representative background sample strictly from training data.
    Never uses the held-out test set to prevent leakage.
    """
    if X_train is None:
        X_train, _, _, _ = get_train_test_split(random_state=42, stratify=True)
    if len(X_train) <= n_samples:
        return X_train.copy()
    return shap.sample(X_train, n_samples, random_state=random_state)


def get_model_explainer(
    model_name: str,
    pipeline: Any,
    X_train: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Constructs and returns the appropriate SHAP explainer for the given model architecture.

    Selection Justification:
    ------------------------
    1. Random Forest -> shap.TreeExplainer
       TreeExplainer computes exact polynomial-time Shapley values across all decision trees
       without sampling variance.
    2. Logistic Regression -> shap.LinearExplainer
       LinearExplainer computes exact closed-form Shapley values for linear decision margins
       using an independent training feature masker.
    3. Support Vector Machine (SVM) -> shap.KernelExplainer
       Non-linear RBF kernel cannot use tree/linear methods; KernelExplainer evaluates
       calibrated predict_proba with a representative background sample.
    4. K-Nearest Neighbors (KNN) -> shap.KernelExplainer
       Non-parametric, distance-based neighborhood algorithm evaluated via KernelExplainer
       with a representative background sample.
    """
    canonical_name = str(model_name).lower().strip()
    if canonical_name in ["rf", "random forest"]:
        canonical_name = "random_forest"
    elif canonical_name in ["lr", "logistic regression"]:
        canonical_name = "logistic_regression"

    if X_train is None:
        X_train, _, _, _ = get_train_test_split(random_state=42, stratify=True)

    estimator = pipeline.steps[-1][1]

    if canonical_name == "random_forest":
        explainer = shap.TreeExplainer(estimator)
        explainer_type = "TreeExplainer"
    elif canonical_name == "logistic_regression":
        scaler = pipeline.named_steps.get("scaler")
        if scaler is not None:
            X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=FEATURE_NAMES)
            masker = shap.maskers.Independent(X_train_scaled)
            explainer = shap.LinearExplainer(estimator, masker=masker)
        else:
            explainer = shap.LinearExplainer(estimator, masker=shap.maskers.Independent(X_train))
        explainer_type = "LinearExplainer"
    elif canonical_name in ["svm", "knn"]:
        background = get_training_background(X_train, n_samples=30)
        predict_fn = lambda x: pipeline.predict_proba(pd.DataFrame(x, columns=FEATURE_NAMES))[:, 1]
        explainer = shap.KernelExplainer(predict_fn, background)
        explainer_type = "KernelExplainer"
    else:
        raise ValueError(
            f"Unsupported model architecture '{model_name}'. "
            "Supported models: 'random_forest', 'logistic_regression', 'svm', 'knn'."
        )

    return {
        "model_name": canonical_name,
        "explainer": explainer,
        "explainer_type": explainer_type,
        "pipeline": pipeline
    }


def explain_instance(
    sample: Union[pd.DataFrame, Dict[str, float], List[float]],
    model_name: str = "random_forest",
    pipeline: Optional[Any] = None,
    X_train: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Computes local SHAP explanations for an individual biopsy sample.
    Consistently attributes contributions toward the Malignant class (Class 1).

    Parameters
    ----------
    sample : pd.DataFrame, dict, or list of float
    model_name : str
    pipeline : fitted pipeline (optional)
    X_train : training DataFrame for background (optional)

    Returns
    -------
    dict
        Local explanation details, base value, SHAP values, top positive/negative features,
        and a shap.Explanation object.
    """
    canonical_name = str(model_name).lower().strip()
    if canonical_name in ["rf", "random forest"]:
        canonical_name = "random_forest"
    elif canonical_name in ["lr", "logistic regression"]:
        canonical_name = "logistic_regression"

    if pipeline is None:
        artifact = load_trained_pipeline(canonical_name)
        pipeline = artifact["pipeline"]

    # Align input to DataFrame with canonical columns
    if isinstance(sample, dict):
        df_input = pd.DataFrame([sample])
    elif isinstance(sample, list):
        if len(sample) != len(FEATURE_NAMES):
            raise ValueError(f"Expected {len(FEATURE_NAMES)} features, got {len(sample)}.")
        df_input = pd.DataFrame([sample], columns=FEATURE_NAMES)
    elif isinstance(sample, pd.DataFrame):
        df_input = sample.copy()
    else:
        raise TypeError("sample must be a DataFrame, dict, or list of numbers.")

    missing_cols = [c for c in FEATURE_NAMES if c not in df_input.columns]
    if missing_cols:
        raise ValueError(f"Sample missing required feature columns: {missing_cols[:5]}")

    df_aligned = df_input[FEATURE_NAMES].copy()
    if df_aligned.isnull().any().any():
        nan_cols = df_aligned.columns[df_aligned.isnull().any()].tolist()
        raise ValueError(f"Sample contains NaN/null feature values in columns: {nan_cols}")

    explainer_dict = get_model_explainer(canonical_name, pipeline, X_train)
    explainer = explainer_dict["explainer"]
    explainer_type = explainer_dict["explainer_type"]

    raw_features = df_aligned.iloc[0].values

    # Model prediction and probabilities
    pred_class = int(pipeline.predict(df_aligned)[0])
    pred_label = "Malignant" if pred_class == 1 else "Benign"
    probs = pipeline.predict_proba(df_aligned)[0]
    prob_malignant = float(probs[1])

    # Compute SHAP values consistently for Malignant class (1)
    if explainer_type == "TreeExplainer":
        raw_exp = explainer(df_aligned)
        if len(raw_exp.values.shape) == 3:  # (1, 30, 2)
            shap_vals = raw_exp.values[0, :, 1]
            base_val = float(raw_exp.base_values[0, 1])
        else:
            shap_vals = raw_exp.values[0]
            base_val = float(raw_exp.base_values[0])
        score_name = "Probability of Malignancy (P(Malignant))"
        score_type = "probability"
        unit_label = "Probability (0.0 to 1.0)"
        model_score = prob_malignant

    elif explainer_type == "LinearExplainer":
        scaler = pipeline.named_steps.get("scaler")
        df_scaled = pd.DataFrame(scaler.transform(df_aligned), columns=FEATURE_NAMES) if scaler else df_aligned
        raw_exp = explainer(df_scaled)
        shap_vals = np.array(raw_exp.values[0], dtype=float)
        base_val = float(raw_exp.base_values[0]) if hasattr(raw_exp, "base_values") else 0.0
        score_name = "Log-Odds Margin (Logit toward Malignant)"
        score_type = "log_odds"
        unit_label = "Log-Odds Margin (Logit)"
        # Decision function / margin (evaluated using values to avoid feature names warning)
        estimator = pipeline.steps[-1][1]
        model_score = float(estimator.decision_function(df_scaled.values)[0])

    else:  # KernelExplainer (SVM / KNN)
        raw_vals = explainer.shap_values(df_aligned, nsamples=100, silent=True)
        if isinstance(raw_vals, list):
            shap_vals = np.array(raw_vals[1][0], dtype=float)
        elif len(np.array(raw_vals).shape) == 2:
            shap_vals = np.array(raw_vals[0], dtype=float)
        else:
            shap_vals = np.array(raw_vals, dtype=float).ravel()
        
        base_val = float(explainer.expected_value if not isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0])
        score_name = "Probability of Malignancy (P(Malignant))"
        score_type = "probability"
        unit_label = "Probability (0.0 to 1.0)"
        model_score = prob_malignant

    # Build shap.Explanation object with true clinical units
    shap_explanation = shap.Explanation(
        values=shap_vals,
        base_values=base_val,
        data=raw_features,
        feature_names=FEATURE_NAMES
    )

    # Rank features by contribution
    feature_contributions = []
    for feat_name, s_val, f_val in zip(FEATURE_NAMES, shap_vals, raw_features):
        feature_contributions.append({
            "feature": feat_name,
            "shap_value": float(s_val),
            "feature_value": float(f_val),
            "direction": "Pushes toward Malignant" if s_val > 0 else "Pushes toward Benign",
            "abs_impact": abs(float(s_val))
        })

    # Sort contributions
    feature_contributions.sort(key=lambda x: x["abs_impact"], reverse=True)
    top_positive = [f for f in feature_contributions if f["shap_value"] > 0][:5]
    top_negative = [f for f in feature_contributions if f["shap_value"] < 0][:5]

    return {
        "model_name": canonical_name,
        "explainer_type": explainer_type,
        "prediction_code": pred_class,
        "diagnosis": pred_label,
        "score_name": score_name,
        "score_type": score_type,
        "unit_label": unit_label,
        "model_score": model_score,
        "probability_malignant": prob_malignant,
        "base_value": base_val,
        "feature_contributions": feature_contributions,
        "top_positive_features": top_positive,
        "top_negative_features": top_negative,
        "shap_explanation": shap_explanation
    }


def plot_local_waterfall(
    explanation: shap.Explanation,
    max_display: int = 10,
    unit_label: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Renders and optionally saves a SHAP waterfall plot for an individual patient instance.
    Explicitly labels whether the attribution is in probability or log-odds margin units.
    """
    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    
    unit_text = f" ({unit_label})" if unit_label else ""
    plt.title(f"SHAP Local Feature Attribution toward Malignant{unit_text}", fontsize=11, pad=15)
    plt.xlabel(f"SHAP Contribution Value{unit_text}", fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def compute_global_feature_importance(
    model_name: str = "random_forest",
    pipeline: Optional[Any] = None,
    X_train: Optional[pd.DataFrame] = None,
    n_samples: int = 100
) -> pd.DataFrame:
    """
    Computes global feature importance based on mean absolute SHAP values across evaluation samples.
    Notice: SHAP values quantify model attribution and predictive reliance, NOT biological causation.
    """
    canonical_name = str(model_name).lower().strip()
    if canonical_name in ["rf", "random forest"]:
        canonical_name = "random_forest"
    elif canonical_name in ["lr", "logistic regression"]:
        canonical_name = "logistic_regression"

    if X_train is None:
        X_train, _, _, _ = get_train_test_split(random_state=42, stratify=True)

    if pipeline is None:
        pipeline = load_trained_pipeline(canonical_name)["pipeline"]

    eval_data = shap.sample(X_train, min(n_samples, len(X_train)), random_state=42)

    explainer_dict = get_model_explainer(canonical_name, pipeline, X_train)
    explainer = explainer_dict["explainer"]
    explainer_type = explainer_dict["explainer_type"]

    if explainer_type == "TreeExplainer":
        exp = explainer(eval_data)
        if len(exp.values.shape) == 3:
            shap_matrix = exp.values[:, :, 1]
        else:
            shap_matrix = exp.values
    elif explainer_type == "LinearExplainer":
        scaler = pipeline.named_steps.get("scaler")
        eval_scaled = pd.DataFrame(scaler.transform(eval_data), columns=FEATURE_NAMES) if scaler else eval_data
        exp = explainer(eval_scaled)
        shap_matrix = exp.values
    else:  # KernelExplainer
        raw_vals = explainer.shap_values(eval_data, nsamples=60, silent=True)
        if isinstance(raw_vals, list):
            shap_matrix = np.array(raw_vals[1])
        else:
            shap_matrix = np.array(raw_vals)

    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
    df_importance = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "mean_abs_shap": mean_abs_shap
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

    df_importance["importance_rank"] = range(1, len(df_importance) + 1)
    df_importance["model"] = canonical_name
    return df_importance


def generate_all_global_explanations(
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Computes and generates global SHAP visualizations and machine-readable reports
    for all supported models, saving artifacts to assets/ and reports/.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets_dir = os.path.join(base_dir, "assets")
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(assets_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    X_train, _, _, _ = get_train_test_split(random_state=42, stratify=True)

    print("=" * 60)
    print("      COMPUTING GLOBAL SHAP EXPLANATIONS ACROSS MODELS     ")
    print("=" * 60)

    # 1. Compute Random Forest TreeExplainer summary and beeswarm
    print("-> Computing TreeSHAP for Random Forest (Beeswarm & Summary)...")
    rf_pipe = load_trained_pipeline("random_forest")["pipeline"]
    rf_eval = shap.sample(X_train, 100, random_state=42)
    rf_explainer = shap.TreeExplainer(rf_pipe.steps[-1][1])
    rf_exp = rf_explainer(rf_eval)

    # Class 1 (Malignant) slice
    if len(rf_exp.values.shape) == 3:
        rf_exp_mal = shap.Explanation(
            values=rf_exp.values[:, :, 1],
            base_values=rf_exp.base_values[:, 1],
            data=rf_eval.values,
            feature_names=FEATURE_NAMES
        )
    else:
        rf_exp_mal = rf_exp

    # Beeswarm plot for Random Forest
    fig_summary = plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(rf_exp_mal, max_display=12, show=False)
    plt.title("Random Forest Global TreeSHAP Summary (Malignant Class)", fontsize=12, pad=15)
    plt.tight_layout()
    summary_plot_path = os.path.join(assets_dir, "shap_summary_random_forest.png")
    plt.savefig(summary_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   [OK] Saved: {summary_plot_path}")

    # 2. Compute global importance across all 4 models
    all_importances = {}
    summary_rows = []

    models = ["random_forest", "logistic_regression", "svm", "knn"]
    for m in models:
        print(f"-> Computing global feature importance for {m}...")
        df_imp = compute_global_feature_importance(model_name=m, X_train=X_train, n_samples=80)
        all_importances[m] = df_imp.set_index("feature")["mean_abs_shap"].to_dict()
        summary_rows.append(df_imp)

    # Combine into unified DataFrame
    df_combined = pd.concat(summary_rows, ignore_index=True)
    csv_report_path = os.path.join(reports_dir, "shap_feature_importance.csv")
    df_combined.to_csv(csv_report_path, index=False)
    print(f"   [OK] Saved: {csv_report_path}")

    json_report_path = os.path.join(reports_dir, "shap_feature_importance.json")
    with open(json_report_path, "w") as f:
        json.dump(all_importances, f, indent=4)
    print(f"   [OK] Saved: {json_report_path}")

    # 3. Comparative Bar Chart of Top Biomarkers
    print("-> Plotting comparative biomarker importance chart...")
    fig_comp, ax = plt.subplots(figsize=(12, 7))
    top_rf_feats = df_combined[df_combined["model"] == "random_forest"].head(8)["feature"].tolist()

    bar_width = 0.2
    y_pos = np.arange(len(top_rf_feats))

    for i, m in enumerate(models):
        m_vals = [all_importances[m].get(feat, 0.0) for feat in top_rf_feats]
        # Normalize each model's values to 0-1 range for fair visual comparison
        max_val = max(m_vals) if max(m_vals) > 0 else 1.0
        m_norm = [v / max_val for v in m_vals]
        ax.barh(y_pos + i * bar_width, m_norm, bar_width, label=m.replace("_", " ").title(), alpha=0.85)

    clean_labels = [f.replace("_", " ").title() for f in top_rf_feats]
    ax.set_yticks(y_pos + bar_width * 1.5)
    ax.set_yticklabels(clean_labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Normalized Mean Absolute SHAP Attribution (Relative Reliance)", fontsize=11)
    ax.set_title("Top 8 Morphological Biomarkers Identified Across 4 Classifiers (SHAP)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    comp_plot_path = os.path.join(assets_dir, "shap_feature_importance.png")
    plt.savefig(comp_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   [OK] Saved: {comp_plot_path}")
    print("=" * 60)

    return {
        "summary_plot": summary_plot_path,
        "comparison_plot": comp_plot_path,
        "csv_report": csv_report_path,
        "json_report": json_report_path
    }


if __name__ == "__main__":
    generate_all_global_explanations()
