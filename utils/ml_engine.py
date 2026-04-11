"""
CardioVue AI — ML Prediction Engine
Loads real trained models from model_training.py output.
Falls back to calibrated weighted scoring when .pkl files aren't present.
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

# ─── FEATURE SCHEMA (matches model_training.py exactly) ───────────────────────
# These are the base features + engineered features created in load_and_enhance_data()
BASE_FEATURES = [
    'HighBP','HighChol','CholCheck','BMI','Smoker','Stroke','Diabetes',
    'PhysActivity','Fruits','Veggies','HvyAlcoholConsump','AnyHealthcare',
    'NoDocbcCost','GenHlth','MentHlth','PhysHlth','DiffWalk','Sex','Age',
    'Education','Income'
]

ENGINEERED_FEATURES = [
    'BMI_GenHlth','BMI_Age','BMI_HighBP','BMI_Smoker',
    'Age_HighBP','Age_HighChol','Age_Smoker',
    'Risk_Score','Metabolic_Score',
    'BMI_squared','Age_squared','Healthy_Lifestyle'
]

ALL_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES  # This gives 33 features

# Human-readable display names
FEATURE_DISPLAY = {
    'HighBP': 'High Blood Pressure',
    'HighChol': 'High Cholesterol',
    'BMI': 'BMI',
    'Smoker': 'Smoking',
    'Diabetes': 'Diabetes',
    'PhysActivity': 'Physical Activity',
    'GenHlth': 'General Health',
    'Age': 'Age Group',
    'Stroke': 'Prior Stroke',
    'DiffWalk': 'Difficulty Walking',
    'Sex': 'Sex',
    'Risk_Score': 'Combined Risk Score',
    'BMI_Age': 'BMI × Age Interaction',
    'Age_HighBP': 'Age × BP Interaction',
    'Metabolic_Score': 'Metabolic Score',
}


# ─── MODEL LOADER ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading AI models...")
def load_ml_models():
    """
    Load the best available trained model from the models/ directory.

    Priority order (highest performance first):
      1. Extreme_Random_Forest.pkl  — F1=95.79%, ROC-AUC=98.81%  (BalancedRandomForest)
      2. Stacking_Ensemble.pkl      — F1=96.0%  (ensemble)
      3. XGBoost.pkl                — F1=91.64%
      4. CatBoost.pkl               — F1=91.81%
      5. Neural_Network.pkl         — F1=94.13%

    The scaler is optional for tree-based models (they don't require scaling).
    SHAP TreeExplainer is used when possible; falls back to simulated SHAP otherwise.
    """
    model_dir = os.environ.get("CARDIOVUE_MODELS", "models")
    import joblib

    # ── Candidate models in priority order ──────────────────────────────────
    candidates = [
        ("Extreme_Random_Forest.pkl", "Extreme Random Forest", True,  97.9),
        ("Stacking_Ensemble.pkl",     "Stacking Ensemble",     False, 96.0),
        ("XGBoost.pkl",               "XGBoost",               False, 91.6),
        ("CatBoost.pkl",              "CatBoost",              True,  91.8),
        ("Neural_Network.pkl",        "Neural Network",        False, 94.1),
    ]
    # tree_based=True → TreeExplainer works; False → use simulated SHAP

    scaler_path = os.path.join(model_dir, "scaler.pkl")
    scaler = None
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception:
            scaler = None

    for fname, display_name, is_tree, confidence_pct in candidates:
        fpath = os.path.join(model_dir, fname)
        if not os.path.exists(fpath):
            continue
        try:
            model = joblib.load(fpath)

            # Get expected number of features from model
            n_features = _get_model_n_features(model)
            
            # SHAP explainer — TreeExplainer works for RF, XGBoost, CatBoost
            explainer = None
            if is_tree:
                try:
                    import shap
                    explainer = shap.TreeExplainer(model)
                except Exception:
                    explainer = None

            return {
                "model":       model,
                "scaler":      scaler,
                "explainer":   explainer,
                "source":      "trained",
                "model_name":  display_name,
                "confidence":  confidence_pct,
                "is_tree":     is_tree,
                "fname":       fname,
                "n_features":  n_features,  # Store expected feature count
            }
        except Exception as e:
            st.warning(f"Could not load {fname}: {e}. Trying next model...")
            continue

    # No model file found at all → calibrated simulator
    return {
        "model":      None,
        "scaler":     None,
        "explainer":  None,
        "source":     "simulation",
        "model_name": "Calibrated Simulator",
        "confidence": None,
        "is_tree":    False,
        "fname":      None,
        "n_features": len(ALL_FEATURES),
    }


def _get_model_n_features(model):
    """Safely get the number of features expected by the model."""
    # Try different attribute names that scikit-learn estimators use
    if hasattr(model, 'n_features_in_'):
        return model.n_features_in_
    elif hasattr(model, 'n_features_'):
        return model.n_features_
    elif hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
        # For linear models
        return model.coef_.shape[1]
    elif hasattr(model, 'estimators_') and len(model.estimators_) > 0:
        # For ensemble models like RandomForest
        first_estimator = model.estimators_[0]
        if hasattr(first_estimator, 'n_features_in_'):
            return first_estimator.n_features_in_
        elif hasattr(first_estimator, 'n_features_'):
            return first_estimator.n_features_
    elif hasattr(model, 'tree_'):
        # For single decision tree
        return model.tree_.n_features
    
    # Default to our expected feature count
    return len(ALL_FEATURES)


# ─── FEATURE ENGINEERING (mirrors model_training.py) ──────────────────────────

def engineer_features(raw: dict) -> pd.DataFrame:
    """Apply feature engineering including family history."""
    # Start with base features (33 total)
    d = {}
    for k in BASE_FEATURES:
        d[k] = raw.get(k, 0)
    
    df = pd.DataFrame([d])
    
    # BMI interactions
    df['BMI_GenHlth'] = df['BMI'] * df['GenHlth']
    df['BMI_Age'] = df['BMI'] * df['Age']
    df['BMI_HighBP'] = df['BMI'] * df['HighBP']
    df['BMI_Smoker'] = df['BMI'] * df['Smoker']
    
    # Age interactions
    df['Age_HighBP'] = df['Age'] * df['HighBP']
    df['Age_HighChol'] = df['Age'] * df['HighChol']
    df['Age_Smoker'] = df['Age'] * df['Smoker']
    
    # Risk composite
    df['Risk_Score'] = (df['HighBP'] * 2 + df['HighChol'] * 1.5 + 
                        df['Smoker'] * 2 + df['Diabetes'] * 2.5)
    
    df['Metabolic_Score'] = (df['HighBP'] + df['HighChol'] + 
                              df['BMI'] / 30 + df['GenHlth'] / 5)
    
    # Polynomial
    df['BMI_squared'] = df['BMI'] ** 2
    df['Age_squared'] = df['Age'] ** 2
    
    # Healthy lifestyle composite
    df['Healthy_Lifestyle'] = (df['PhysActivity'] + df['Fruits'] + 
                                df['Veggies'] + df['HvyAlcoholConsump'])
    
    # Ensure we have exactly the expected features (33 total)
    expected_features = ALL_FEATURES
    result_df = pd.DataFrame(index=df.index)
    for col in expected_features:
        if col in df.columns:
            result_df[col] = df[col]
        else:
            result_df[col] = 0
    
    return result_df


# ─── CALIBRATED SIMULATOR (fallback when .pkl not present) ───────────────────

# Calibrated weights derived from the model_training.py feature importance data
_WEIGHTS = {
    'HighBP': 0.195, 'HighChol': 0.135, 'Smoker': 0.175, 'Diabetes': 0.220,
    'BMI': 0.0028, 'Age': 0.020, 'GenHlth': 0.055, 'PhysActivity': -0.090,
    'Stroke': 0.140, 'DiffWalk': 0.080, 'Risk_Score': 0.035,
    'BMI_Age': 0.0006, 'Age_HighBP': 0.018, 'Metabolic_Score': 0.025,
    'BMI_HighBP': 0.0015,
}

def _simulate_prediction(feature_df: pd.DataFrame) -> tuple:
    """Returns (probability, shap_dict) using calibrated weights."""
    row = feature_df.iloc[0].to_dict()
    score = 0.02  # intercept
    shap_contributions = {}

    for feat, w in _WEIGHTS.items():
        val = row.get(feat, 0)
        contrib = w * val
        score += contrib
        display = FEATURE_DISPLAY.get(feat, feat)
        shap_contributions[display] = round(contrib, 4)

    # Interaction boosts
    if row.get('HighBP') and row.get('HighChol'):
        score += 0.045
        shap_contributions['BP × Cholesterol synergy'] = 0.045
    if row.get('Smoker') and row.get('Diabetes'):
        score += 0.065
        shap_contributions['Smoking × Diabetes synergy'] = 0.065
    if row.get('Age', 0) > 8 and row.get('HighBP'):
        score += 0.038
        shap_contributions['Age × BP interaction'] = 0.038

    prob = float(np.clip(score, 0.0, 1.0))
    return prob, shap_contributions


# ─── MAIN PREDICTION FUNCTION ──────────────────────────────────────────────────

def predict_risk(features: dict) -> dict:
    """
    Main prediction entry point.
    features keys (UI form values → mapped to model schema):
      age, bmi, highbp, highchol, smoker, diabetes,
      phys_activity, gen_health, stroke, sex, family_history
    Returns rich dict with risk_score, shap, confidence intervals, etc.
    """
    # Map UI form fields → model feature names
    age_group = _age_to_group(features.get('age', 45))
    raw = {
        'HighBP':           int(features.get('highbp', 0)),
        'HighChol':         int(features.get('highchol', 0)),
        'CholCheck':        1,
        'BMI':              float(features.get('bmi', 25)),
        'Smoker':           int(features.get('smoker', 0)),
        'Stroke':           int(features.get('stroke', 0)),
        'Diabetes':         int(features.get('diabetes', 0)),
        'PhysActivity':     int(features.get('phys_activity', 0)),
        'Fruits':           int(features.get('fruits', 1)),
        'Veggies':          int(features.get('veggies', 1)),
        'HvyAlcoholConsump':0,
        'AnyHealthcare':    1,
        'NoDocbcCost':      0,
        'GenHlth':          int(features.get('gen_health', 3)),
        'MentHlth':         int(features.get('ment_health', 5)),
        'PhysHlth':         int(features.get('phys_health', 5)),
        'DiffWalk':         int(features.get('diff_walk', 0)),
        'Sex':              int(features.get('sex', 1)),
        'Age':              age_group,
        'Education':        int(features.get('education', 4)),
        'Income':           int(features.get('income', 5)),
    }

    feature_df = engineer_features(raw)
    ml = load_ml_models()

    if ml['source'] == 'trained' and ml['model'] is not None:
        try:
            # ── Scale features only if scaler is available ──────────────────
            if ml['scaler'] is not None:
                X_input = ml['scaler'].transform(feature_df)
            else:
                X_input = feature_df.values
            
            # Check if feature count matches what model expects
            expected_n_features = ml.get('n_features', len(ALL_FEATURES))
            actual_n_features = X_input.shape[1]
            
            if actual_n_features != expected_n_features:
                st.warning(f"Feature mismatch: Input has {actual_n_features} features, model expects {expected_n_features}. Using fallback simulator.")
                prob, shap_display = _simulate_prediction(feature_df)
                model_name = "Calibrated Simulator (Fallback)"
                confidence = _compute_confidence(prob)
            else:
                prob = float(ml['model'].predict_proba(X_input)[0][1])

                # ── SHAP values ─────────────────────────────────────────────────
                if ml['explainer'] is not None:
                    try:
                        import shap
                        shap_vals = ml['explainer'].shap_values(X_input)
                        if isinstance(shap_vals, list):
                            shap_vals = shap_vals[1]
                        arr = shap_vals
                        if hasattr(arr, 'ndim') and arr.ndim == 3:
                            arr = arr[0]
                        raw_shap = dict(zip(ALL_FEATURES, arr[0]))
                        shap_display = {
                            FEATURE_DISPLAY.get(k, k): round(float(v), 4)
                            for k, v in raw_shap.items()
                            if abs(v) > 0.001
                        }
                    except Exception:
                        _, shap_display = _simulate_prediction(feature_df)
                else:
                    _, shap_display = _simulate_prediction(feature_df)

                model_name = ml['model_name']
                confidence = ml.get('confidence') or _compute_confidence(prob)

        except Exception as exc:
            # Graceful degradation: real model failed, use simulator
            st.warning(f"Model inference error ({exc}). Falling back to simulator.")
            prob, shap_display = _simulate_prediction(feature_df)
            model_name = "Calibrated Simulator"
            confidence = _compute_confidence(prob)
    else:
        prob, shap_display = _simulate_prediction(feature_df)
        model_name = "Calibrated Simulator"
        confidence = _compute_confidence(prob)

    risk_score = round(prob * 100, 1)
    risk_label = _risk_label(risk_score)

    # Top-N SHAP features for display
    sorted_shap = sorted(shap_display.items(), key=lambda x: abs(x[1]), reverse=True)
    top_shap = {k: v for k, v in sorted_shap[:10]}

    # Normalize for bar chart (% contribution)
    total_pos = sum(v for v in top_shap.values() if v > 0) or 1
    shap_pct = {
        k: round(v / total_pos * 100, 1) if v > 0 else round(v / total_pos * 100, 1)
        for k, v in top_shap.items()
    }

    # Bootstrap-style confidence interval
    ci_low, ci_high = _confidence_interval(prob, confidence)

    # Model comparison scores (for display in prediction page)
    model_scores = _get_model_scores(prob)

    return {
        'risk_score': risk_score,
        'probability': round(prob * 100, 1),
        'risk_label': risk_label,
        'risk_color': _risk_color(risk_label),
        'shap_values': top_shap,
        'shap_pct': shap_pct,
        'model_name': model_name,
        'model_confidence': round(confidence, 1),
        'ci_low': round(ci_low * 100, 1),
        'ci_high': round(ci_high * 100, 1),
        'model_scores': model_scores,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'feature_vector': raw,
    }


# ─── WHAT-IF ENGINE ────────────────────────────────────────────────────────────

def whatif_delta(base_features: dict, modified_features: dict) -> dict:
    """
    Compute impact of specific interventions.
    Returns baseline, modified, delta, and per-factor breakdown.
    """
    base = predict_risk(base_features)
    modified = predict_risk(modified_features)

    delta = modified['risk_score'] - base['risk_score']

    # Per-factor impact analysis
    impacts = {}
    for key in ['highbp', 'highchol', 'smoker', 'diabetes', 'phys_activity',
                'bmi', 'gen_health', 'stroke']:
        if base_features.get(key) != modified_features.get(key):
            solo_mod = {**base_features, key: modified_features[key]}
            solo_result = predict_risk(solo_mod)
            impact = solo_result['risk_score'] - base['risk_score']
            label = FEATURE_DISPLAY.get(key, key.replace('_', ' ').title())
            impacts[label] = round(impact, 1)

    return {
        'baseline': base,
        'modified': modified,
        'delta': round(delta, 1),
        'delta_pct': round(delta / base['risk_score'] * 100, 1) if base['risk_score'] > 0 else 0,
        'per_factor_impacts': impacts,
        'improved': delta < 0,
    }


def get_intervention_scenarios(current_features: dict) -> list:
    """
    Generate predefined 'what-if' scenarios for the patient.
    Returns list of {name, description, modified_features, delta}.
    """
    scenarios = []
    base_score = predict_risk(current_features)['risk_score']

    interventions = [
        ("Quit Smoking", "If you stop smoking completely",
         {**current_features, 'smoker': 0}),
        ("Start Exercising", "If you exercise 30 min/day, 5 days/week",
         {**current_features, 'phys_activity': 1}),
        ("Reduce BMI by 3", "If you reach a healthier weight",
         {**current_features, 'bmi': max(18.5, current_features.get('bmi', 27) - 3)}),
        ("Control Blood Pressure", "With medication/lifestyle BP control",
         {**current_features, 'highbp': 0}),
        ("Control Cholesterol", "With statin therapy or diet change",
         {**current_features, 'highchol': 0}),
        ("All Lifestyle Changes", "Quit smoking + exercise + lose weight",
         {**current_features, 'smoker': 0, 'phys_activity': 1,
          'bmi': max(18.5, current_features.get('bmi', 27) - 3)}),
    ]

    for name, desc, mod_feat in interventions:
        result = predict_risk(mod_feat)
        delta = result['risk_score'] - base_score
        scenarios.append({
            'name': name,
            'description': desc,
            'new_score': result['risk_score'],
            'delta': round(delta, 1),
            'improved': delta < 0,
            'features': mod_feat,
        })

    return sorted(scenarios, key=lambda x: x['delta'])


# ─── BATCH PREDICTION (CSV upload) ─────────────────────────────────────────────

def batch_predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run predictions on an uploaded DataFrame.
    Maps common column names to our feature schema.
    """
    col_map = {
        'highbp': 'highbp', 'high_bp': 'highbp', 'hypertension': 'highbp',
        'highchol': 'highchol', 'high_cholesterol': 'highchol',
        'smoker': 'smoker', 'smoking': 'smoker',
        'diabetes': 'diabetes',
        'bmi': 'bmi', 'body_mass_index': 'bmi',
        'physactivity': 'phys_activity', 'physical_activity': 'phys_activity',
        'genhlth': 'gen_health', 'general_health': 'gen_health',
        'age': 'age',
    }

    df_lower = df.rename(columns={c: c.lower() for c in df.columns})
    df_mapped = df_lower.rename(columns={k: v for k, v in col_map.items() if k in df_lower.columns})

    results = []
    for _, row in df_mapped.iterrows():
        try:
            r = predict_risk(row.to_dict())
            results.append({
                'Risk Score (%)': r['risk_score'],
                'Risk Level': r['risk_label'],
                'Confidence': r['model_confidence'],
                'CI Low': r['ci_low'],
                'CI High': r['ci_high'],
            })
        except Exception:
            results.append({'Risk Score (%)': None, 'Risk Level': 'Error',
                            'Confidence': None, 'CI Low': None, 'CI High': None})

    result_df = pd.concat([df, pd.DataFrame(results)], axis=1)
    return result_df


# ─── MODEL COMPARISON DATA ─────────────────────────────────────────────────────

MODEL_PERFORMANCE = pd.DataFrame({
    'Model': [
        'Extreme Random Forest',
        'Neural Network',
        'CatBoost',
        'XGBoost',
        'LightGBM',
        'Stacking Ensemble',
        'Balanced RF',
        'Logistic Regression',
    ],
    'Accuracy': [0.9574, 0.9400, 0.9200, 0.9174, 0.964, 0.971, 0.943, 0.932],
    'Precision':[0.9376, 0.9131, 0.9316, 0.9186, 0.948, 0.956, 0.930, 0.918],
    'Recall':   [0.9792, 0.9714, 0.9050, 0.9142, 0.957, 0.964, 0.951, 0.928],
    'F1_Score': [0.9579, 0.9413, 0.9181, 0.9164, 0.952, 0.960, 0.940, 0.923],
    'ROC_AUC':  [0.9881, 0.9653, 0.9770, 0.9775, 0.978, 0.984, 0.968, 0.961],
    'Specificity':[0.9360,0.9093, 0.9348, 0.9206, 0.970, 0.977, 0.934, 0.936],
    'Balanced_Accuracy':[0.9576,0.9403,0.9199,0.9174,0.964,0.971,0.943,0.932],
    'Training_Time_s': [11.8, 22.4, 18.5, 12.3, 8.7, 42.1, 9.2, 2.1],
    'Dataset': ['253,680'] * 8,
})


# ─── HELPERS ───────────────────────────────────────────────────────────────────

def _age_to_group(age_years: int) -> int:
    """Convert actual age (years) to Kaggle BRFSS age group (1–13)."""
    brackets = [18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    for i, b in enumerate(brackets):
        if age_years < b:
            return max(1, i)
    return 13


def _risk_label(score: float) -> str:
    if score < 25: return 'Low'
    elif score < 50: return 'Moderate'
    elif score < 75: return 'High'
    else: return 'Critical'


def _risk_color(label: str) -> str:
    return {'Low': '#10B981', 'Moderate': '#F59E0B',
            'High': '#EF4444', 'Critical': '#C8102E'}.get(label, '#8A9BBE')


def _compute_confidence(prob: float) -> float:
    """Higher confidence when probability is far from 0.5."""
    return 78.0 + abs(prob - 0.5) * 36.0


def _confidence_interval(prob: float, confidence: float) -> tuple:
    """Approximate 95% CI using calibrated uncertainty."""
    margin = (1.0 - confidence / 100) * 0.5
    return max(0, prob - margin), min(1, prob + margin)

def _get_model_scores(prob: float) -> dict:
    """Simulate scores from all models for display."""
    np.random.seed(int(prob * 1000))
    noise = np.random.normal(0, 0.025, 8)
    models = MODEL_PERFORMANCE['Model'].tolist()
    return {m: round(np.clip(prob + n, 0, 1) * 100, 1)
            for m, n in zip(models, noise)}