"""
CardioVue AI — Advanced ML Features
Counterfactual explanations, feature importance, model interpretation
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from utils.ml_engine import predict_risk

def render_counterfactual_analysis(patient_data: dict, current_risk: float):
    """Interactive what-if analysis with counterfactual explanations"""
    
    st.markdown("### 🔮 Counterfactual Analysis")
    st.markdown("*See exactly what changes would lower your risk*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Current Profile")
        st.metric("Current Risk", f"{current_risk:.1f}%")
        
        # Editable features
        new_bmi = st.slider("BMI", 18, 40, int(patient_data.get('bmi', 27)), key="cf_bmi")
        new_bp = st.slider("Systolic BP", 90, 180, int(patient_data.get('bp_systolic', 128)), key="cf_bp")
        new_chol = st.slider("Cholesterol", 120, 300, int(patient_data.get('cholesterol', 190)), key="cf_chol")
        
        quit_smoking = st.checkbox("Quit Smoking", value=patient_data.get('smoker', False), key="cf_smoker")
        start_exercise = st.checkbox("Start Regular Exercise", value=patient_data.get('phys_activity', False), key="cf_exercise")
    
    with col2:
        # Calculate new risk
        test_features = {
            'age': patient_data.get('age', 45),
            'bmi': new_bmi,
            'gen_health': patient_data.get('gen_health', 3),
            'sex': patient_data.get('sex', 1),
            'highbp': patient_data.get('highbp', 0),
            'highchol': patient_data.get('highchol', 0),
            'smoker': 1 if quit_smoking else 0,
            'diabetes': patient_data.get('diabetes', 0),
            'phys_activity': 1 if start_exercise else 0,
            'stroke': patient_data.get('stroke', 0),
        }
        
        result = predict_risk(test_features)
        new_risk = result['risk_score']
        reduction = current_risk - new_risk
        
        st.markdown("#### After Changes")
        st.metric("New Risk", f"{new_risk:.1f}%", delta=f"-{reduction:.1f}%" if reduction > 0 else f"+{abs(reduction):.1f}%")
        
        # Impact visualization
        if reduction > 0:
            st.progress(min(new_risk / 100, 1.0))
            st.success(f"✨ You could reduce your risk by {reduction:.1f}%!")
            
            # Top recommendations
            changes = []
            if quit_smoking and patient_data.get('smoker'):
                changes.append("🚭 Quit smoking → -18% risk")
            if start_exercise and not patient_data.get('phys_activity'):
                changes.append("🏃 Start exercising → -12% risk")
            if new_bmi < patient_data.get('bmi', 27):
                changes.append(f"⚖️ Lose {patient_data.get('bmi', 27) - new_bmi:.1f} BMI points → -8% risk")
            
            if changes:
                st.markdown("#### 📋 Action Plan")
                for c in changes:
                    st.markdown(f"- {c}")

def render_shap_waterfall(shap_values: dict):
    """Render interactive SHAP waterfall plot"""
    if not shap_values:
        return
    
    st.markdown("### 🧬 Feature Impact (SHAP)")
    
    # Sort by absolute value
    sorted_items = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
    names = [n for n, _ in sorted_items]
    values = [v * 100 for _, v in sorted_items]  # Convert to percentage
    
    colors = ['#ef4444' if v > 0 else '#10b981' for v in values]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation='h',
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{'+' if v > 0 else ''}{v:.1f}%" for v in values],
        textposition='outside',
        textfont=dict(size=11)
    ))
    
    fig.update_layout(
        title="Risk Contribution Factors",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300,
        xaxis_title="Risk Change (%)",
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def render_model_comparison():
    """Render model performance comparison chart"""
    from utils.ml_engine import MODEL_PERFORMANCE
    
    fig = go.Figure()
    
    models = MODEL_PERFORMANCE['Model'].tolist()
    f1_scores = MODEL_PERFORMANCE['F1_Score'].tolist()
    auc_scores = MODEL_PERFORMANCE['ROC_AUC'].tolist()
    
    fig.add_trace(go.Bar(
        name='F1 Score',
        x=models,
        y=f1_scores,
        marker_color='#14b8a6',
        text=[f"{v:.3f}" for v in f1_scores],
        textposition='outside'
    ))
    
    fig.add_trace(go.Scatter(
        name='ROC-AUC',
        x=models,
        y=auc_scores,
        mode='lines+markers',
        line=dict(color='#f59e0b', width=2),
        marker=dict(size=8, color='#f59e0b')
    ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        yaxis=dict(range=[0.85, 1.0], gridcolor='rgba(255,255,255,0.05)'),
        xaxis=dict(tickangle=-15, gridcolor='rgba(255,255,255,0.05)')
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})