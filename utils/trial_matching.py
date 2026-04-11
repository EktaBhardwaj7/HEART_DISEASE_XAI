"""
CardioVue AI — Clinical Trial Matching
Find relevant clinical trials based on patient profile
"""

import streamlit as st
import pandas as pd
from typing import Dict, List

# Sample clinical trials database
TRIALS_DB = [
    {
        'id': 'NCT04512345',
        'name': 'ASCVD Risk Reduction with PCSK9 Inhibitors',
        'phase': 'Phase III',
        'status': 'Recruiting',
        'conditions': ['ASCVD', 'High Cholesterol'],
        'location': 'Multiple Sites',
        'contact': 'clinicaltrials@cardiovue.ai',
        'url': 'https://clinicaltrials.gov/ct2/show/NCT04512345'
    },
    {
        'id': 'NCT04567890',
        'name': 'Lifestyle Intervention for High-Risk CVD Patients',
        'phase': 'Phase II',
        'status': 'Recruiting',
        'conditions': ['High Risk CVD', 'Obesity'],
        'location': 'Remote / Virtual',
        'contact': 'research@cardiovue.ai',
        'url': 'https://clinicaltrials.gov/ct2/show/NCT04567890'
    },
    {
        'id': 'NCT04513579',
        'name': 'Digital Health Intervention for Medication Adherence',
        'phase': 'Phase IV',
        'status': 'Enrolling',
        'conditions': ['Hypertension', 'Heart Failure'],
        'location': 'Multiple Sites',
        'contact': 'digitalhealth@cardiovue.ai',
        'url': 'https://clinicaltrials.gov/ct2/show/NCT04513579'
    }
]

class TrialMatcher:
    """Match patients to clinical trials"""
    
    def __init__(self):
        self.trials = TRIALS_DB
    
    def match_patient(self, patient_data: Dict, risk_score: float) -> List[Dict]:
        """Find eligible trials for patient"""
        matches = []
        
        for trial in self.trials:
            if trial['status'] not in ['Recruiting', 'Enrolling']:
                continue
            
            # Simple matching logic
            score = 0
            reasons = []
            
            # Risk-based matching
            if risk_score > 20 and 'High Risk' in str(trial['conditions']):
                score += 30
                reasons.append("High risk profile matches trial criteria")
            
            # Condition matching
            patient_conditions = patient_data.get('conditions', [])
            for condition in patient_conditions:
                if condition in str(trial['conditions']):
                    score += 25
                    reasons.append(f"Condition '{condition}' matches")
            
            if score > 0:
                matches.append({
                    'trial': trial,
                    'match_score': min(score, 100),
                    'reasons': reasons,
                    'eligibility': 'Eligible' if score >= 50 else 'Potential Match'
                })
        
        return sorted(matches, key=lambda x: x['match_score'], reverse=True)

def render_trial_matching(patient_data: Dict, risk_score: float):
    """Render clinical trial matching UI"""
    st.markdown("### 🔬 Clinical Trial Matching")
    st.markdown("*Find research opportunities that match your profile*")
    
    matcher = TrialMatcher()
    matches = matcher.match_patient(patient_data, risk_score)
    
    if not matches:
        st.info("No matching trials found at this time. New trials are added regularly.")
        return
    
    for match in matches[:3]:
        trial = match['trial']
        score = match['match_score']
        
        score_color = "#22c55e" if score >= 70 else "#f59e0b" if score >= 40 else "#8aa0b5"
        
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(20,184,166,0.2); 
                    border-radius:16px; padding:1rem; margin-bottom:1rem;">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div>
                    <div style="font-weight:600;">{trial['name']}</div>
                    <div style="font-size:0.75rem; color:#8aa0b5;">{trial['phase']} · {trial['status']}</div>
                    <div style="font-size:0.7rem; margin-top:0.25rem;">📍 {trial['location']}</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:1.5rem; font-weight:800; color:{score_color};">{score}%</div>
                    <div style="font-size:0.65rem;">Match</div>
                </div>
            </div>
            <div style="margin-top:0.75rem;">
                <span class="cv-badge cv-badge-teal">{match['eligibility']}</span>
            </div>
            <div style="margin-top:0.5rem;">
                <a href="{trial['url']}" target="_blank" style="color:#14b8a6; text-decoration:none;">View Details →</a>
            </div>
        </div>
        """, unsafe_allow_html=True)