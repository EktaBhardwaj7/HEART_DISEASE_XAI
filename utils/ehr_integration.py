"""
CardioVue AI — EHR Integration (FHIR)
Working mock implementation for demo purposes
"""

import streamlit as st
import json
from datetime import datetime
from typing import Dict, List, Optional

class FHIRConnector:
    """FHIR API connector for EHR integration"""
    
    def __init__(self):
        self.connected = False
        self.patient_data = None
    
    def connect(self, patient_id: str = None) -> bool:
        """Simulate EHR connection"""
        if patient_id:
            self.connected = True
            self.patient_data = {
                'id': patient_id,
                'name': 'Demo Patient',
                'gender': 'female',
                'birth_date': '1980-01-01',
                'conditions': ['Hypertension', 'Hyperlipidemia'],
                'medications': ['Lisinopril', 'Atorvastatin']
            }
            return True
        return False
    
    def get_patient_demographics(self) -> Dict:
        """Get patient demographics"""
        if self.connected:
            return self.patient_data
        return {}
    
    def get_latest_vitals(self) -> Dict:
        """Get latest vital signs"""
        return {
            'bp_systolic': 128,
            'bp_diastolic': 82,
            'heart_rate': 72,
            'temperature': 98.6,
            'oxygen_saturation': 98
        }
    
    def get_latest_labs(self) -> Dict:
        """Get latest lab results"""
        return {
            'hdl': 52,
            'ldl': 118,
            'triglycerides': 145,
            'glucose': 96,
            'hba1c': 5.4,
            'creatinine': 0.92
        }
    
    def push_observation(self, code: str, value: float, unit: str) -> bool:
        """Push observation to EHR"""
        st.success(f"✓ Pushed to EHR: {code} = {value} {unit}")
        return True

def render_ehr_panel():
    """Render EHR integration panel in sidebar"""
    with st.expander("🏥 EHR Integration", expanded=False):
        st.markdown("Connect to Electronic Health Record")
        
        if 'ehr_connected' not in st.session_state:
            st.session_state.ehr_connected = False
        
        if not st.session_state.ehr_connected:
            patient_id = st.text_input("Patient ID", placeholder="Enter MRN or Patient ID")
            if st.button("🔗 Connect to EHR", use_container_width=True):
                connector = FHIRConnector()
                if connector.connect(patient_id or "DEMO001"):
                    st.session_state.ehr_connected = True
                    st.session_state.ehr_data = connector.get_patient_demographics()
                    st.success("✅ Connected to EHR")
                    st.rerun()
        else:
            st.success("🔗 Connected to EHR")
            if st.button("Disconnect", use_container_width=True):
                st.session_state.ehr_connected = False
                st.rerun()