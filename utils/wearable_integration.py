"""
CardioVue AI — Wearable Device Integration
Connect Apple Health, Fitbit, Google Fit, Garmin
"""

import streamlit as st
from datetime import datetime, timedelta
import random

class WearableConnector:
    """Unified wearable device connector"""
    
    SUPPORTED_DEVICES = {
        'apple_health': {'name': '🍎 Apple Health', 'color': '#000000'},
        'fitbit': {'name': '⌚ Fitbit', 'color': '#00B0B9'},
        'google_fit': {'name': '🔵 Google Fit', 'color': '#4285F4'},
        'garmin': {'name': '📊 Garmin', 'color': '#0F6B3A'},
        'samsung_health': {'name': '📱 Samsung Health', 'color': '#1428A0'},
    }
    
    def __init__(self):
        self.connected_devices = {}
    
    def connect(self, device_type: str) -> bool:
        """Simulate device connection"""
        if device_type in self.SUPPORTED_DEVICES:
            self.connected_devices[device_type] = {
                'connected_at': datetime.now(),
                'last_sync': datetime.now()
            }
            return True
        return False
    
    def disconnect(self, device_type: str):
        """Disconnect device"""
        if device_type in self.connected_devices:
            del self.connected_devices[device_type]
    
    def get_today_data(self) -> dict:
        """Get aggregated today's data from all connected devices"""
        data = {
            'steps': random.randint(5000, 12000),
            'heart_rate': random.randint(65, 85),
            'sleep_hours': round(random.uniform(6, 8.5), 1),
            'active_calories': random.randint(300, 800),
            'distance_km': round(random.uniform(3, 10), 1),
            'resting_hr': random.randint(60, 75),
            'vo2_max': random.randint(30, 45)
        }
        return data
    
    def get_heart_rate_history(self, days: int = 7) -> list:
        """Get heart rate history"""
        history = []
        for i in range(days):
            date = (datetime.now() - timedelta(days=days-1-i)).strftime('%a')
            history.append({
                'date': date,
                'avg_hr': random.randint(68, 82),
                'min_hr': random.randint(55, 65),
                'max_hr': random.randint(110, 140)
            })
        return history

def render_wearable_dashboard():
    """Render wearable device dashboard"""
    st.markdown("### ⌚ Connected Devices")
    
    if 'wearable_connected' not in st.session_state:
        st.session_state.wearable_connected = {}
    
    connector = WearableConnector()
    
    # Device connection buttons
    cols = st.columns(len(WearableConnector.SUPPORTED_DEVICES))
    for col, (device_id, info) in zip(cols, WearableConnector.SUPPORTED_DEVICES.items()):
        with col:
            is_connected = st.session_state.wearable_connected.get(device_id, False)
            if st.button(f"{info['name']}", key=f"wear_{device_id}", use_container_width=True):
                if not is_connected:
                    if connector.connect(device_id):
                        st.session_state.wearable_connected[device_id] = True
                        st.rerun()
                else:
                    st.session_state.wearable_connected[device_id] = False
                    st.rerun()
            
            if is_connected:
                st.caption("✅ Connected")
    
    # Show data if any device connected
    if any(st.session_state.wearable_connected.values()):
        st.markdown("### 📊 Today's Activity")
        
        data = connector.get_today_data()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("👣 Steps", f"{data['steps']:,}")
        with col2:
            st.metric("💓 Heart Rate", f"{data['heart_rate']} bpm")
        with col3:
            st.metric("😴 Sleep", f"{data['sleep_hours']} hrs")
        with col4:
            st.metric("🔥 Calories", f"{data['active_calories']}")
        
        # Heart rate trend
        hr_history = connector.get_heart_rate_history(7)
        
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[h['date'] for h in hr_history],
            y=[h['avg_hr'] for h in hr_history],
            mode='lines+markers',
            name='Avg HR',
            line=dict(color='#14b8a6', width=2),
            marker=dict(size=6)
        ))
        fig.update_layout(
            title="Heart Rate Trend (7 Days)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=250,
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})