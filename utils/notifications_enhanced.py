"""
CardioVue AI — Enhanced Notification System
Smart alerts, reminders, and health insights
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import List, Dict

class NotificationEngine:
    """Smart notification engine"""
    
    def __init__(self, username: str):
        self.username = username
    
    def check_risk_alerts(self, current_risk: float, previous_risk: float = None) -> List[Dict]:
        """Generate risk-based alerts"""
        alerts = []
        
        if current_risk >= 75:
            alerts.append({
                'type': 'critical',
                'title': '🚨 Critical Risk Level',
                'message': 'Your cardiovascular risk is critical. Please consult a cardiologist immediately.',
                'action': 'Schedule Emergency Visit'
            })
        elif current_risk >= 50:
            alerts.append({
                'type': 'warning',
                'title': '⚠️ High Risk Alert',
                'message': 'Your risk level is high. Schedule a follow-up appointment soon.',
                'action': 'Book Appointment'
            })
        
        if previous_risk and (current_risk - previous_risk) > 10:
            alerts.append({
                'type': 'warning',
                'title': '📈 Risk Increasing',
                'message': f'Your risk increased by {current_risk - previous_risk:.1f}%. Review your lifestyle.',
                'action': 'View Recommendations'
            })
        elif previous_risk and (previous_risk - current_risk) > 5:
            alerts.append({
                'type': 'success',
                'title': '🎉 Risk Decreasing!',
                'message': f'Your risk dropped by {previous_risk - current_risk:.1f}%. Keep it up!',
                'action': 'Share Progress'
            })
        
        return alerts
    
    def check_medication_reminders(self, medications: List[Dict]) -> List[Dict]:
        """Generate medication reminders"""
        reminders = []
        
        for med in medications:
            last_taken = med.get('last_taken')
            frequency = med.get('frequency', 'daily')
            
            if last_taken:
                last_date = datetime.strptime(last_taken, '%Y-%m-%d')
                days_since = (datetime.now() - last_date).days
                
                if frequency == 'daily' and days_since >= 1:
                    reminders.append({
                        'type': 'reminder',
                        'title': f'💊 {med["name"]} Reminder',
                        'message': f'Time to take your {med["name"]}.',
                        'action': 'Mark as Taken'
                    })
        
        return reminders
    
    def check_health_milestones(self, goals: List[Dict]) -> List[Dict]:
        """Generate goal achievement notifications"""
        milestones = []
        
        for goal in goals:
            progress = goal.get('progress', 0)
            if progress >= 100 and not goal.get('notified'):
                milestones.append({
                    'type': 'achievement',
                    'title': '🏆 Goal Achieved!',
                    'message': f'Congratulations! You achieved your {goal["goal_type"]} goal!',
                    'action': 'Share Achievement'
                })
                goal['notified'] = True
        
        return milestones

def render_notification_center(username: str):
    """Render notification center in UI"""
    from utils.database import get_notifications, mark_all_read
    
    st.markdown("### 🔔 Notification Center")
    
    notifications = get_notifications(username)
    
    if not notifications:
        st.info("No notifications yet")
        return
    
    # Group by priority
    high_priority = [n for n in notifications if n.get('type') in ['critical', 'alert']]
    
    if high_priority:
        st.markdown("#### ⚠️ High Priority")
        for n in high_priority:
            icon = '🚨' if n['type'] == 'critical' else '⚠️'
            st.markdown(f"""
            <div style="background:rgba(239,68,68,0.1); border-left:3px solid #ef4444; 
                        padding:0.75rem; border-radius:8px; margin-bottom:0.5rem;">
                <div style="display:flex; gap:0.5rem;">
                    <span>{icon}</span>
                    <div><strong>{n.get('title', 'Alert')}</strong><br>{n['msg']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Regular notifications
    regular = [n for n in notifications if n.get('type') not in ['critical', 'alert']]
    if regular:
        st.markdown("#### 📬 All Notifications")
        for n in regular[:10]:
            icon = {'reminder': '💊', 'appointment': '📅', 'success': '✅', 'info': 'ℹ️'}.get(n['type'], '🔔')
            st.markdown(f"""
            <div style="padding:0.5rem 0; border-bottom:1px solid rgba(255,255,255,0.05);">
                <div style="display:flex; gap:0.5rem;">
                    <span>{icon}</span>
                    <div style="flex:1">{n['msg']}</div>
                    <span style="font-size:0.7rem; color:#8aa0b5;">{n['time_str']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    if st.button("✓ Mark All as Read", type="primary", use_container_width=True):
        mark_all_read(username)
        st.rerun()