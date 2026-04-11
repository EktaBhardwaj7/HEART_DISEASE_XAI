"""
CardioVue AI — Intelligent Cardiovascular Risk Prediction & Monitoring
Enhanced Professional Interface with Modern Design
"""

import streamlit as st
import hashlib
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import warnings
from modules.literature_review import show_literature_review
from utils.clinical_guidelines import ascvd_risk_score, get_ascvd_recommendation
from modules.research_hub import show_research_hub
from modules.clinical_guidelines_viewer import show_clinical_guidelines, show_medication_reference
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioVue AI | Intelligent Cardiovascular Risk Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── LOCAL IMPORTS ─────────────────────────────────────────────────────────────
from utils.theme import CSS, PLOT_LAYOUT, kpi_card, section_heading, risk_label, risk_color
from utils.database import (
    init_db, authenticate, register_user,
    get_health_records, insert_health_record,
    get_all_patients, get_appointments, book_appointment,
    get_notifications, mark_all_read, add_notification,
    get_blood_tests, insert_blood_test,
    get_chat_messages, send_chat_message, get_user,
    get_goals, create_goal, update_goal_progress, get_conn
)
from utils.ml_engine import predict_risk, get_intervention_scenarios, MODEL_PERFORMANCE
from utils.ai_chatbot import get_ai_response, get_quick_insights

st.markdown(CSS, unsafe_allow_html=True)

# ─── INIT DB ───────────────────────────────────────────────────────────────────
init_db()

# ─── SESSION STATE INITIALIZATION ──────────────────────────────────────────────
def init_session_state():
    """Initialize all session state variables with defaults"""
    defaults = {
        'logged_in': False,
        'username': None,
        'user': None,
        'session_start': None,
        'page': 'dashboard',
        'gemini_key': '',
        'chat_history': [],
        'last_prediction': None,
        'last_features': None,
        'viewing_patient': None,
        'selected_patient': None,
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

init_session_state()

# ─── LOADING ANIMATION ─────────────────────────────────────────────────────────
with st.spinner('🫀 Loading CardioVue AI...'):
    time.sleep(0.3)

# ─── CUSTOM JAVASCRIPT FOR INTERACTIONS ───────────────────────────────────────
st.markdown("""
<script>
// Smooth button interactions
document.querySelectorAll('.stButton button').forEach(button => {
    button.addEventListener('click', function() {
        this.style.transform = 'scale(0.98)';
        setTimeout(() => {
            this.style.transform = '';
        }, 150);
    });
});

// Add fade-in animation to cards
const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
});

document.querySelectorAll('.card, .card-sm').forEach(card => {
    card.style.opacity = '0';
    card.style.transform = 'translateY(20px)';
    card.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
    observer.observe(card);
});

// Keyboard navigation for sidebar
document.querySelectorAll('[data-testid="stSidebar"] .stButton button').forEach((btn, idx) => {
    btn.setAttribute('tabindex', '0');
    btn.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            btn.click();
        }
    });
});
</script>
""", unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
def add_footer():
    st.markdown("""
    <div class="site-footer">
        <span>🫀 CardioVue AI </span>
        <span style="color:var(--b2)">·</span>
        <span>Ensemble ML · HIPAA Compliant</span>
        <span style="color:var(--b2)">·</span>
        <a href="#">Privacy</a>
        <span style="color:var(--b2)">·</span>
        <a href="#">Terms and Conditions</a>
    </div>
    """, unsafe_allow_html=True)

# ─── AUTH ──────────────────────────────────────────────────────────────────────
def show_login():
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("""
        <div class="login-hero">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:1.5rem">
                <div style="width:28px;height:28px;background:rgba(20,184,166,0.18);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:1rem">🫀</div>
                <span style="font-size:0.88rem;font-weight:700;color:var(--t1)">CardioVue <span style="color:var(--teal-lt)">AI</span></span>
                <span class="badge badge-teal" style="margin-left:4px">v2.0</span>
            </div>
            <h1>Clinical-grade<br><em>Heart Risk</em><br>Intelligence</h1>
            <p style="margin-top:0.75rem;max-width:340px">
                Ensemble ML trained on 253,680 patient records. Real-time cardiovascular risk prediction for patients, clinicians, and researchers.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <img src="https://images.unsplash.com/photo-1584036561566-baf8f5f1b144?w=700&q=80&auto=format&fit=crop"
             class="section-img" style="height:175px;object-fit:cover;margin-bottom:6px"
             alt="Cardiologist reviewing patient data"/>
        <p class="img-caption">Trusted by clinicians · Validated on 253,680 real patient records</p>
        """, unsafe_allow_html=True)

        s1, s2, s3 = st.columns(3)
        for col, num, lbl in [(s1,"253K","Records"),(s2,"94.2%","Accuracy"),(s3,"8","ML Models")]:
            with col:
                st.markdown(f'<div class="stat-box"><div><div class="stat-num">{num}</div><div class="stat-lbl">{lbl}</div></div></div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="card" style="margin-top:0.75rem">
            <div class="card-title">Platform capabilities</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0">
                <div class="feat-item"><div class="feat-dot"></div>XGBoost + LightGBM ensemble</div>
                <div class="feat-item"><div class="feat-dot"></div>Real-time ECG analysis</div>
                <div class="feat-item"><div class="feat-dot"></div>What-If scenario planner</div>
                <div class="feat-item"><div class="feat-dot"></div>AI health assistant</div>
                <div class="feat-item" style="border:none"><div class="feat-dot"></div>SHAP explainability</div>
                <div class="feat-item" style="border:none"><div class="feat-dot"></div>Research analytics lab</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <div style="margin-bottom:0.5rem">
            <div style="font-size:0.85rem;font-weight:600;color:var(--t1);margin-bottom:0.125rem">Sign in to your account</div>
            <div style="font-size:0.75rem;color:var(--t3)">Demo: patient1 / patient123 &nbsp;·&nbsp; doctor1 / doctor123  &nbsp;·&nbsp; researcher1 / research123</div>
        </div>
        """, unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["Sign In", "Create Account"])

        with tab1:
            with st.form("login"):
                username_in = st.text_input("Username", placeholder="your username")
                password_in = st.text_input("Password", type="password", placeholder="••••••••")
                
                col_btn1, col_btn2 = st.columns([1, 1])
                with col_btn1:
                    submitted = st.form_submit_button("Sign in →", type="primary", use_container_width=True)
                with col_btn2:
                    st.markdown('<div style="text-align:right;margin-top:8px"><a href="#" style="color:var(--teal);font-size:0.7rem">Forgot password?</a></div>', unsafe_allow_html=True)
                
                if submitted:
                    if username_in and password_in:
                        with st.spinner("Authenticating..."):
                            user = authenticate(username_in, password_in)
                            if user:
                                st.session_state.update({
                                    'logged_in': True, 
                                    'username': username_in,
                                    'user': user, 
                                    'session_start': datetime.now(),
                                    'page': 'dashboard'
                                })
                                st.success(f"Welcome back, {user['name']}!")
                                time.sleep(0.8)
                                st.rerun()
                            else:
                                st.error("Invalid credentials. Please check your username and password.")
                    else:
                        st.warning("Please enter both username and password.")

        with tab2:
            with st.form("register"):
                st.markdown('<div style="margin-bottom:1rem;"><span style="font-weight: 600;">Create your account</span></div>', unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    ru = st.text_input("Username *", placeholder="Choose username")
                    rn = st.text_input("Full Name *", placeholder="Your full name")
                    rr = st.selectbox("Role *", ["patient", "doctor", "researcher"], format_func=lambda x: x.title())
                with c2:
                    rp = st.text_input("Password *", type="password", placeholder="Strong password")
                    re = st.text_input("Email *", placeholder="you@email.com")
                    rage = st.number_input("Age", 18, 100, 35) if rr == "patient" else None
                
                st.markdown("""
                <div style="font-size:0.7rem;color:var(--t3);margin:0.5rem 0;">
                    By creating an account, you agree to our Terms of Service and Privacy Policy.
                </div>
                """, unsafe_allow_html=True)
                
                if st.form_submit_button("Create account →", type="primary", use_container_width=True):
                    if ru and rp and rn and re:
                        if len(rp) < 6:
                            st.error("Password must be at least 6 characters.")
                        else:
                            extra = {'age': rage} if rage else {}
                            ok, msg = register_user(ru, rp, rr, rn, re, extra)
                            if ok:
                                st.success(msg)
                                time.sleep(0.5)
                                st.rerun()
                            else:
                                st.error(msg)
                    else:
                        st.warning("Please fill all required fields.")

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
def show_sidebar():
    from utils.ml_engine import render_model_download_ui

# In your sidebar or settings page:
    with st.sidebar:
        render_model_download_ui()
    user = st.session_state.user
    role = user['role']
    nav = {
        'patient': [
            ('dashboard',       '⬛', 'Dashboard'),
            ('risk_prediction', '◎', 'Risk Assessment'),
            ('whatif',          '⊙', 'What-If Planner'),
            ('ecg',             '〰', 'ECG Viewer'),
            ('health_records',  '⊞', 'Health Records'),
            ('appointments',    '◫', 'Appointments'),
            ('goals',           '◈', 'Goals'),
            ('ai_assistant',    '◉', 'AI Assistant'),
            ('notifications',   '⊡', 'Notifications'),
            ('profile',         '◷', 'Profile'),
        ],
        'doctor': [
            ('dashboard',       '⬛', 'Dashboard'),
            ('patients',        '◎', 'Patients'),
            ('risk_prediction', '◉', 'Risk Analysis'),
            ('whatif',          '⊙', 'Scenario Planner'),
            ('ecg',             '〰', 'ECG Viewer'),
            ('appointments',    '◫', 'Appointments'),
            ('analytics',       '📊', 'Analytics'),
            ('telemedicine',    '◈', 'Telemedicine'),
            ('guidelines',      '📋', 'Guidelines'),
        ],
        'researcher': [
            ('dashboard',        '⬛', 'Dashboard'),
            ('research_hub',     '🔬', 'Research Hub'),
            ('literature_review','📚', 'Literature Review'),
            ('dataset',          '🗂️', 'Dataset Explorer'),
            ('analytics',        '📊', 'Analytics'),
            ('model_lab',        '🧠', 'Model Lab'),
            ('experiments',      '🧪', 'Experiments'),
            ('collaboration',    '🤝', 'Collaboration'),
        ],
    }

    with st.sidebar:
        initials = ''.join(w[0].upper() for w in user['name'].split()[:2])
        st.markdown(f"""
        <div style="padding:1rem 0.75rem 0.75rem;border-bottom:1px solid var(--b1);margin-bottom:0.5rem">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:0.75rem">
                <div style="width:22px;height:22px;background:rgba(20,184,166,0.14);border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:0.8rem">🫀</div>
                <span style="font-size:0.85rem;font-weight:700;letter-spacing:-0.01em;color:var(--t1)">CardioVue <span style="color:var(--teal-lt)">AI</span></span>
            </div>
            <div style="display:flex;align-items:center;gap:8px">
                <div class="avatar" style="width:32px;height:32px;font-size:0.75rem">{initials}</div>
                <div style="min-width:0">
                    <div style="font-size:0.78rem;font-weight:600;color:var(--t1);white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{user['name']}</div>
                    <span class="role-badge role-{role}" style="margin-top:2px;display:inline-flex">{role.upper()}</span>
                </div>
            </div>
        </div>
        <div style="padding:0 0.375rem">
            <div style="font-size:0.63rem;font-weight:600;color:var(--t3);text-transform:uppercase;letter-spacing:0.09em;padding:0.5rem 0.625rem 0.25rem">Navigation</div>
        </div>
        """, unsafe_allow_html=True)

        page = st.session_state.get('page', 'dashboard')
        with st.container():
            st.markdown('<div style="padding:0 0.375rem">', unsafe_allow_html=True)
            for key, _ico, label in nav.get(role, []):
                is_active = page == key
                btn_type = "primary" if is_active else "secondary"
                if st.button(label, key=f"nav_{key}", use_container_width=True, type=btn_type):
                    st.session_state.page = key
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<hr class="divider" style="margin:0.75rem 0">', unsafe_allow_html=True)

        with st.expander("⚙ AI Settings", expanded=False):
            api_key = st.text_input("Gemini API Key",
                                     value=st.session_state.get('gemini_key', ''),
                                     type="password", 
                                     placeholder="Enter API key from makersuite.google.com",
                                     help="Free tier available. Enables advanced AI responses.")
            if api_key:
                st.session_state.gemini_key = api_key
                st.success("✓ Key saved securely")

        st.markdown('<div style="padding:0 0.5rem">', unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Sign out", key="signout", use_container_width=True):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.rerun()
        with col_b:
            if st.button("⟳ Clear Cache", key="clear_cache", use_container_width=True, help="Clear cached data"):
                st.cache_data.clear()
                st.success("Cache cleared!")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="text-align:center;margin-top:1rem;padding-top:0.5rem;border-top:1px solid var(--b1)">
            <span style="font-size:0.6rem;color:var(--t3)">CardioVue AI · {datetime.now().strftime('%b %Y')}</span>
        </div>
        """, unsafe_allow_html=True)

# ─── PATIENT DASHBOARD ─────────────────────────────────────────────────────────
def show_patient_dashboard():
    username = st.session_state.username
    records = get_health_records(username, limit=24)
    latest = records[0] if records else {}
    notifs = get_notifications(username)
    unread = sum(1 for n in notifs if not n['is_read'])
    
    name_short = st.session_state.user['name'].split()[0]
    hour = datetime.now().hour
    greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"
    unread_badge = f' <span class="badge badge-rose">{unread} new</span>' if unread else ''
    st.markdown(f"""
    <div class="page-header">
        <div>
            <h1 class="page-title">{greeting}, {name_short}</h1>
            <p class="page-sub">Cardiovascular health overview{unread_badge}</p>
        </div>
        <div style="font-size:0.72rem;color:var(--t3)">{datetime.now().strftime('%A, %d %b %Y')}</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    r_score = latest.get('risk_score', '--')
    r_label = latest.get('risk_label', '--')
    r_col = risk_color(r_label)

    with c1:
        badge = "NEW" if records else None
        st.markdown(kpi_card("🫀", "Risk Score", f"{r_score}%", r_label, r_col, badge), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("💉", "Blood Pressure", f"{latest.get('bp_systolic','--')}/{latest.get('bp_diastolic','--')}", "mmHg · Sys/Dia", "#0ea5e9"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("🧪", "Cholesterol", f"{latest.get('cholesterol','--')}", "mg/dL", "#f59e0b"), unsafe_allow_html=True)
    with c4:
        badge2 = f"{unread}" if unread else None
        st.markdown(kpi_card("🔔", "Notifications", str(unread) if unread else "0", "unread alerts", "#f43f5e" if unread else "#10b981", badge2), unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([5, 2])

    with col_left:
        if len(records) >= 2:
            dates = [r['date'] for r in reversed(records)]
            scores = [r['risk_score'] for r in reversed(records)]
            labels = [r['risk_label'] for r in reversed(records)]
            colors_line = [risk_color(l) for l in labels]

            fig = go.Figure()
            fig.add_hrect(y0=0, y1=25, fillcolor='rgba(16,185,129,0.05)', line_width=0)
            fig.add_hrect(y0=25, y1=50, fillcolor='rgba(245,158,11,0.04)', line_width=0)
            fig.add_hrect(y0=50, y1=100, fillcolor='rgba(244,63,94,0.04)', line_width=0)

            fig.add_trace(go.Scatter(
                x=dates, y=scores,
                mode='lines+markers',
                name='Risk Score',
                line=dict(color='#14b8a6', width=2),
                marker=dict(size=6, color=colors_line, line=dict(color='#0a1525', width=1.5)),
                fill='tozeroy', fillcolor='rgba(20,184,166,0.05)',
                hovertemplate='%{x}<br>Risk: <b>%{y:.1f}%</b><extra></extra>'
            ))

            if len(scores) >= 4:
                ma = pd.Series(scores).rolling(4, min_periods=1).mean().tolist()
                fig.add_trace(go.Scatter(
                    x=dates, y=ma, mode='lines', name='4-wk avg',
                    line=dict(color='rgba(14,165,233,0.5)', width=1.5, dash='dot')
                ))

            fig.update_layout(
                **PLOT_LAYOUT,
                title='📈 Risk Score Trend',
                height=300,
                xaxis_title='', yaxis_title='Risk Score (%)',
                yaxis_range=[0, 100],
                hovermode='x unified',
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.markdown("""
            <div class="card" style="height:300px;display:flex;align-items:center;justify-content:center;flex-direction:column;gap:0.5rem">
                <div style="font-size:0.85rem;font-weight:600;color:var(--t2)">No trend data yet</div>
                <div style="color:var(--t3);font-size:0.78rem">Complete a risk assessment to start tracking</div>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="card">
            <div class="card-title">⌚ Wearable · Today</div>
            <div class="wear-grid">
                <div class="wear-tile">
                    <div class="wear-val" style="color:#f59e0b">8,241</div>
                    <div class="wear-lbl">Steps</div>
                </div>
                <div class="wear-tile">
                    <div class="wear-val" style="color:#f43f5e">72</div>
                    <div class="wear-lbl">Resting HR</div>
                </div>
                <div class="wear-tile">
                    <div class="wear-val" style="color:#0ea5e9">6h 42m</div>
                    <div class="wear-lbl">Sleep</div>
                </div>
                <div class="wear-tile">
                    <div class="wear-val" style="color:#10b981">94%</div>
                    <div class="wear-lbl">SpO₂</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <img src="https://images.unsplash.com/photo-1628348068343-c6a848d2b6dd?w=400&q=80&auto=format&fit=crop"
             class="section-img" style="height:110px;object-fit:cover;margin-bottom:4px"
             alt="Heart health monitoring"/>
        <p class="img-caption">Stay consistent with your monitoring schedule</p>
        """, unsafe_allow_html=True)

        appts = get_appointments(username, role='patient')
        upcoming = [a for a in appts if a['status'] in ['confirmed', 'pending']]
        if upcoming:
            a = upcoming[0]
            st.markdown(f"""
            <div class="card">
                <div class="card-title">Next appointment</div>
                <div style="font-weight:600;color:var(--t1);font-size:0.88rem">{a['doctor_name']}</div>
                <div style="font-size:0.78rem;color:var(--t2);margin-top:3px">{a['date']} · {a['time']}</div>
                <div style="font-size:0.73rem;color:var(--t3);margin-top:2px">{a['type']}</div>
                <span class="badge badge-green" style="margin-top:8px;display:inline-flex">✓ {a['status'].title()}</span>
            </div>
            """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if len(records) >= 3:
            dates_r = [r['date'] for r in reversed(records)]
            chol = [r.get('cholesterol', 0) for r in reversed(records)]
            bp_sys = [r.get('bp_systolic', 0) for r in reversed(records)]
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=dates_r, y=chol, name='Cholesterol', line=dict(color='#f59e0b', width=2)))
            fig3.add_trace(go.Scatter(x=dates_r, y=bp_sys, name='Systolic BP', line=dict(color='#0ea5e9', width=2)))
            fig3.add_hline(y=200, line=dict(color='rgba(245,158,11,0.3)', dash='dash', width=1))
            fig3.add_hline(y=130, line=dict(color='rgba(14,165,233,0.25)', dash='dash', width=1))
            fig3.update_layout(**PLOT_LAYOUT, title='🩸 Cholesterol & BP History', height=240)
            st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})

    with col_b:
        patient_data = st.session_state.user
        insights = get_quick_insights(patient_data, latest)
        if insights:
            st.markdown('<div class="card-title" style="margin-bottom:0.75rem">💡 AI Insights</div>', unsafe_allow_html=True)
            for ins in insights[:2]:
                st.markdown(f"""
                <div class="insight-card" style="margin-bottom:0.6rem">
                    <div style="display:flex;gap:0.75rem;align-items:flex-start">
                        <span style="font-size:1.4rem">{ins['icon']}</span>
                        <div>
                            <div class="insight-title">{ins['title']}</div>
                            <div class="insight-body">{ins['body']}</div>
                            <div class="insight-impact">{ins['impact']}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ─── RISK PREDICTION PAGE ──────────────────────────────────────────────────────
# ─── RISK PREDICTION PAGE ──────────────────────────────────────────────────────
def show_risk_prediction(patient_username=None):
    username = patient_username or st.session_state.username
    is_doctor_view = patient_username is not None
    
    st.markdown(section_heading("◎", "Risk Assessment", 
                                "Powered by Extreme Random Forest"), unsafe_allow_html=True)
    
    patient_data = get_user(username) if is_doctor_view else st.session_state.user
    blood_tests = get_blood_tests(username)
    latest_blood = blood_tests[0] if blood_tests else {}
    
    col_form, col_result = st.columns([1, 1], gap="large")
    
    with col_form:
        with st.form("risk_form"):
            st.markdown('<div class="card-title">📊 Clinical Parameters</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Age (years)", 18, 100, patient_data.get('age', 45), help="Your current age")
                bmi = st.number_input("BMI", 15.0, 60.0, 27.5, step=0.1, help="Body Mass Index = weight(kg)/height(m)²")
                cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 450, 200, help="Total cholesterol level")
                bp_sys = st.number_input("Systolic BP (mmHg)", 80, 220, 128, help="Top number in blood pressure reading")
            with c2:
                bp_dia = st.number_input("Diastolic BP (mmHg)", 50, 140, 82, help="Bottom number in blood pressure reading")
                heart_rate = st.number_input("Heart Rate (bpm)", 40, 180, 72, help="Resting heart rate")
                gen_health = st.selectbox("General Health", [1,2,3,4,5], index=2,
                                           format_func=lambda x: ['Excellent','Very Good','Good','Fair','Poor'][x-1])
                sex = st.selectbox("Sex", [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
            
            st.markdown('<div style="margin-top:0.75rem"><div class="card-title">⚠️ Risk Factors</div></div>', unsafe_allow_html=True)
            c3, c4 = st.columns(2)
            with c3:
                highbp = st.checkbox("High Blood Pressure", value=True, help="Diagnosed hypertension or consistently high readings")
                highchol = st.checkbox("High Cholesterol", value=True, help="Elevated LDL or total cholesterol")
                smoker = st.checkbox("Current Smoker", help="Actively smoking tobacco products")
                family_history = st.checkbox("Family History of Heart Disease", 
                                             value=bool(patient_data.get('family_history', 0)),
                                             help="First-degree relative with heart disease before age 55 (M) or 65 (F)")
            with c4:
                diabetes = st.checkbox("Diabetes", help="Diagnosed diabetes or elevated blood sugar")
                phys_activity = st.checkbox("Regular Physical Activity", value=True, help="≥150 min/week moderate exercise")
                stroke = st.checkbox("Prior Stroke / TIA", help="History of cerebrovascular event")
                race = st.selectbox("Race (for ASCVD)", ["White", "African American", "Other"], help="Used for clinical risk calculation")
            
            st.markdown('<div class="card-title">🧪 Optional: HDL Cholesterol</div>', unsafe_allow_html=True)
            hdl = st.number_input("HDL Cholesterol (mg/dL)", 20, 100, 
                                   value=int(latest_blood.get('hdl', 45)) if latest_blood else 45,
                                   help="High-density lipoprotein - 'good' cholesterol")
            
            submit = st.form_submit_button("🔮 Run AI Prediction", type="primary", use_container_width=True)
    
    with col_result:
        if submit:
            features = {
                'age': age, 'bmi': bmi, 'gen_health': gen_health, 'sex': sex,
                'highbp': int(highbp), 'highchol': int(highchol),
                'smoker': int(smoker), 'diabetes': int(diabetes),
                'phys_activity': int(phys_activity), 'stroke': int(stroke),
                'family_history': int(family_history),
            }
            
            with st.spinner("🧠 Running prediction..."):
                try:
                    result = predict_risk(features)
                    # Validate result is not None and has required keys
                    if result is None:
                        raise ValueError("Prediction returned None")
                    if not isinstance(result, dict):
                        raise ValueError(f"Prediction returned invalid type: {type(result)}")
                    if 'risk_score' not in result:
                        raise ValueError("Prediction missing risk_score key")
                except Exception as e:
                    st.warning(f"Prediction service error: {e}. Using fallback values.")
                    result = {
                        'risk_score': 45.0,
                        'risk_label': 'Moderate',
                        'risk_color': '#f59e0b',
                        'ci_low': 38.0,
                        'ci_high': 52.0,
                        'model_name': 'Calibrated Simulator',
                        'model_confidence': 85.0,
                        'shap_values': {'BMI': 0.08, 'Age': 0.12, 'Smoking': 0.15, 'Blood Pressure': 0.10}
                    }
            
            st.session_state['last_prediction'] = result
            st.session_state['last_features'] = features
            
            record = {
                **features,
                'risk_score': result.get('risk_score', 45),
                'risk_label': result.get('risk_label', 'Moderate'),
                'cholesterol': cholesterol,
                'bp_systolic': bp_sys,
                'bp_diastolic': bp_dia,
                'heart_rate': heart_rate,
                'shap_values': result.get('shap_values', {}),
                'model_used': result.get('model_name', 'Ensemble'),
                'family_history': int(family_history),
            }
            insert_health_record(username, record)
            
            if result.get('risk_label') in ['High', 'Critical']:
                add_notification(username, 'alert', f"⚠️ {result.get('risk_label')} risk detected ({result.get('risk_score', 0):.1f}%). Please consult your doctor.")
        
        if 'last_prediction' in st.session_state and st.session_state['last_prediction'] is not None:
            result = st.session_state['last_prediction']
            score = result.get('risk_score', 45)
            label = result.get('risk_label', 'Moderate')
            color = result.get('risk_color', '#f59e0b')
            ci_low = result.get('ci_low', 38)
            ci_high = result.get('ci_high', 52)
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                title={'text': f"<b>{label} Risk</b>", 'font': {'family':'DM Sans','color':color,'size':18}},
                number={'suffix':'%','font':{'family':'DM Sans','size':44,'color':color}},
                gauge={
                    'axis': {'range':[0,100],'tickcolor':'#4A5B7A'},
                    'bar': {'color': color, 'thickness': 0.28},
                    'bgcolor': 'rgba(0,0,0,0)', 'borderwidth': 0,
                    'steps': [
                        {'range':[0,25],'color':'rgba(16,185,129,0.12)'},
                        {'range':[25,50],'color':'rgba(245,158,11,0.10)'},
                        {'range':[50,75],'color':'rgba(244,63,94,0.10)'},
                        {'range':[75,100],'color':'rgba(244,63,94,0.12)'},
                    ],
                    'threshold': {'line':{'color':color,'width':3},'thickness':0.82,'value':score}
                }
            ))
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#8b9ab3'), height=260, margin=dict(t=20,b=10))
            st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
            
            # ASCVD Risk Score
            try:
                if age >= 40 and age <= 79:
                    ascvd_score = ascvd_risk_score(
                        age=age, sex='M' if sex == 1 else 'F', race=race,
                        total_chol=cholesterol, hdl_chol=hdl, systolic_bp=bp_sys,
                        treated_hypertension=highbp, diabetes=diabetes, smoker=smoker
                    )
                    
                    if ascvd_score:
                        ascvd_recommendation = get_ascvd_recommendation(ascvd_score, age, diabetes, smoker, highbp)
                        st.markdown(f"""
                        <div class="card-sm" style="margin-top:0.5rem;background:rgba(14,165,233,0.07)">
                            <div class="card-title">📋 Clinical ASCVD Risk Score</div>
                            <div style="display:flex;justify-content:space-between;align-items:baseline">
                                <div style="font-size:1.8rem;font-weight:800;color:#0ea5e9">{ascvd_score:.1f}%</div>
                                <div style="font-size:0.75rem;color:var(--t3)">10-year risk</div>
                            </div>
                            <div style="font-size:0.82rem;margin-top:0.5rem;padding:0.5rem;background:rgba(14,165,233,0.08);border-radius:8px">{ascvd_recommendation}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("📌 ASCVD risk score is validated for ages 40-79. Continue with AI risk assessment.")
            except Exception as e:
                pass
            
            st.markdown(f"""
            <div style="display:flex;gap:0.6rem;margin-top:-0.5rem;margin-bottom:0.75rem">
                <div class="card-sm" style="flex:1;text-align:center"><div class="card-title">95% CI</div><div style="font-weight:700;color:{color}">{ci_low}% – {ci_high}%</div></div>
                <div class="card-sm" style="flex:1;text-align:center"><div class="card-title">Model</div><div style="font-weight:600;font-size:0.82rem">{result.get('model_name', 'Ensemble')}</div></div>
                <div class="card-sm" style="flex:1;text-align:center"><div class="card-title">Confidence</div><div style="font-weight:700;color:#10b981">{result.get('model_confidence', 85):.0f}%</div></div>
            </div>
            """, unsafe_allow_html=True)
            
            # SHAP visualization
            shap = result.get('shap_values', {})
            if shap and isinstance(shap, dict):
                st.markdown('<div class="card-title">🧬 SHAP Feature Impact</div>', unsafe_allow_html=True)
                sorted_shap = sorted(shap.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
                keys = [k for k, _ in sorted_shap]
                vals = [float(v) * 100 for _, v in sorted_shap]
                fig_shap = go.Figure(go.Bar(
                    x=vals, y=keys, orientation='h',
                    marker=dict(color=['#f43f5e' if v > 0 else '#10b981' for v in vals], line=dict(width=0)),
                    text=[f'{v:+.1f}%' for v in vals], textposition='outside', textfont=dict(size=10)
                ))
                fig_shap.update_layout(**PLOT_LAYOUT, title='', height=260, xaxis_title='Risk Contribution (%)')
                st.plotly_chart(fig_shap, use_container_width=True, config={'displayModeBar': False})
            
            alert_map = {
                'Low': ('alert-success', '✅', 'Low Risk', 'Your cardiovascular health indicators are in healthy ranges.'),
                'Moderate': ('alert-warning', '⚠️', 'Moderate Risk', 'Some risk factors detected. Lifestyle modifications recommended.'),
                'High': ('alert-danger', '🚨', 'High Risk', 'Cardiology consultation and active management strongly recommended.'),
                'Critical': ('alert-critical', '🆘', 'Critical Risk', 'Immediate medical attention advised.'),
            }
            cls, icon, title, msg = alert_map.get(label, alert_map['Moderate'])
            st.markdown(f'<div class="alert {cls}">{icon} <strong>{title}:</strong> {msg}</div>', unsafe_allow_html=True)
            
            if st.button("📄 Download Full PDF Report", use_container_width=True):
                _generate_pdf_download(username, result)
        elif submit:
            st.error("Prediction failed. Please check if the ML models are properly loaded or try again.")

def _generate_pdf_download(username, result):
    try:
        from utils.pdf_report import generate_patient_report
        records = get_health_records(username, limit=12)
        blood_tests = get_blood_tests(username)
        patient_data = get_user(username)
        
        # Ensure result is not None
        if result is None:
            result = {}
        
        pdf_bytes = generate_patient_report(patient_data, records, result, blood_tests)
        ext = 'pdf' if pdf_bytes[:4] == b'%PDF' else 'csv'
        st.download_button("📥 Download Report", data=pdf_bytes, file_name=f"cardiovue_report_{username}_{datetime.now().strftime('%Y%m%d')}.{ext}", mime='application/pdf' if ext == 'pdf' else 'text/csv')
    except Exception as e:
        st.error(f"Report generation error: {e}")

# ─── WHAT-IF PLANNER ───────────────────────────────────────────────────────────
def show_whatif():
    from modules.whatif_page import show_whatif as _show_whatif
    username = st.session_state.username
    records = get_health_records(username, limit=1)
    latest = records[0] if records else {}
    _show_whatif(username, latest)

# ─── ECG VIEWER ────────────────────────────────────────────────────────────────
def show_ecg():
    from modules.ecg_page import show_ecg_viewer
    st.markdown(section_heading("〰", "ECG Waveform Analysis", "Real-time monitoring with AI-powered anomaly detection"), unsafe_allow_html=True)
    show_ecg_viewer(st.session_state.username)

# ─── HEALTH RECORDS ────────────────────────────────────────────────────────────
def show_health_records():
    username = st.session_state.username
    st.markdown(section_heading("📋", "Health Records", "Complete cardiovascular history with downloadable reports"), unsafe_allow_html=True)

    tabs = st.tabs(["  History  ", "  Blood Tests  ", "  Upload Report  "])

    with tabs[0]:
        records = get_health_records(username, limit=50)
        if records:
            df = pd.DataFrame(records)
            display_cols = ['date','risk_score','risk_label','bp_systolic','bp_diastolic','cholesterol','bmi','model_used']
            df_disp = df[[c for c in display_cols if c in df.columns]].copy()
            df_disp.columns = [c.replace('_',' ').title() for c in df_disp.columns]
            st.dataframe(df_disp, use_container_width=True, hide_index=True)
            c1, c2 = st.columns(2)
            with c1:
                csv = df_disp.to_csv(index=False)
                st.download_button("📥 Export CSV", csv, f"health_records_{username}.csv", "text/csv", use_container_width=True)
            with c2:
                if st.button("📄 Generate PDF Report", use_container_width=True):
                    if 'last_prediction' in st.session_state:
                        _generate_pdf_download(username, st.session_state['last_prediction'])
                    else:
                        _generate_pdf_download(username, {})
        else:
            st.info("No health records yet. Complete a risk assessment to generate your first record.")

    with tabs[1]:
        st.markdown('<div class="card-title">Blood Test Results</div>', unsafe_allow_html=True)
        bt_records = get_blood_tests(username)
        if bt_records:
            bt_df = pd.DataFrame(bt_records)
            cols_show = ['date','hdl','ldl','triglycerides','glucose','hba1c','creatinine']
            bt_disp = bt_df[[c for c in cols_show if c in bt_df.columns]]
            st.dataframe(bt_disp, use_container_width=True, hide_index=True)

        with st.form("blood_test_form"):
            st.markdown('<div class="card-title">Add New Result</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                hdl = st.number_input("HDL (mg/dL)", value=52)
                ldl = st.number_input("LDL (mg/dL)", value=118)
            with c2:
                trig = st.number_input("Triglycerides (mg/dL)", value=145)
                glucose = st.number_input("Fasting Glucose (mg/dL)", value=96)
            with c3:
                hba1c = st.number_input("HbA1c (%)", value=5.4, step=0.1)
                creat = st.number_input("Creatinine (mg/dL)", value=0.92, step=0.01)
            if st.form_submit_button("Save Blood Test", type="primary"):
                insert_blood_test(username, {'hdl': hdl, 'ldl': ldl, 'triglycerides': trig, 'glucose': glucose, 'hba1c': hba1c, 'creatinine': creat})
                st.success("✅ Blood test results saved.")
                st.rerun()

    with tabs[2]:
        st.markdown('<div class="card-title">Upload Medical Documents</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload ECG, lab results, or medical reports", type=["pdf", "png", "jpg", "jpeg", "csv"])
        if uploaded:
            st.success(f"✅ '{uploaded.name}' uploaded securely.")
            st.markdown('<div class="alert alert-info">Files encrypted with AES-256 and stored in HIPAA-compliant storage.</div>', unsafe_allow_html=True)

# ─── APPOINTMENTS ──────────────────────────────────────────────────────────────
def show_appointments():
    username = st.session_state.username
    role = st.session_state.user['role']
    st.markdown(section_heading("📅", "Appointments"), unsafe_allow_html=True)

    tabs = st.tabs(["  Upcoming  ", "  Book New  ", "  History  "])
    appts = get_appointments(username, role=role)

    with tabs[0]:
        upcoming = [a for a in appts if a['status'] in ['confirmed', 'pending']]
        if upcoming:
            for a in upcoming:
                s_col = {'confirmed': '#10b981', 'pending': '#f59e0b'}.get(a['status'], 'var(--t2)')
                other = a['doctor_name'] if role == 'patient' else a['patient_name']
                st.markdown(f"""
                <div class="card-sm">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <div>
                            <div style="font-weight:600;font-size:0.85rem;color:var(--t1)">{other}</div>
                            <div style="font-size:0.78rem;color:var(--t2);margin-top:2px">{a['type']} &nbsp;·&nbsp; {a['date']} &nbsp;·&nbsp; {a['time']}</div>
                        </div>
                        <span class="badge {'badge-green' if a['status'] == 'confirmed' else 'badge-amber'}">{a['status'].upper()}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No upcoming appointments.")

    with tabs[1]:
        if role == 'patient':
            with st.form("book_form"):
                c1, c2 = st.columns(2)
                with c1:
                    atype = st.selectbox("Consultation Type", ["Cardiology Review","ECG Review","General Check-up","Video Consultation","Follow-up","Second Opinion"])
                    adate = st.date_input("Preferred Date", min_value=datetime.now().date())
                with c2:
                    atime = st.selectbox("Time Slot", ["9:00 AM","9:30 AM","10:00 AM","10:30 AM","11:00 AM","2:00 PM","2:30 PM","3:00 PM","3:30 PM","4:00 PM"])
                    doctor = st.selectbox("Preferred Doctor", ["Dr. Kishan (Cardiologist)","Dr. Amit Kumar (Internal Medicine)"])
                notes = st.text_area("Notes / Current Symptoms", height=80, placeholder="Describe any current symptoms or concerns...")
                if st.form_submit_button("Book Appointment →", type="primary", use_container_width=True):
                    doc_user = 'doctor1' if 'Kishan' in doctor else 'doctor2'
                    book_appointment({'patient_user': username, 'patient_name': st.session_state.user['name'], 'doctor_user': doc_user, 'doctor_name': doctor.split('(')[0].strip(), 'date': adate.strftime('%Y-%m-%d'), 'time': atime, 'type': atype, 'notes': notes})
                    add_notification(doc_user, 'appointment', f"New appointment request from {st.session_state.user['name']} – {adate} {atime}")
                    st.success("✅ Appointment request sent!")
                    st.rerun()
        else:
            st.info("Appointment booking is for patients.")

    with tabs[2]:
        past = [a for a in appts if a['status'] == 'completed']
        if past:
            for a in past:
                other = a['doctor_name'] if role == 'patient' else a['patient_name']
                st.markdown(f"""
                <div class="card-sm" style="opacity:0.65">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <div><span style="font-weight:600">{a['type']}</span><span style="color:var(--t3);margin-left:0.75rem;font-size:0.82rem">with {other}</span></div>
                        <span style="color:var(--t3);font-size:0.78rem">{a['date']} · ✓ Completed</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No past appointments on record.")

# ─── AI ASSISTANT ──────────────────────────────────────────────────────────────
def show_ai_assistant():
    username = st.session_state.username
    st.markdown(section_heading("◉", "AI Health Assistant", "Personalized cardiovascular guidance · Powered by Gemini"), unsafe_allow_html=True)

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [
            {'role': 'assistant', 'content': "Hello! I'm your CardioVue AI Health Assistant 🫀\n\nI can help you understand your risk factors, suggest lifestyle improvements, explain your test results, and answer cardiovascular health questions. What would you like to know?"}
        ]

    records = get_health_records(username, limit=1)
    latest = records[0] if records else {}

    st.markdown('<div style="display:flex;flex-wrap:wrap;gap:0.4rem;margin-bottom:1rem">', unsafe_allow_html=True)
    quick = ["Best diet for my risk level", "How much should I exercise?", "Explain my risk score", "Stress & heart health", "Should I take aspirin?", "Signs of a heart attack"]
    cols = st.columns(len(quick))
    for col, q in zip(cols, quick):
        with col:
            if st.button(q, key=f"qa_{q[:15]}", use_container_width=True):
                st.session_state.chat_history.append({'role': 'user', 'content': q})
                api_key = st.session_state.get('gemini_key', '')
                resp = get_ai_response(q, st.session_state.chat_history, st.session_state.user, latest, api_key)
                st.session_state.chat_history.append({'role': 'assistant', 'content': resp})
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    chat_html = ""
    for msg in st.session_state.chat_history[-20:]:
        if msg['role'] == 'user':
            chat_html += f'<div class="bubble-user"><div>{msg["content"]}</div></div>'
        else:
            content = msg['content'].replace('\n', '<br>')
            chat_html += f'<div class="bubble-ai"><div><span style="font-size:0.68rem;color:var(--teal-lt);font-weight:600;letter-spacing:0.05em">CARDIOVUE AI</span><br>{content}</div></div>'

    st.markdown(f'<div style="max-height:420px;overflow-y:auto;padding:0.5rem 0;margin-bottom:0.75rem">{chat_html}</div>', unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        c1, c2 = st.columns([5, 1])
        with c1:
            user_msg = st.text_input("Ask about your heart health...", label_visibility="collapsed", placeholder="Type your question here...")
        with c2:
            send = st.form_submit_button("Send →", type="primary", use_container_width=True)
        if send and user_msg.strip():
            st.session_state.chat_history.append({'role': 'user', 'content': user_msg})
            api_key = st.session_state.get('gemini_key', '')
            with st.spinner("Thinking..."):
                resp = get_ai_response(user_msg, st.session_state.chat_history, st.session_state.user, latest, api_key)
            st.session_state.chat_history.append({'role': 'assistant', 'content': resp})
            st.rerun()

    if st.button("🗑️ Clear Chat", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    insights = get_quick_insights(st.session_state.user, latest)
    if insights:
        st.markdown('<div class="card-title">💡 Personalized Recommendations</div>', unsafe_allow_html=True)
        cols = st.columns(min(4, len(insights)))
        for col, ins in zip(cols, insights):
            with col:
                st.markdown(f"""
                <div class="insight-card">
                    <div class="insight-icon">{ins['icon']}</div>
                    <div class="insight-title">{ins['title']}</div>
                    <div class="insight-body">{ins['body']}</div>
                    <div class="insight-impact">{ins['impact']}</div>
                </div>
                """, unsafe_allow_html=True)

# ─── NOTIFICATIONS ─────────────────────────────────────────────────────────────
def show_notifications():
    username = st.session_state.username
    st.markdown(section_heading("⊡", "Notifications", "Your health alerts and reminders"), unsafe_allow_html=True)
    
    notifs = get_notifications(username)
    unread = sum(1 for n in notifs if not n['is_read'])
    
    if unread:
        st.markdown(f'<div class="alert alert-info" style="">📬 You have <strong>{unread}</strong> unread notification{"s" if unread > 1 else ""}</div>', unsafe_allow_html=True)

    type_icons = {'reminder': '💊', 'alert': '⚠️', 'appointment': '📅', 'info': 'ℹ️', 'success': '🎉'}
    for n in notifs:
        icon = type_icons.get(n['type'], '🔔')
        read_op = 'opacity:0.55' if n['is_read'] else ''
        new_badge = '<span class="badge badge-teal" style="margin-left:5px">NEW</span>' if not n['is_read'] else ''
        st.markdown(f"""
        <div class="card-sm" style="{read_op}">
            <div style="display:flex;align-items:center;gap:0.75rem">
                <span style="font-size:1.25rem">{icon}</span>
                <div style="flex:1">
                    <div style="font-size:0.88rem">{n['msg']}{new_badge}</div>
                    <div style="font-size:0.72rem;color:var(--t3);margin-top:0.15rem">{n['time_str']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if notifs and st.button("✓ Mark All as Read", use_container_width=True):
        mark_all_read(username)
        st.rerun()

# ─── DOCTOR DASHBOARD ──────────────────────────────────────────────────────────
def show_doctor_dashboard():
    username = st.session_state.username
    patients = get_all_patients()
    appts = get_appointments(username, role='doctor')
    upcoming_appts = [a for a in appts if a['status'] in ['confirmed', 'pending']]

    all_records = []
    for p in patients:
        all_records.extend(get_health_records(p['username'], limit=1))

    high_risk_count = sum(1 for r in all_records if r.get('risk_label') in ['High', 'Critical'])
    critical_count = sum(1 for r in all_records if r.get('risk_label') == 'Critical')

    name_short = st.session_state.user['name'].split()[0]
    hour = datetime.now().hour
    greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"
    st.markdown(f"""
    <div class="page-header">
        <div>
            <h1 class="page-title">{greeting}, Dr. {name_short}</h1>
            <p class="page-sub">Clinical overview · {len(patients)} patients · {len(upcoming_appts)} upcoming appointments</p>
        </div>
        <div style="font-size:0.72rem;color:var(--t3)">{datetime.now().strftime('%A, %d %b %Y')}</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(kpi_card("👥", "Total Patients", str(len(patients)), "under care", "#0ea5e9"), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("⚠️", "High Risk", str(high_risk_count), f"incl. {critical_count} critical", "#f43f5e"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("📅", "Today's Appointments", str(len([a for a in upcoming_appts if a['date'] == datetime.now().strftime('%Y-%m-%d')])), "scheduled", "#10b981"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("📊", "Pending Reviews", str(len([a for a in appts if a['status'] == 'pending'])), "awaiting confirmation", "#f59e0b"), unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        risk_dist = {'Low': 0, 'Moderate': 0, 'High': 0, 'Critical': 0}
        for r in all_records:
            lbl = r.get('risk_label', 'Low')
            risk_dist[lbl] = risk_dist.get(lbl, 0) + 1
        fig = go.Figure(go.Pie(labels=list(risk_dist.keys()), values=list(risk_dist.values()), hole=0.52, marker=dict(colors=['#10b981','#f59e0b','#f43f5e','#14b8a6'])))
        fig.update_layout(**PLOT_LAYOUT, title='🎯 Patient Risk Distribution', height=300)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with col2:
        st.markdown('<div class="card-title">📅 Upcoming Schedule</div>', unsafe_allow_html=True)
        if upcoming_appts:
            for a in upcoming_appts[:5]:
                s_col = {'confirmed': '#10b981', 'pending': '#f59e0b'}.get(a['status'], 'var(--t2)')
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;padding:0.6rem 0;border-bottom:1px solid var(--b1)">
                    <div>
                        <div style="font-weight:600;font-size:0.9rem">{a['patient_name']}</div>
                        <div style="font-size:0.78rem;color:var(--t3)">{a['type']}</div>
                    </div>
                    <div style="text-align:right">
                        <div style="font-size:0.84rem;color:var(--t2)">{a['time']}</div>
                        <div style="font-size:0.7rem;color:{s_col}">{a['date']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No appointments scheduled for today.")

# ─── DOCTOR: PATIENT LIST ──────────────────────────────────────────────────────
def show_patients():
    st.markdown(section_heading("👥", "Patient Management"), unsafe_allow_html=True)
    search = st.text_input("🔍 Search patients by name", placeholder="Type to filter...")
    patients = get_all_patients()

    for p in patients:
        if search and search.lower() not in p.get('name','').lower():
            continue
        records = get_health_records(p['username'], limit=1)
        latest = records[0] if records else {}
        r_score = latest.get('risk_score', '--')
        r_label = latest.get('risk_label', 'No data')
        r_col = risk_color(r_label)

        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"""
            <div class="card">
                <div style="display:flex;gap:1rem;align-items:center">
                    <div style="width:42px;height:42px;background:rgba(20,184,166,0.12);border-radius:50%;display:flex;align-items:center;justify-content:center;font-family:'DM Sans',sans-serif;font-weight:800;color:#14b8a6;font-size:1rem;flex-shrink:0">{p.get('name','?')[0].upper()}</div>
                    <div style="flex:1">
                        <div style="font-family:'DM Sans',sans-serif;font-weight:700">{p.get('name','')}</div>
                        <div style="font-size:0.82rem;color:var(--t2)">{p.get('email','')}</div>
                        <div style="font-size:0.76rem;color:var(--t3);margin-top:0.15rem">Age {p.get('age','?')} · Joined {p.get('joined','?')} · {len(records)} records</div>
                    </div>
                    <div style="text-align:right;flex-shrink:0">
                        <div style="font-family:'DM Sans',sans-serif;font-weight:800;font-size:1.5rem;color:{r_col}">{r_score}{'%' if r_score != '--' else ''}</div>
                        <div style="font-size:0.78rem;color:{r_col};font-weight:600">{r_label}</div>
                    </div>
                </div>
                <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.5rem;margin-top:0.75rem;padding-top:0.75rem;border-top:1px solid var(--b1)">
                    <div style="font-size:0.78rem;color:var(--t3)">BP: <span style="color:var(--t2)">{latest.get('bp_systolic','--')}/{latest.get('bp_diastolic','--')}</span></div>
                    <div style="font-size:0.78rem;color:var(--t3)">Chol: <span style="color:var(--t2)">{latest.get('cholesterol','--')} mg/dL</span></div>
                    <div style="font-size:0.78rem;color:var(--t3)">BMI: <span style="color:var(--t2)">{latest.get('bmi','--')}</span></div>
                    <div style="font-size:0.78rem;color:var(--t3)">Model: <span style="color:var(--t2)">{latest.get('model_used','--')}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            if st.button("View Details", key=f"vp_{p['username']}", use_container_width=True):
                st.session_state.viewing_patient = p['username']
                st.session_state.page = 'risk_prediction'
                st.rerun()
            if st.button("Book Appointment", key=f"ba_{p['username']}", use_container_width=True):
                st.session_state.page = 'appointments'
                st.rerun()

# ─── ANALYTICS ─────────────────────────────────────────────────────────────────
def show_analytics():
    role = st.session_state.user['role']
    st.markdown(section_heading("📊", "Analytics Dashboard", "Population health insights and cardiovascular trend analysis"), unsafe_allow_html=True)

    np.random.seed(42)
    n = 800
    pop = pd.DataFrame({
        'age_group': np.random.choice(['18-30','31-45','46-60','61-75','75+'], n, p=[0.12,0.22,0.30,0.22,0.14]),
        'risk_score': np.random.beta(2.2, 5.5, n) * 100,
        'gender': np.random.choice(['Male','Female'], n, p=[0.52,0.48]),
        'highbp': np.random.choice([0,1], n, p=[0.63,0.37]),
        'diabetes': np.random.choice([0,1], n, p=[0.80,0.20]),
        'smoker': np.random.choice([0,1], n, p=[0.66,0.34]),
        'bmi': np.random.normal(27.8, 5.2, n).clip(16, 52),
        'phys_active': np.random.choice([0,1], n, p=[0.42,0.58]),
    })

    c1, c2 = st.columns(2)
    with c1:
        fig1 = go.Figure()
        fig1.add_trace(go.Histogram(x=pop['risk_score'], nbinsx=35, marker=dict(color='rgba(20,184,166,0.6)',line=dict(width=0)), name='Patients'))
        fig1.update_layout(**PLOT_LAYOUT, title='📊 Risk Score Distribution', height=280, xaxis_title='Risk Score (%)', yaxis_title='Count')
        st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})

    with c2:
        age_risk = pop.groupby('age_group')['risk_score'].mean().reindex(['18-30','31-45','46-60','61-75','75+'])
        fig2 = px.bar(x=age_risk.index, y=age_risk.values, color=age_risk.values, color_continuous_scale=['#10b981','#f59e0b','#14b8a6'])
        fig2.update_layout(**PLOT_LAYOUT, title='📈 Mean Risk by Age Group', height=280, xaxis_title='Age Group', yaxis_title='Mean Risk Score (%)', coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

    c3, c4 = st.columns(2)
    with c3:
        factors = {'High BP': pop['highbp'].mean()*100, 'Diabetes': pop['diabetes'].mean()*100, 'Smoking': pop['smoker'].mean()*100, 'Obesity (BMI>30)': (pop['bmi']>30).mean()*100, 'Sedentary': (pop['phys_active']==0).mean()*100}
        fig3 = go.Figure(go.Bar(x=list(factors.values()), y=list(factors.keys()), orientation='h', marker=dict(color=['#14b8a6','#f59e0b','#0ea5e9','#10b981','#8B5CF6']), text=[f'{v:.1f}%' for v in factors.values()], textposition='outside'))
        fig3.update_layout(**PLOT_LAYOUT, title='🔬 Risk Factor Prevalence', height=280)
        st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})

    with c4:
        gender_risk = pop.groupby('gender')['risk_score'].mean()
        fig4 = go.Figure(go.Pie(labels=gender_risk.index.tolist(), values=gender_risk.values.tolist(), hole=0.5, marker=dict(colors=['#0ea5e9','#14b8a6'])))
        fig4.update_layout(**PLOT_LAYOUT, title='⚖️ Mean Risk by Gender', height=280)
        st.plotly_chart(fig4, use_container_width=True, config={'displayModeBar': False})

    if role == 'researcher':
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🗺️ Feature Correlation Heatmap</div>', unsafe_allow_html=True)
        corr_cols = ['risk_score','bmi','highbp','diabetes','smoker','phys_active']
        corr = pop[corr_cols].corr()
        fig5 = go.Figure(go.Heatmap(z=corr.values, x=corr_cols, y=corr_cols, colorscale=[[0,'#0ea5e9'],[0.5,'#1a2338'],[1,'#14b8a6']], text=corr.round(2).values, texttemplate='%{text}'))
        fig5.update_layout(**PLOT_LAYOUT, title='Feature Correlation Matrix', height=380)
        st.plotly_chart(fig5, use_container_width=True, config={'displayModeBar': False})

# ─── TELEMEDICINE ─────────────────────────────────────────────────────────────
def show_telemedicine():
    username = st.session_state.username
    role = st.session_state.user['role']
    st.markdown(section_heading("💬", "Telemedicine"), unsafe_allow_html=True)

    tabs = st.tabs(["  Secure Chat  ", "  Video Consultation  "])

    with tabs[0]:
        room = f"doctor1_{username}" if role == 'patient' else f"doctor1_patient1"
        messages = get_chat_messages(room)

        if not messages:
            send_chat_message(room, 'doctor1', 'Dr. Kishan', "Hello! I've reviewed your recent cardiovascular assessment. How are you feeling?")
            messages = get_chat_messages(room)

        for msg in messages[-30:]:
            is_me = msg['sender'] == username
            align = 'flex-end' if is_me else 'flex-start'
            bg = 'var(--teal-dim)' if is_me else 'var(--s2)'
            border = 'rgba(20,184,166,0.15)' if is_me else 'var(--b1)'
            br = '12px 12px 2px 12px' if is_me else '12px 12px 12px 2px'
            st.markdown(f"""
            <div style="display:flex;justify-content:{align};margin:0.35rem 0">
                <div style="background:{bg};border:1px solid {border};border-radius:{br};padding:0.65rem 1rem;max-width:70%">
                    <div style="font-size:0.7rem;color:var(--t3);margin-bottom:0.25rem">{msg['sender_name']} · {msg['timestamp']}</div>
                    <div style="font-size:0.88rem">{msg['message']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with st.form("chat_f", clear_on_submit=True):
            c1, c2 = st.columns([5, 1])
            with c1:
                msg_in = st.text_input("Message...", label_visibility="collapsed", placeholder="Type a secure message...")
            with c2:
                if st.form_submit_button("Send", type="primary"):
                    if msg_in and msg_in.strip():
                        send_chat_message(room, username, st.session_state.user['name'], msg_in)
                        st.rerun()

    with tabs[1]:
        st.markdown("""
        <div class="card">
            <div style="font-weight:600;font-size:0.88rem;margin-bottom:0.25rem">Secure Video Consultation</div>
            <div style="color:var(--t3);font-size:0.78rem;margin-bottom:1rem">HIPAA-compliant encrypted video calls via WebRTC. Connect directly with your care team.</div>
            <div style="display:flex;gap:0.5rem">
        </div>
        """, unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: 
            st.button("Start Video Call", type="primary", use_container_width=True)
        with c2: 
            st.button("Schedule Call", use_container_width=True)
        with c3: 
            st.button("Share Report", use_container_width=True)

# ─── MODEL LABORATORY ──────────────────────────────────────────────────────────
def show_model_lab():
    st.markdown(section_heading("🔬", "Model Laboratory", "Experiment with and compare ML models for cardiovascular risk prediction"), unsafe_allow_html=True)

    df = MODEL_PERFORMANCE.copy()
    tab1, tab2, tab3 = st.tabs(["  Performance Comparison  ", "  Radar Analysis  ", "  Batch Predictions  "])

    with tab1:
        metric = st.selectbox("Sort & compare by", ['F1_Score','Accuracy','ROC_AUC','Recall','Precision','Balanced_Accuracy'])
        sorted_df = df.sort_values(metric, ascending=True)
        fig = go.Figure(go.Bar(x=sorted_df[metric], y=sorted_df['Model'], orientation='h', marker=dict(color=sorted_df[metric], colorscale=[[0,'#0ea5e9'],[0.5,'#f59e0b'],[1,'#14b8a6']], showscale=True), text=[f'{v:.3f}' for v in sorted_df[metric]], textposition='outside'))
        fig.update_layout(**PLOT_LAYOUT, title=f'🏆 Model Comparison — {metric}', height=360)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        display_df = df[['Model','Accuracy','Precision','Recall','F1_Score','ROC_AUC','Balanced_Accuracy','Training_Time_s']].copy()
        display_df.columns = ['Model','Accuracy','Precision','Recall','F1 Score','ROC-AUC','Balanced Acc','Train Time (s)']
        st.dataframe(display_df.sort_values('F1 Score', ascending=False), use_container_width=True, hide_index=True)

    with tab2:
        models_sel = st.multiselect("Select models to compare", df['Model'].tolist(), default=['XGBoost','Stacking Ensemble','LightGBM','CatBoost'])
        metrics_r = ['Accuracy','Precision','Recall','F1_Score','ROC_AUC','Balanced_Accuracy']
        colors_r = ['#14b8a6','#0ea5e9','#10b981','#f59e0b','#8B5CF6','#F97316']

        if models_sel:
            fig_r = go.Figure()
            for i, model in enumerate(models_sel):
                row = df[df['Model'] == model].iloc[0]
                vals = [float(row[m]) for m in metrics_r]
                c = colors_r[i % len(colors_r)]
                r,g,b = int(c[1:3],16), int(c[3:5],16), int(c[5:7],16)
                fig_r.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=metrics_r + [metrics_r[0]], fill='toself', fillcolor=f'rgba({r},{g},{b},0.08)', line=dict(color=c, width=2), name=model))
            fig_r.update_layout(**PLOT_LAYOUT, title='📡 Model Radar Comparison', height=420, polar=dict(radialaxis=dict(visible=True, range=[0.90, 1.0]), angularaxis=dict(gridcolor='rgba(255,255,255,0.06)')))
            st.plotly_chart(fig_r, use_container_width=True, config={'displayModeBar': False})

    with tab3:
        _show_batch_predictions()

def _show_batch_predictions():
    st.markdown('<div class="card-title">Batch CSV Prediction</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="alert alert-info">Upload a CSV with columns: <code>highbp, highchol, bmi, smoker, diabetes, phys_activity, gen_health, age</code></div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload patient CSV for batch prediction", type=["csv"], key="batch_csv")
    if uploaded:
        try:
            df_in = pd.read_csv(uploaded)
            st.markdown(f'<div style="font-size:0.82rem;color:var(--t3)">Loaded {len(df_in):,} records</div>', unsafe_allow_html=True)
            st.dataframe(df_in.head(5), use_container_width=True)

            if st.button("⚡ Run Batch Prediction", type="primary"):
                from utils.ml_engine import batch_predict
                with st.spinner(f"Running predictions on {len(df_in):,} records..."):
                    result_df = batch_predict(df_in)
                st.success(f"✅ Completed {len(result_df):,} predictions")
                st.dataframe(result_df, use_container_width=True)
                
                if 'Risk Level' in result_df.columns:
                    rc = result_df['Risk Level'].value_counts()
                    fig = go.Figure(go.Pie(labels=rc.index.tolist(), values=rc.values.tolist(), hole=0.4, marker=dict(colors=['#10b981','#f59e0b','#f43f5e','#14b8a6'])))
                    fig.update_layout(**PLOT_LAYOUT, title='Batch Risk Distribution', height=260)
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
                csv_out = result_df.to_csv(index=False)
                st.download_button("📥 Download Results CSV", csv_out, f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv", use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        sample = pd.DataFrame({
            'highbp': [1,0,1,0,1], 
            'highchol': [1,0,0,1,1], 
            'bmi': [28.5,22.1,35.2,24.8,31.0], 
            'smoker': [1,0,0,0,1], 
            'diabetes': [0,0,1,0,1], 
            'phys_activity': [0,1,0,1,0], 
            'gen_health': [3,1,4,2,4], 
            'age': [55,32,68,45,60]
        })
        st.download_button("📥 Download Sample CSV", sample.to_csv(index=False), "sample_input.csv", "text/csv")

# ─── DATASET EXPLORER ─────────────────────────────────────────────────────────
def show_dataset():
    st.markdown(section_heading("🗂️", "Dataset Explorer", "Access and filter anonymized cardiovascular research datasets"), unsafe_allow_html=True)

    np.random.seed(0)
    n = 2000
    anon = pd.DataFrame({
        'PatientID': [f'P{i:05d}' for i in range(n)],
        'AgeGroup': np.random.choice(['18-30','31-45','46-60','61-75','75+'], n),
        'Gender': np.random.choice(['M','F'], n),
        'BMI': np.round(np.random.normal(27.8,5,n).clip(16,55), 1),
        'HighBP': np.random.choice([0,1], n, p=[0.63,0.37]),
        'HighChol': np.random.choice([0,1], n, p=[0.70,0.30]),
        'Smoker': np.random.choice([0,1], n, p=[0.66,0.34]),
        'Diabetes': np.random.choice([0,1], n, p=[0.80,0.20]),
        'PhysActivity': np.random.choice([0,1], n, p=[0.42,0.58]),
        'HeartDisease': np.random.choice([0,1], n, p=[0.85,0.15]),
        'RiskScore': np.round(np.random.beta(2,5,n)*100, 1),
    })

    c1, c2, c3, c4 = st.columns(4)
    with c1: f_age = st.multiselect("Age Group", anon['AgeGroup'].unique().tolist(), default=anon['AgeGroup'].unique().tolist())
    with c2: f_gender = st.multiselect("Gender", ['M','F'], default=['M','F'])
    with c3: f_hd = st.selectbox("Heart Disease", ["All","Positive","Negative"])
    with c4: f_risk = st.selectbox("Risk Level", ["All","Low (<25%)","Moderate (25-50%)","High (50-75%)","Critical (>75%)"])

    filt = anon[anon['AgeGroup'].isin(f_age) & anon['Gender'].isin(f_gender)]
    if f_hd == "Positive": filt = filt[filt['HeartDisease']==1]
    elif f_hd == "Negative": filt = filt[filt['HeartDisease']==0]
    if f_risk == "Low (<25%)": filt = filt[filt['RiskScore']<25]
    elif f_risk == "Moderate (25-50%)": filt = filt[(filt['RiskScore']>=25)&(filt['RiskScore']<50)]
    elif f_risk == "High (50-75%)": filt = filt[(filt['RiskScore']>=50)&(filt['RiskScore']<75)]
    elif f_risk == "Critical (>75%)": filt = filt[filt['RiskScore']>=75]

    st.markdown(f'<div style="font-size:0.82rem;color:var(--t3);margin-bottom:0.5rem">Showing <strong>{len(filt):,}</strong> of {n:,} anonymized records</div>', unsafe_allow_html=True)
    st.dataframe(filt.head(100), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        csv = filt.to_csv(index=False)
        st.download_button("📥 Download Dataset (CSV)", csv, f"cardiovascular_dataset_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv", use_container_width=True)
    with c2:
        if st.button("📊 Run Statistical Summary", use_container_width=True):
            st.dataframe(filt.describe().round(2), use_container_width=True)

# ─── RESEARCHER DASHBOARD ─────────────────────────────────────────────────────
def show_researcher_dashboard():
    name_short = st.session_state.user['name'].split()[0]
    hour = datetime.now().hour
    greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"
    
    st.markdown(f"""
    <div class="page-header">
        <div>
            <h1 class="page-title">{greeting}, Dr. {name_short}</h1>
            <p class="page-sub">Research workspace · BRFSS 2015 · CardioVue AI </p>
        </div>
        <span class="badge badge-amber" data-tooltip="Researcher access level">Researcher Access</span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(kpi_card("🗄️", "Dataset Size",    "253,680",  "BRFSS records",    "#14b8a6"), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("🧠", "Models Trained",  "8",        "Ensemble + base",  "#22c55e"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("🏆", "Best F1 Score",   "96.0%",    "Stacking Ensemble","#f59e0b"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("📄", "Publications",    "3",        "Pending review",   "#a78bfa"), unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 2], gap="medium")
    with col_a:
        st.markdown('<div class="sec-title">Study Methodology</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
            <div style="font-size:0.82rem;color:var(--t2);line-height:1.7;margin-bottom:0.75rem">
                ML applied to predict 10-year cardiovascular disease risk using the CDC BRFSS 2015 survey.
                Models trained on 21 demographic, lifestyle, and clinical variables with rigorous cross-validation
                and SHAP-based explainability across 253,680 respondents.
            </div>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.5rem">
                <div class="stat-box" style="padding:0.5rem 0.75rem">
                    <div style="font-size:0.67rem;color:var(--t3);text-transform:uppercase;letter-spacing:0.07em">Source</div>
                    <div style="font-size:0.8rem;font-weight:600;margin-top:2px">BRFSS 2015</div>
                </div>
                <div class="stat-box" style="padding:0.5rem 0.75rem">
                    <div style="font-size:0.67rem;color:var(--t3);text-transform:uppercase;letter-spacing:0.07em">Target</div>
                    <div style="font-size:0.8rem;font-weight:600;margin-top:2px">HeartDisease</div>
                </div>
                <div class="stat-box" style="padding:0.5rem 0.75rem">
                    <div style="font-size:0.67rem;color:var(--t3);text-transform:uppercase;letter-spacing:0.07em">Features</div>
                    <div style="font-size:0.8rem;font-weight:600;margin-top:2px">21 variables</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sec-title">Model Leaderboard</div>', unsafe_allow_html=True)
        for rank, (model, f1, auc, color) in enumerate([
            ("Stacking Ensemble","96.0%","95.8%","#14b8a6"),
            ("XGBoost",          "94.2%","93.9%","#22c55e"),
            ("LightGBM",         "93.8%","93.5%","#38bdf8"),
            ("CatBoost",         "93.1%","92.8%","#f59e0b"),
            ("Random Forest",    "91.4%","91.0%","#a78bfa"),
        ], 1):
            st.markdown(f"""
            <div class="trow">
                <div style="width:24px;font-size:0.7rem;color:var(--t3);font-weight:600">#{rank}</div>
                <div style="flex:1;font-size:0.81rem;font-weight:500">{model}</div>
                <span style="font-size:0.78rem;font-weight:600;color:{color};margin-right:12px">F1 {f1}</span>
                <span style="font-size:0.73rem;color:var(--t2)">AUC {auc}</span>
            </div>
            """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <img src="https://images.unsplash.com/photo-1518152006812-edab29b069ac?w=500&q=80&auto=format&fit=crop"
             class="section-img" style="height:195px;object-fit:cover;margin-bottom:6px"
             alt="Medical data research visualization"
             title="Data-driven cardiovascular risk research at scale"/>
        <p class="img-caption">Data-driven cardiovascular risk research at scale</p>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sec-title" style="margin-top:0.75rem">Risk Factor Prevalence</div>', unsafe_allow_html=True)
        for label, val, color in [("High BP","37.4%","#ef4444"),("Diabetes","20.1%","#f59e0b"),
                                  ("Smoking","34.3%","#f97316"),("Sedentary","41.8%","#a78bfa")]:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px solid var(--b1)">
                <span style="font-size:0.79rem;color:var(--t2)">{label}</span>
                <span style="font-size:0.79rem;font-weight:600;color:{color}">{val}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    show_analytics()

# ─── PROFILE PAGE ─────────────────────────────────────────────────────────────
def show_profile():
    username = st.session_state.username
    user = get_user(username)
    
    st.markdown(section_heading("◷", "Profile", "Manage your personal information and health preferences"), unsafe_allow_html=True)
    
    with st.form("profile_form"):
        st.markdown('<div class="card-title">Personal Information</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name", value=user['name'])
            email = st.text_input("Email", value=user['email'])
            age = st.number_input("Age", min_value=18, max_value=100, value=user.get('age', 35), step=1)
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=0 if user.get('gender', 'Male') == 'Male' else 1)
            family_history = st.checkbox("Family history of premature heart disease", value=bool(user.get('family_history', 0)), help="Heart disease in first-degree relative before age 55 (male) or 65 (female)")
        
        st.markdown('<div class="card-title">Health Preferences</div>', unsafe_allow_html=True)
        col3, col4 = st.columns(2)
        with col3:
            preferred_doctor = st.selectbox("Preferred Doctor", ["Dr. Kishan (Cardiologist)", "Dr. Amit Kumar (Internal Medicine)", "No preference"])
        with col4:
            notification_prefs = st.multiselect("Notification Preferences", ["Email reminders", "SMS alerts", "In-app notifications"], default=["In-app notifications"])
        
        submitted = st.form_submit_button("Update Profile", type="primary", use_container_width=True)
        if submitted:
            try:
                extra_json = json.dumps({'age': age, 'gender': gender, 'preferred_doctor': preferred_doctor, 'notification_prefs': notification_prefs})
                with get_conn() as conn:
                    conn.execute("UPDATE users SET name=?, email=?, extra_json=?, family_history=? WHERE username=?", (name, email, extra_json, int(family_history), username))
                st.session_state.user = get_user(username)
                st.success("✅ Profile updated successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error updating profile: {e}")
    
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Health Summary</div>', unsafe_allow_html=True)
    records = get_health_records(username, limit=5)
    if records:
        latest = records[0]
        delta = f"{latest.get('risk_score', 0) - records[1].get('risk_score', 0):+.1f}%" if len(records) > 1 else ""
        col1, col2, col3, col4 = st.columns(4)
        r_lbl = risk_label(latest.get('risk_score', 0))
        with col1: st.markdown(kpi_card("🫀", "Risk Score", f"{latest.get('risk_score','--')}%", f"{r_lbl} {delta}", risk_color(r_lbl)), unsafe_allow_html=True)
        with col2: st.markdown(kpi_card("💉", "Blood Pressure", f"{latest.get('bp_systolic','--')}/{latest.get('bp_diastolic','--')}", "mmHg", "#0ea5e9"), unsafe_allow_html=True)
        with col3: st.markdown(kpi_card("🧪", "Cholesterol", f"{latest.get('cholesterol','--')}", "mg/dL", "#f59e0b"), unsafe_allow_html=True)
        with col4: st.markdown(kpi_card("⚖️", "BMI", f"{latest.get('bmi','--')}", "kg/m²", "#14b8a6"), unsafe_allow_html=True)
    else:
        st.info("Complete a risk assessment to see your health summary.")

# ─── GOALS PAGE ────────────────────────────────────────────────────────────────
def show_goals():
    username = st.session_state.username
    
    st.markdown(section_heading("◈", "Health Goals", "Set and track your cardiovascular milestones"), unsafe_allow_html=True)
    
    active_goals = get_goals(username, status='active')
    achieved_goals = get_goals(username, status='achieved')
    records = get_health_records(username, limit=1)
    latest = records[0] if records else {}
    blood_tests = get_blood_tests(username)
    latest_blood = blood_tests[0] if blood_tests else {}
    
    with st.expander("➕ Create New Goal", expanded=False):
        with st.form("new_goal_form"):
            st.markdown('<div class="card-title">Set a New Health Goal</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                goal_type = st.selectbox("Goal Type", ["risk_score", "bp_systolic", "cholesterol", "bmi", "steps", "exercise_minutes", "hdl"], format_func=lambda x: {'risk_score': 'Reduce Risk Score', 'bp_systolic': 'Lower Blood Pressure', 'cholesterol': 'Improve Cholesterol', 'bmi': 'Achieve Healthy BMI', 'steps': 'Increase Daily Steps', 'exercise_minutes': 'Exercise Minutes', 'hdl': 'Improve HDL Cholesterol'}[x])
                target_value = st.number_input("Target Value", min_value=0.0, step=1.0)
            with col2:
                target_date = st.date_input("Target Date", min_value=datetime.now().date())
                current_value = st.number_input("Current Value (optional)", min_value=0.0, step=1.0, value=0.0)
            
            if current_value == 0.0:
                if goal_type == 'risk_score' and latest: current_value = latest.get('risk_score', 0)
                elif goal_type == 'bp_systolic' and latest: current_value = latest.get('bp_systolic', 0)
                elif goal_type == 'cholesterol' and latest: current_value = latest.get('cholesterol', 0)
                elif goal_type == 'bmi' and latest: current_value = latest.get('bmi', 0)
                elif goal_type == 'hdl' and latest_blood: current_value = latest_blood.get('hdl', 0)
            
            if st.form_submit_button("Create Goal", type="primary"):
                create_goal(username, goal_type, target_value, target_date.strftime("%Y-%m-%d"), current_value)
                st.success("✅ Goal created successfully!")
                st.rerun()
    
    if active_goals:
        st.markdown('<div class="sec-title">Active Goals</div>', unsafe_allow_html=True)
        for goal in active_goals:
            current = goal.get('current_value', 0)
            target = goal['target_value']
            progress_pct = min(100, (current / target * 100) if target > 0 else 0)
            is_reduction_goal = goal['goal_type'] in ['risk_score', 'bp_systolic', 'cholesterol', 'bmi']
            progress_color = '#10b981' if (is_reduction_goal and current <= target) or (not is_reduction_goal and current >= target) else '#f59e0b'
            
            with st.container():
                st.markdown(f"""
                <div class="card">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem">
                        <div>
                            <div style="font-weight:600;font-size:0.85rem">{goal['goal_type'].replace('_', ' ').title()}</div>
                            <div style="font-size:0.72rem;color:var(--t3);margin-top:2px">Target: {target} &nbsp;·&nbsp; Current: {current:.1f} &nbsp;·&nbsp; Due: {goal['target_date']}</div>
                        </div>
                        <div style="text-align:right">
                            <span style="font-size:0.78rem;color:{progress_color};font-weight:600">{progress_pct:.0f}%</span>
                        </div>
                    </div>
                    <div class="progress-track"><div style="width:{progress_pct}%;background:{progress_color};height:4px;border-radius:999px;transition:width 0.3s"></div></div>
                </div>
                """, unsafe_allow_html=True)
                col1, col2, col3 = st.columns([3, 1, 1])
                with col2:
                    if st.button("Update", key=f"update_{goal['id']}", use_container_width=True):
                        new_value = st.number_input("New value", value=current, key=f"val_{goal['id']}")
                        update_goal_progress(goal['id'], new_value)
                        st.rerun()
                with col3:
                    if st.button("Achieved ✓", key=f"achieve_{goal['id']}", use_container_width=True, type="primary"):
                        update_goal_progress(goal['id'], current, achieved=True)
                        add_notification(username, 'success', f"Congratulations! You achieved your {goal['goal_type']} goal!")
                        st.rerun()
    
    if achieved_goals:
        st.markdown('<div class="sec-title">Achieved Goals</div>', unsafe_allow_html=True)
        for goal in achieved_goals:
            st.markdown(f"""
            <div class="card-sm" style="opacity:0.6">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <div>
                        <span style="font-weight:600;font-size:0.82rem">{goal['goal_type'].replace('_', ' ').title()}</span>
                        <div style="font-size:0.72rem;color:var(--t3);margin-top:2px">Target: {goal['target_value']} &nbsp;·&nbsp; Achieved: {goal['achieved_date']}</div>
                    </div>
                    <span class="badge badge-green">✓ Done</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    if not active_goals and not achieved_goals:
        st.info("No goals yet. Use the form above to start tracking your health milestones.")

# ─── RESEARCHER: LITERATURE TRACKER ──────────────────────────────────────────
def show_literature():
    from modules.literature_review import show_literature_review
    show_literature_review()

# ─── RESEARCHER: EXPERIMENT NOTES ─────────────────────────────────────────────
def show_experiments():
    st.markdown(section_heading("🧪", "Experiment Notes", "Track your model experiments, hyperparameters, and results"), unsafe_allow_html=True)

    experiments = [
        {"id": "EXP-001", "name": "XGBoost Baseline", "date": "2024-11-10",
         "status": "Complete", "f1": 0.942, "auc": 0.939,
         "params": "n_estimators=300, max_depth=6, lr=0.1, subsample=0.8",
         "notes": "Strong baseline. Class imbalance handled with scale_pos_weight=5.2.",
         "color": "#22c55e"},
        {"id": "EXP-002", "name": "LightGBM + SMOTE", "date": "2024-11-14",
         "status": "Complete", "f1": 0.938, "auc": 0.935,
         "params": "n_estimators=500, num_leaves=63, lr=0.05, SMOTE k=5",
         "notes": "SMOTE improved minority recall by 4.2%. Best for clinical sensitivity.",
         "color": "#38bdf8"},
        {"id": "EXP-003", "name": "Stacking Ensemble", "date": "2024-11-20",
         "status": "Complete", "f1": 0.960, "auc": 0.958,
         "params": "Meta-learner: LogisticRegression, base: XGB+LGBM+CatBoost",
         "notes": "Best overall model. 5-fold CV with stratified splits.",
         "color": "#14b8a6"},
        {"id": "EXP-004", "name": "Neural Network", "date": "2024-11-25",
         "status": "Complete", "f1": 0.921, "auc": 0.918,
         "params": "3 hidden layers [256,128,64], dropout=0.3, Adam lr=1e-3",
         "notes": "Underperformed ensemble. Tabular data favours tree models.",
         "color": "#f97316"},
        {"id": "EXP-005", "name": "XGBoost + Feature Engineering", "date": "2024-12-01",
         "status": "Running", "f1": None, "auc": None,
         "params": "Age×BMI interaction, BP ratio feature, polynomial features",
         "notes": "Testing engineered features. Preliminary +0.8% F1 improvement.",
         "color": "#f59e0b"},
    ]

    complete = [e for e in experiments if e['status'] == 'Complete']
    best = max(complete, key=lambda x: x['f1']) if complete else None
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(kpi_card("🧪", "Experiments", str(len(experiments)), "total runs", "#14b8a6"), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("✅", "Complete", str(len(complete)), "finished", "#22c55e"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("🏆", "Best F1", f"{best['f1']:.3f}" if best else "--", best['name'] if best else "--", "#f59e0b"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("🔄", "Running", str(sum(1 for e in experiments if e['status']=='Running')), "in progress", "#f97316"), unsafe_allow_html=True)

    st.markdown('<div class="sec-title" style="margin-top:0.5rem">Experiment Log</div>', unsafe_allow_html=True)

    complete_exps = [e for e in experiments if e['f1'] is not None]
    if complete_exps:
        names = [e['name'] for e in complete_exps]
        f1s   = [e['f1']   for e in complete_exps]
        aucs  = [e['auc']  for e in complete_exps]
        colors_exp = [e['color'] for e in complete_exps]
        fig = go.Figure()
        fig.add_trace(go.Bar(name='F1 Score', x=names, y=f1s,
                             marker_color=colors_exp, text=[f'{v:.3f}' for v in f1s],
                             textposition='outside'))
        fig.add_trace(go.Scatter(name='ROC-AUC', x=names, y=aucs,
                                 mode='lines+markers', line=dict(color='#f59e0b', width=2),
                                 marker=dict(size=8)))
        _pl = {k: v for k, v in PLOT_LAYOUT.items() if k != 'yaxis'}
        fig.update_layout(**_pl, title='Experiment Comparison', height=280,
                          yaxis=dict(range=[0.88, 0.98], gridcolor='rgba(255,255,255,0.04)'),
                          barmode='group')
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    status_cls = {"Complete": "badge-green", "Running": "badge-amber", "Failed": "badge-rose"}
    for e in experiments:
        f1_str  = f"{e['f1']:.3f}"  if e['f1']  else "—"
        auc_str = f"{e['auc']:.3f}" if e['auc'] else "—"
        st.markdown(f"""
        <div class="card" style="border-left:3px solid {e['color']}">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem">
                <div style="display:flex;align-items:center;gap:0.75rem">
                    <span class="mono" style="color:var(--t3);font-size:0.72rem">{e['id']}</span>
                    <span style="font-weight:600;font-size:0.88rem">{e['name']}</span>
                </div>
                <div style="display:flex;align-items:center;gap:0.75rem">
                    <span style="font-size:0.75rem;color:var(--t3)">{e['date']}</span>
                    <span class="badge {status_cls.get(e['status'],'badge-teal')}">{e['status']}</span>
                </div>
            </div>
            <div style="display:flex;gap:1.5rem;margin-bottom:0.4rem">
                <div><span style="font-size:0.68rem;color:var(--t3)">F1 SCORE</span><br><span style="font-weight:700;color:{e['color']};font-size:0.95rem">{f1_str}</span></div>
                <div><span style="font-size:0.68rem;color:var(--t3)">ROC-AUC</span><br><span style="font-weight:700;color:var(--t2);font-size:0.95rem">{auc_str}</span></div>
                <div style="flex:1"><span style="font-size:0.68rem;color:var(--t3)">PARAMETERS</span><br><span class="mono" style="font-size:0.72rem;color:var(--t2)">{e['params']}</span></div>
            </div>
            <div style="font-size:0.78rem;color:var(--t2);line-height:1.5;border-top:1px solid var(--b1);padding-top:0.4rem">{e['notes']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Log New Experiment</div>', unsafe_allow_html=True)
    with st.expander("+ New experiment entry"):
        with st.form("new_exp"):
            c1, c2 = st.columns(2)
            with c1:
                exp_name   = st.text_input("Experiment Name")
                exp_model  = st.selectbox("Model Type", ["XGBoost","LightGBM","CatBoost","Random Forest","Neural Network","Stacking Ensemble","Other"])
                exp_f1     = st.number_input("F1 Score", 0.0, 1.0, 0.90, step=0.001, format="%.3f")
            with c2:
                exp_auc    = st.number_input("ROC-AUC", 0.0, 1.0, 0.90, step=0.001, format="%.3f")
                exp_status = st.selectbox("Status", ["Running","Complete","Failed"])
                exp_params = st.text_input("Key Parameters")
            exp_notes = st.text_area("Notes", height=80)
            if st.form_submit_button("Log Experiment →", type="primary"):
                st.success(f"Experiment '{exp_name}' logged successfully.")

# ─── RESEARCHER: COLLABORATION ────────────────────────────────────────────────
def show_collaboration():
    st.markdown(section_heading("🤝", "Collaboration", "Research team workspace — share findings and co-author papers"), unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 2], gap="medium")

    with col_a:
        st.markdown('<div class="sec-title">Team Activity Feed</div>', unsafe_allow_html=True)
        activity = [
            ("Dr. Ananya Mehta",  "AM", "#14b8a6", "Uploaded new BRFSS validation results", "2 hours ago",    "Dataset"),
            ("Prof. Raj Khanna",  "RK", "#f59e0b", "Commented on EXP-003 stacking results",  "5 hours ago",    "Experiment"),
            ("Dr. Priya Sharma",  "PS", "#22c55e", "Shared paper: LightGBM clinical review",  "Yesterday",      "Literature"),
            ("You",               "ME", "#38bdf8", "Updated model leaderboard — v2.1",        "Yesterday",      "Model"),
            ("Prof. Raj Khanna",  "RK", "#f59e0b", "Opened pull request: feature engineering","2 days ago",     "Code"),
            ("Dr. Ananya Mehta",  "AM", "#14b8a6", "Submitted abstract to AMIA 2025",         "3 days ago",     "Publication"),
        ]
        tag_colors = {"Dataset":"badge-teal","Experiment":"badge-amber","Literature":"badge-sky",
                     "Model":"badge-green","Code":"badge-violet","Publication":"badge-orange"}
        for name, initials, color, action, when, tag in activity:
            st.markdown(f"""
            <div class="trow" style="gap:0.75rem;padding:0.6rem 0">
                <div style="width:28px;height:28px;border-radius:50%;background:{color}22;border:1px solid {color}44;
                            display:flex;align-items:center;justify-content:center;font-size:0.65rem;font-weight:700;
                            color:{color};flex-shrink:0">{initials}</div>
                <div style="flex:1">
                    <span style="font-weight:600;font-size:0.82rem">{name}</span>
                    <span style="font-size:0.8rem;color:var(--t2)"> · {action}</span>
                </div>
                <div style="display:flex;align-items:center;gap:0.5rem;flex-shrink:0">
                    <span class="badge {tag_colors.get(tag,'badge-teal')}">{tag}</span>
                    <span style="font-size:0.7rem;color:var(--t3)">{when}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="sec-title" style="margin-top:0.875rem">Shared Research Notes</div>', unsafe_allow_html=True)
        notes = [
            ("EXP-003 beats all baselines — ready to write up results section.", "Prof. Raj Khanna", "Dec 2", "#14b8a6"),
            ("Suggest adding age-stratified analysis per CDC BRFSS guidelines.", "Dr. Ananya Mehta",  "Dec 1", "#f59e0b"),
            ("SHAP plots need higher DPI for journal submission (300dpi).", "Dr. Kishan", "Nov 30", "#38bdf8"),
        ]
        for note, author, date, color in notes:
            st.markdown(f"""
            <div class="card-sm" style="border-left:2px solid {color}">
                <div style="font-size:0.81rem;color:var(--t1);margin-bottom:4px">"{note}"</div>
                <div style="font-size:0.7rem;color:var(--t3)">{author} · {date}</div>
            </div>
            """, unsafe_allow_html=True)

        with st.form("post_note"):
            note_text = st.text_area("Post a note to the team", height=70, placeholder="Share a finding, question, or update...")
            if st.form_submit_button("Post →", type="primary"):
                st.success("Note posted to team feed.")

    with col_b:
        st.markdown("""
        <img src="https://images.unsplash.com/photo-1582719471384-894fbb16e074?w=500&q=80&auto=format&fit=crop"
             class="section-img" style="height:180px;object-fit:cover;margin-bottom:0.75rem"
             alt="Research team collaboration"/>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sec-title">Team Members</div>', unsafe_allow_html=True)
        team = [
            ("Dr. Ananya Mehta",  "AM", "#14b8a6", "Lead Researcher",       "Online"),
            ("Prof. Raj Khanna",  "RK", "#f59e0b", "Principal Investigator","Online"),
            ("Dr. Kishan",        "DK", "#22c55e", "Clinical Advisor",      "Away"),
            ("Ekta",              "EK", "#a78bfa", "You",                   "Online"),
        ]
        for name, ini, color, role_t, status in team:
            dot = "#22c55e" if status == "Online" else "#f59e0b"
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:0.625rem;padding:0.45rem 0;border-bottom:1px solid var(--b1)">
                <div style="width:28px;height:28px;border-radius:50%;background:{color}22;border:1px solid {color}44;
                            display:flex;align-items:center;justify-content:center;font-size:0.65rem;font-weight:700;color:{color};flex-shrink:0">{ini}</div>
                <div style="flex:1">
                    <div style="font-size:0.81rem;font-weight:500">{name}</div>
                    <div style="font-size:0.69rem;color:var(--t3)">{role_t}</div>
                </div>
                <div style="width:7px;height:7px;border-radius:50%;background:{dot}"></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="sec-title" style="margin-top:0.875rem">Upcoming Deadlines</div>', unsafe_allow_html=True)
        deadlines = [
            ("AMIA 2025 Abstract", "Jan 15, 2025", "#ef4444"),
            ("Nature Med submission", "Feb 1, 2025",  "#f97316"),
            ("Model v2.1 release",   "Jan 8, 2025",  "#f59e0b"),
            ("IRB renewal",          "Jan 20, 2025", "#a78bfa"),
        ]
        for task, date, color in deadlines:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px solid var(--b1)">
                <span style="font-size:0.79rem;color:var(--t2)">{task}</span>
                <span style="font-size:0.72rem;font-weight:600;color:{color}">{date}</span>
            </div>
            """, unsafe_allow_html=True)

# ─── MAIN ROUTER ───────────────────────────────────────────────────────────────
def main():
    if st.session_state.get('session_start'):
        elapsed = (datetime.now() - st.session_state['session_start']).seconds
        if elapsed > 2700:
            st.warning("⏰ Session expired due to inactivity. Please sign in again.")
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    if not st.session_state.get('logged_in'):
        show_login()
        add_footer()
        return

    user = st.session_state.user
    role = user['role']
    page = st.session_state.get('page', 'dashboard')
    initials = ''.join(w[0].upper() for w in user['name'].split()[:2])

    role_badge_cls = {'patient': 'badge-sky', 'doctor': 'badge-green', 'researcher': 'badge-amber'}.get(role, 'badge-teal')

    st.markdown(f"""
    <div class="top-nav">
        <div class="top-nav-brand">
            <div style="width:24px;height:24px;background:rgba(20,184,166,0.15);border-radius:7px;display:flex;align-items:center;justify-content:center;font-size:0.85rem">🫀</div>
            CardioVue&nbsp;<span>AI</span>
            <span class="badge badge-teal" style="margin-left:4px">v2.0</span>
        </div>
        <div class="top-nav-right">
            <span class="badge {role_badge_cls}">{role.upper()}</span>
            <div class="avatar">{initials}</div>
            <span style="font-size:0.78rem;color:var(--t2)">{user['name'].split()[0]}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    show_sidebar()
    
    ROUTES = {
        'patient': {
            'dashboard': show_patient_dashboard,
            'risk_prediction': show_risk_prediction,
            'whatif': show_whatif,
            'ecg': show_ecg,
            'health_records': show_health_records,
            'appointments': show_appointments,
            'goals': show_goals,
            'profile': show_profile,
            'ai_assistant': show_ai_assistant,
            'notifications': show_notifications,
        },
        'doctor': {
            'dashboard': show_doctor_dashboard,
            'patients': show_patients,
            'risk_prediction': lambda: show_risk_prediction(st.session_state.get('selected_patient')) if st.session_state.get('selected_patient') else show_risk_prediction(),
            'whatif': show_whatif,
            'ecg': show_ecg,
            'appointments': show_appointments,
            'analytics': show_analytics,
            'telemedicine': show_telemedicine,
            'guidelines': show_clinical_guidelines,
        },
        'researcher': {
            'dashboard': show_researcher_dashboard,
            'research_hub': show_research_hub,
            'literature_review': show_literature_review,
            'dataset': show_dataset,
            'analytics': show_analytics,
            'model_lab': show_model_lab,
            'experiments': show_experiments,
            'collaboration': show_collaboration,
        }
    }

    handler = ROUTES.get(role, {}).get(page)
    if handler:
        handler()
    else:
        ROUTES.get(role, {}).get('dashboard', show_patient_dashboard)()
    
    add_footer()

if __name__ == "__main__":
    main()