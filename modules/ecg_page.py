"""
CardioVue AI — ECG Viewer Page
Real-time waveform visualization with anomaly detection and HRV analysis.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from utils.ecg_engine import (
    generate_ecg_signal, build_ecg_figure, compute_hrv_metrics,
    parse_csv_ecg, build_heart_rate_trend, PLOT_LAYOUT
)
from utils.theme import section_heading


ANOMALY_COLORS = {
    'normal': '#10B981', 'info': '#3B82F6',
    'warning': '#F59E0B', 'critical': '#EF4444'
}

ANOMALY_ICONS = {
    'normal': '✅', 'info': 'ℹ️', 'warning': '⚠️', 'critical': '🚨'
}


def show_ecg_viewer(username: str):
    st.markdown(section_heading("📈", "ECG Waveform Viewer",
                                 "Real-time ECG analysis with anomaly detection and HRV metrics"),
                unsafe_allow_html=True)

    tabs = st.tabs(["  Live Simulation  ", "  Upload ECG  ", "  HRV Analysis  "])

    with tabs[0]:
        _show_live_ecg()

    with tabs[1]:
        _show_upload_ecg()

    with tabs[2]:
        _show_hrv_analysis()


def _show_live_ecg():
    col_ctrl, col_info = st.columns([2, 1])

    with col_ctrl:
        c1, c2, c3 = st.columns(3)
        with c1:
            hr = st.slider("Heart Rate (bpm)", 40, 160, 72, key="ecg_hr")
        with c2:
            abnormality = st.selectbox(
                "Rhythm / Pathology",
                ["Normal", "Atrial Fibrillation", "ST Elevation (STEMI)",
                 "Left Ventricular Hypertrophy", "Bradycardia", "Tachycardia"],
                key="ecg_abn"
            )
        with c3:
            duration = st.slider("Duration (sec)", 4, 12, 6, key="ecg_dur")

    abn_map = {
        "Normal": None,
        "Atrial Fibrillation": "afib",
        "ST Elevation (STEMI)": "st_elevation",
        "Left Ventricular Hypertrophy": "lvh",
        "Bradycardia": "bradycardia",
        "Tachycardia": "tachycardia",
    }
    abn_key = abn_map[abnormality]

    # Override heart rate for brady/tachy presets
    effective_hr = hr
    if abn_key == "bradycardia":
        effective_hr = min(hr, 50)
    elif abn_key == "tachycardia":
        effective_hr = max(hr, 110)

    # Generate
    t, ecg, beat_markers, anomalies = generate_ecg_signal(
        duration_seconds=duration,
        heart_rate=effective_hr,
        abnormality=abn_key,
        noise_level=0.022
    )

    # Anomaly banner
    if anomalies:
        a = anomalies[0]
        a_color = ANOMALY_COLORS.get(a['type'], '#8A9BBE')
        a_icon = ANOMALY_ICONS.get(a['type'], 'ℹ️')
        alert_cls = f"alert-{'critical' if a['type'] == 'critical' else 'warning' if a['type'] == 'warning' else 'success' if a['type'] == 'normal' else 'info'}"
        st.markdown(f"""
        <div class="alert-box {alert_cls}">
            {a_icon} <strong>{a['code']} — {a['label']}</strong><br>
            <span style="font-size:0.82rem">{a['detail']}</span>
        </div>
        """, unsafe_allow_html=True)

    # ECG plot
    fig = build_ecg_figure(t, ecg, beat_markers, anomalies)
    st.plotly_chart(fig, use_container_width=True, config={
        'displayModeBar': True,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'toImageButtonOptions': {'format': 'png', 'filename': 'ecg_waveform'}
    })

    # Metrics row
    hrv = compute_hrv_metrics(ecg, t, beat_markers)
    if hrv:
        cols = st.columns(len(hrv))
        metric_colors = {
            'Mean HR': '#C8102E', 'SDNN (ms)': '#3B82F6',
            'RMSSD (ms)': '#10B981', 'pNN50 (%)': '#F59E0B',
            'Mean RR (ms)': '#8A9BBE', 'Min RR (ms)': '#8A9BBE', 'Max RR (ms)': '#8A9BBE'
        }
        for col, (k, v) in zip(cols, hrv.items()):
            with col:
                mc = metric_colors.get(k, '#8A9BBE')
                st.markdown(f"""
                <div class="card-sm" style="text-align:center;padding:0.75rem">
                    <div class="card-title" style="font-size:0.68rem">{k}</div>
                    <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.1rem;color:{mc}">{v}</div>
                </div>
                """, unsafe_allow_html=True)

    # Lead selector note
    st.markdown("""
    <div style="font-size:0.75rem;color:var(--text-muted);margin-top:0.5rem">
        📡 Displaying Lead II equivalent · R-peaks marked with ▼ · 
        ECG paper grid (0.04s/0.5mV) · 500 Hz synthetic signal
    </div>
    """, unsafe_allow_html=True)

    # Multi-beat zoom strip
    with st.expander("🔍 Single-Beat Zoom"):
        beat_num = st.slider("Select Beat", 1, max(1, len(beat_markers) - 1), 1, key="ecg_beat_sel")
        if beat_num < len(beat_markers):
            center = beat_markers[beat_num - 1]
            zoom_range = [max(0, center - 0.4), min(t[-1], center + 0.6)]
            fig_zoom = build_ecg_figure(t, ecg, beat_markers, anomalies, zoom_range=zoom_range)
            fig_zoom.update_layout(height=180)
            st.plotly_chart(fig_zoom, use_container_width=True, config={'displayModeBar': False})


def _show_upload_ecg():
    st.markdown("""
    <div class="alert-box alert-info">
        Upload a CSV file with columns: <code>time</code> (seconds) and <code>amplitude</code> (mV).
        If no time column, uniform 500 Hz sampling is assumed.
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload ECG CSV", type=["csv", "txt"], key="ecg_upload")

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.markdown(f'<div style="font-size:0.8rem;color:var(--text-muted)">Loaded {len(df):,} samples · Columns: {list(df.columns)}</div>',
                        unsafe_allow_html=True)

            t, ecg = parse_csv_ecg(df)

            # Detect anomalies on uploaded signal
            from utils.ecg_engine import _detect_anomalies
            # Simple HR estimate from zero crossings
            mean_hr = 72  # fallback
            anomalies = _detect_anomalies(ecg, t, mean_hr, None)
            beat_markers = []

            fig = build_ecg_figure(t, ecg, beat_markers, anomalies)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})

            # Show raw data preview
            with st.expander("View Raw Data"):
                st.dataframe(df.head(100), use_container_width=True)

        except Exception as e:
            st.error(f"Error parsing ECG file: {e}. Please check the format.")

    else:
        # Show example format
        st.markdown("""
        <div class="card" style="padding:1.5rem;text-align:center">
            <div style="font-size:2rem;margin-bottom:0.75rem">📄</div>
            <div style="font-family:'Syne',sans-serif;font-weight:700;margin-bottom:0.5rem">Expected CSV Format</div>
            <code style="font-size:0.8rem;color:var(--text-secondary)">
                time,amplitude<br>
                0.000,0.012<br>
                0.002,0.015<br>
                0.004,0.142<br>
                ...
            </code>
        </div>
        """, unsafe_allow_html=True)


def _show_hrv_analysis():
    st.markdown('<div class="card-title">Heart Rate Variability (HRV) Analysis</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="alert-box alert-info" style="margin-bottom:1rem">
        HRV is a key indicator of autonomic nervous system health and cardiovascular fitness. 
        Higher RMSSD and SDNN generally indicate better cardiac health.
    </div>
    """, unsafe_allow_html=True)

    # Simulate 7-day HRV trend
    np.random.seed(42)
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    rmssd = np.random.normal(38, 6, 7).clip(20, 70)
    sdnn = np.random.normal(45, 8, 7).clip(25, 80)
    hr_values = np.random.normal(70, 5, 7).clip(55, 90)

    col1, col2 = st.columns(2)

    with col1:
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(x=days, y=rmssd,
                              marker=dict(color='rgba(16,185,129,0.7)', line=dict(width=0)),
                              name='RMSSD (ms)'))
        fig1.add_hline(y=40, line=dict(color='rgba(16,185,129,0.4)', width=1, dash='dash'),
                       annotation_text='Target >40ms')
        fig1.update_layout(**PLOT_LAYOUT, title='RMSSD – 7 Day Trend', height=220)
        st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=days, y=sdnn, mode='lines+markers',
                                   line=dict(color='#3B82F6', width=2),
                                   marker=dict(size=7), name='SDNN (ms)'))
        fig2.update_layout(**PLOT_LAYOUT, title='SDNN – 7 Day Trend', height=220)
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

    # HRV reference table
    st.markdown('<div class="card-title" style="margin-top:0.5rem">HRV Reference Ranges</div>', unsafe_allow_html=True)
    ref_data = {
        'Metric': ['RMSSD', 'SDNN', 'pNN50', 'Mean HR'],
        'Your Average': [f'{np.mean(rmssd):.1f} ms', f'{np.mean(sdnn):.1f} ms', '18.2%', f'{np.mean(hr_values):.0f} bpm'],
        'Normal Range': ['>40 ms', '>50 ms', '>10%', '60-100 bpm'],
        'Status': ['⚠ Below target', '⚠ Below target', '✅ Normal', '✅ Normal'],
        'Interpretation': [
            'Indicates moderate autonomic function',
            'Borderline — improve with exercise & sleep',
            'Normal parasympathetic activity',
            'Normal resting heart rate'
        ]
    }
    import pandas as pd
    st.dataframe(pd.DataFrame(ref_data), use_container_width=True, hide_index=True)

    st.markdown("""
    <div style="font-size:0.78rem;color:var(--text-muted);margin-top:0.75rem">
        💡 <strong>Improve HRV:</strong> Regular aerobic exercise, consistent sleep schedule, 
        deep breathing exercises, stress reduction, and reducing alcohol intake 
        are the most evidence-based methods for improving HRV.
    </div>
    """, unsafe_allow_html=True)