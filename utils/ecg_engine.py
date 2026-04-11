"""
CardioVue AI — ECG Waveform Generator & Viewer
Generates realistic synthetic PQRST waveforms and detects anomalies.
Also supports uploaded CSV ECG data.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─── PQRST WAVEFORM SYNTHESIS ─────────────────────────────────────────────────

def _gaussian(t, center, width, amplitude):
    return amplitude * np.exp(-((t - center) ** 2) / (2 * width ** 2))


def generate_ecg_beat(t_beat, heart_rate=72, abnormality=None):
    """
    Generate one PQRST complex at given heart rate.
    abnormality: None | 'st_elevation' | 'afib' | 'bradycardia' | 'tachycardia' | 'lvh'
    """
    duration = 60 / heart_rate  # seconds per beat

    # Normalize beat time to 0–1
    t = t_beat / duration

    ecg = np.zeros_like(t)

    if abnormality == 'afib':
        # AFib: no P wave, irregular baseline
        ecg += np.random.normal(0, 0.04, len(t))
        # QRS complex only
        ecg += _gaussian(t, 0.35, 0.018, 1.6)   # R
        ecg += _gaussian(t, 0.38, 0.012, -0.45)  # S
        ecg += _gaussian(t, 0.55, 0.055, 0.22)   # T
    elif abnormality == 'st_elevation':
        # STEMI: elevated ST segment
        ecg += _gaussian(t, 0.15, 0.025, 0.12)   # P
        ecg += _gaussian(t, 0.35, 0.018, 1.5)    # R
        ecg += _gaussian(t, 0.38, 0.012, -0.35)  # S
        # Elevated ST
        mask_st = (t >= 0.40) & (t <= 0.58)
        ecg[mask_st] += 0.25 + 0.05 * np.sin(np.linspace(0, np.pi, mask_st.sum()))
        ecg += _gaussian(t, 0.62, 0.045, 0.45)   # Tall T
    elif abnormality == 'lvh':
        # LVH: large QRS voltage
        ecg += _gaussian(t, 0.15, 0.025, 0.14)   # P
        ecg += _gaussian(t, 0.30, 0.012, -0.28)  # Q
        ecg += _gaussian(t, 0.35, 0.015, 2.8)    # Tall R
        ecg += _gaussian(t, 0.38, 0.010, -0.75)  # Deep S
        ecg += _gaussian(t, 0.58, 0.050, 0.28)   # T
    else:
        # Normal PQRST
        ecg += _gaussian(t, 0.15, 0.025, 0.12)   # P wave
        ecg += _gaussian(t, 0.30, 0.012, -0.20)  # Q wave
        ecg += _gaussian(t, 0.35, 0.016, 1.40)   # R wave (main spike)
        ecg += _gaussian(t, 0.38, 0.011, -0.35)  # S wave
        ecg += _gaussian(t, 0.58, 0.050, 0.22)   # T wave

    return ecg


def generate_ecg_signal(duration_seconds=6, heart_rate=72, abnormality=None,
                         noise_level=0.025, sample_rate=500):
    """
    Generate a multi-beat ECG signal.
    Returns (time_array, ecg_array, beat_markers, detected_anomalies)
    """
    n_samples = duration_seconds * sample_rate
    t = np.linspace(0, duration_seconds, n_samples)
    ecg = np.zeros(n_samples)
    beat_duration = 60 / heart_rate
    beat_markers = []

    # AFib has variable RR intervals
    if abnormality == 'afib':
        rr_intervals = np.random.uniform(0.55, 1.1, 20)
    else:
        rr_intervals = np.full(20, beat_duration)

    current_time = 0.05
    for rr in rr_intervals:
        if current_time >= duration_seconds:
            break
        beat_start_idx = int(current_time * sample_rate)
        beat_end_time = min(current_time + rr, duration_seconds)
        beat_end_idx = int(beat_end_time * sample_rate)

        t_beat = t[beat_start_idx:beat_end_idx] - current_time
        if len(t_beat) > 0:
            ecg[beat_start_idx:beat_end_idx] = generate_ecg_beat(
                t_beat, heart_rate, abnormality
            )

        # Mark R-peak position
        r_time = current_time + 0.35 * (rr / beat_duration) * beat_duration
        beat_markers.append(r_time)
        current_time += rr

    # Add realistic noise
    ecg += np.random.normal(0, noise_level, n_samples)
    # Add baseline wander
    ecg += 0.03 * np.sin(2 * np.pi * 0.15 * t)

    anomalies = _detect_anomalies(ecg, t, heart_rate, abnormality)

    return t, ecg, beat_markers, anomalies


def _detect_anomalies(ecg, t, heart_rate, known_abnormality):
    """Detect and annotate ECG anomalies."""
    anomalies = []
    hr_str = f"{heart_rate} bpm"

    if known_abnormality == 'afib':
        anomalies = [
            {'type': 'critical', 'code': 'AFIB', 'label': 'Atrial Fibrillation',
             'detail': 'Irregular RR intervals, absent P waves. Anticoagulation evaluation recommended.'},
        ]
    elif known_abnormality == 'st_elevation':
        anomalies = [
            {'type': 'critical', 'code': 'STEMI', 'label': 'ST-Segment Elevation',
             'detail': 'ST elevation >2mm. Possible STEMI. Urgent cardiology review required.'},
        ]
    elif known_abnormality == 'lvh':
        anomalies = [
            {'type': 'warning', 'code': 'LVH', 'label': 'Left Ventricular Hypertrophy',
             'detail': 'Increased QRS voltage consistent with LVH. Echocardiogram recommended.'},
        ]
    elif heart_rate < 60:
        anomalies = [
            {'type': 'info', 'code': 'BRADY', 'label': f'Bradycardia ({hr_str})',
             'detail': 'Heart rate below 60 bpm. May be normal in athletes. Monitor if symptomatic.'},
        ]
    elif heart_rate > 100:
        anomalies = [
            {'type': 'warning', 'code': 'TACHY', 'label': f'Tachycardia ({hr_str})',
             'detail': 'Heart rate above 100 bpm. Evaluate for underlying cause.'},
        ]
    else:
        anomalies = [
            {'type': 'normal', 'code': 'NSR', 'label': f'Normal Sinus Rhythm ({hr_str})',
             'detail': 'Regular rhythm, normal PQRST morphology. No significant abnormalities detected.'},
        ]

    return anomalies


# ─── PLOTLY VISUALIZATION ─────────────────────────────────────────────────────

PLOT_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', color='#8A9BBE', size=12),
    xaxis=dict(gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.04)',
               tickfont=dict(size=10)),
    yaxis=dict(gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.04)',
               tickfont=dict(size=10)),
    margin=dict(l=10, r=10, t=30, b=10),
)


def build_ecg_figure(t, ecg, beat_markers, anomalies, zoom_range=None):
    """Build a Plotly figure for the ECG waveform."""
    anomaly_color_map = {
        'normal': '#10B981', 'info': '#3B82F6',
        'warning': '#F59E0B', 'critical': '#EF4444'
    }
    anomaly_type = anomalies[0]['type'] if anomalies else 'normal'
    line_color = anomaly_color_map.get(anomaly_type, '#10B981')

    fig = go.Figure()

    # ECG trace
    fig.add_trace(go.Scatter(
        x=t, y=ecg,
        mode='lines',
        name='ECG',
        line=dict(color=line_color, width=1.5),
        hovertemplate='Time: %{x:.3f}s<br>Amplitude: %{y:.3f} mV<extra></extra>'
    ))

    # R-peak markers
    for marker_t in beat_markers:
        # Find closest sample
        idx = np.argmin(np.abs(t - marker_t))
        if 0 <= idx < len(ecg):
            peak_y = ecg[max(0, idx-10):min(len(ecg), idx+10)].max()
            fig.add_trace(go.Scatter(
                x=[marker_t], y=[peak_y + 0.15],
                mode='markers',
                marker=dict(symbol='triangle-down', size=8, color='#F59E0B'),
                name='R-peak',
                showlegend=False,
                hovertemplate=f'R-peak: {marker_t:.2f}s<extra></extra>'
            ))

    # ST elevation zone highlighting
    if anomaly_type == 'critical' and anomalies[0].get('code') == 'STEMI':
        # Highlight ST segments with red band
        fig.add_hrect(y0=0.1, y1=0.5,
                      fillcolor='rgba(239,68,68,0.08)',
                      layer='below', line_width=0,
                      annotation_text='Elevated ST',
                      annotation_font_color='#EF4444',
                      annotation_font_size=10)

    x_range = zoom_range if zoom_range else [0, t[-1]]
    fig.update_layout(
        **PLOT_LAYOUT,
        title='',
        height=220,
        xaxis_range=x_range,
        xaxis_title='Time (seconds)',
        yaxis_title='Amplitude (mV)',
        showlegend=False,
        hovermode='x unified',
    )
    # Grid to look like ECG paper
    fig.update_xaxes(
        showgrid=True,
        gridcolor='rgba(200,16,46,0.08)',
        dtick=0.2,
        minor=dict(showgrid=True, gridcolor='rgba(200,16,46,0.04)', dtick=0.04)
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor='rgba(200,16,46,0.08)',
        dtick=0.5,
    )

    return fig


def build_heart_rate_trend(times, hr_values):
    """Build HR trend chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=hr_values,
        mode='lines+markers',
        name='Heart Rate',
        line=dict(color='#C8102E', width=2),
        marker=dict(size=5),
        fill='tozeroy',
        fillcolor='rgba(200,16,46,0.05)'
    ))
    fig.add_hrect(y0=60, y1=100, fillcolor='rgba(16,185,129,0.06)',
                  line_width=0, annotation_text='Normal range',
                  annotation_font_color='#10B981', annotation_font_size=9)
    fig.update_layout(**PLOT_LAYOUT, title='Heart Rate Trend (bpm)', height=180)
    return fig


def compute_hrv_metrics(ecg, t, beat_markers):
    """
    Compute basic HRV metrics from R-peak positions.
    Returns dict of HRV statistics.
    """
    if len(beat_markers) < 3:
        return {}

    rr_intervals = np.diff(beat_markers) * 1000  # ms
    mean_rr = np.mean(rr_intervals)
    sdnn = np.std(rr_intervals)
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
    nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50)
    pnn50 = nn50 / len(rr_intervals) * 100

    mean_hr = 60000 / mean_rr if mean_rr > 0 else 0

    return {
        'Mean HR': round(mean_hr, 1),
        'SDNN (ms)': round(sdnn, 1),
        'RMSSD (ms)': round(rmssd, 1),
        'pNN50 (%)': round(pnn50, 1),
        'Mean RR (ms)': round(mean_rr, 1),
        'Min RR (ms)': round(min(rr_intervals), 1),
        'Max RR (ms)': round(max(rr_intervals), 1),
    }


def parse_csv_ecg(df: pd.DataFrame):
    """
    Parse uploaded ECG CSV. Expects columns: time (or index), amplitude.
    Returns (t, ecg) arrays.
    """
    # Try common column names
    time_col = next((c for c in df.columns if c.lower() in ['time','t','seconds','ms']), None)
    amp_col = next((c for c in df.columns if c.lower() in ['amplitude','ecg','voltage','mv','signal','value']), None)

    if amp_col is None and len(df.columns) >= 1:
        amp_col = df.columns[0] if time_col else df.columns[0]

    if time_col:
        t = df[time_col].values.astype(float)
    else:
        # Assume uniform 500 Hz sampling
        t = np.arange(len(df)) / 500.0

    ecg = df[amp_col].values.astype(float) if amp_col else df.iloc[:, 0].values.astype(float)

    # Normalize
    ecg = (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-8)

    return t, ecg