"""
CardioVue AI — Premium Dark Theme
Professional medical dashboard with stunning dark mode design
"""

PREMIUM_DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&display=swap');

/* ========================================
   PREMIUM DARK THEME - CARDIOVUE AI
   ======================================== */

:root {
    /* Core Colors */
    --bg-primary: #0a0c10;
    --bg-secondary: #0f1117;
    --bg-tertiary: #151821;
    --bg-card: #1a1d27;
    --bg-card-hover: #1f2330;
    --bg-elevated: #1e212c;
    
    /* Brand Colors */
    --primary: #00d4aa;
    --primary-light: #2ee5c0;
    --primary-dark: #00b894;
    --primary-glow: rgba(0, 212, 170, 0.25);
    
    /* Secondary */
    --secondary: #3b82f6;
    --secondary-light: #60a5fa;
    --secondary-dark: #2563eb;
    
    /* Accent */
    --accent: #f59e0b;
    --accent-light: #fbbf24;
    --purple: #8b5cf6;
    --pink: #ec4899;
    
    /* Status Colors */
    --success: #10b981;
    --success-light: #34d399;
    --warning: #f59e0b;
    --warning-light: #fbbf24;
    --danger: #ef4444;
    --danger-light: #f87171;
    --info: #06b6d4;
    --info-light: #22d3ee;
    
    /* Text Colors */
    --text-primary: #ffffff;
    --text-secondary: #a1a5b0;
    --text-tertiary: #6b6f7d;
    --text-muted: #4a4e5c;
    
    /* Border Colors */
    --border-light: rgba(255, 255, 255, 0.06);
    --border-medium: rgba(255, 255, 255, 0.1);
    --border-heavy: rgba(255, 255, 255, 0.15);
    
    /* Shadows */
    --shadow-sm: 0 2px 4px rgba(0, 0, 0, 0.1);
    --shadow-md: 0 4px 8px rgba(0, 0, 0, 0.12);
    --shadow-lg: 0 8px 24px rgba(0, 0, 0, 0.15);
    --shadow-xl: 0 12px 36px rgba(0, 0, 0, 0.2);
    --shadow-glow: 0 0 20px rgba(0, 212, 170, 0.15);
    --shadow-glow-lg: 0 0 40px rgba(0, 212, 170, 0.1);
    
    /* Spacing */
    --space-xs: 0.25rem;
    --space-sm: 0.5rem;
    --space-md: 1rem;
    --space-lg: 1.5rem;
    --space-xl: 2rem;
    --space-2xl: 3rem;
    
    /* Border Radius */
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 14px;
    --radius-xl: 18px;
    --radius-2xl: 24px;
    
    /* Transitions */
    --transition-fast: 150ms ease;
    --transition-normal: 250ms ease;
    --transition-slow: 350ms ease;
}

/* Base Styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html {
    scroll-behavior: smooth;
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
}

/* Streamlit Overrides */
.stApp {
    background: var(--bg-primary);
}

.stApp > header {
    background: transparent !important;
}

/* Hide default Streamlit elements */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px !important;
}

/* ========================================
   PREMIUM COMPONENTS
   ======================================== */

/* Hero Section */
.hero-premium {
    background: linear-gradient(135deg, #0a0c10 0%, #0f1117 50%, #151821 100%);
    border-radius: var(--radius-2xl);
    padding: 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    border: 1px solid var(--border-light);
}

.hero-premium::before {
    content: '';
    position: absolute;
    top: -30%;
    right: -10%;
    width: 50%;
    height: 150%;
    background: radial-gradient(circle, var(--primary-glow) 0%, transparent 70%);
    pointer-events: none;
}

.hero-premium::after {
    content: '';
    position: absolute;
    bottom: -30%;
    left: -10%;
    width: 40%;
    height: 120%;
    background: radial-gradient(circle, rgba(59, 130, 246, 0.08) 0%, transparent 70%);
    pointer-events: none;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(0, 212, 170, 0.1);
    border: 1px solid rgba(0, 212, 170, 0.2);
    border-radius: 100px;
    padding: 0.35rem 1rem;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--primary-light);
    margin-bottom: 1.5rem;
    letter-spacing: 0.02em;
}

.hero-title {
    font-size: 3.5rem;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.02em;
    margin-bottom: 1rem;
    background: linear-gradient(135deg, #ffffff 0%, #a1a5b0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-title em {
    background: linear-gradient(135deg, var(--primary-light), var(--secondary-light));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-style: normal;
}

.hero-subtitle {
    font-size: 1rem;
    color: var(--text-secondary);
    line-height: 1.6;
    max-width: 500px;
    margin-bottom: 2rem;
}

/* Premium Cards */
.card-premium {
    background: var(--bg-card);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-xl);
    padding: 1.5rem;
    transition: all var(--transition-normal);
    position: relative;
    overflow: hidden;
}

.card-premium:hover {
    border-color: var(--border-medium);
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}

.card-premium::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--primary), var(--secondary));
    opacity: 0;
    transition: opacity var(--transition-fast);
}

.card-premium:hover::before {
    opacity: 1;
}

.card-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.25rem;
}

.card-icon {
    width: 40px;
    height: 40px;
    background: rgba(0, 212, 170, 0.1);
    border-radius: var(--radius-lg);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.25rem;
}

.card-title {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-tertiary);
}

/* KPI Cards */
.kpi-premium {
    background: var(--bg-card);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-xl);
    padding: 1.25rem;
    transition: all var(--transition-normal);
    position: relative;
    overflow: hidden;
}

.kpi-premium:hover {
    transform: translateY(-3px);
    border-color: var(--border-medium);
    box-shadow: var(--shadow-glow);
}

.kpi-icon {
    width: 48px;
    height: 48px;
    background: linear-gradient(135deg, rgba(0, 212, 170, 0.15), rgba(59, 130, 246, 0.1));
    border-radius: var(--radius-lg);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    margin-bottom: 1rem;
}

.kpi-value {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 0.25rem;
    line-height: 1.2;
}

.kpi-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-tertiary);
}

.kpi-trend {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    font-size: 0.7rem;
    font-weight: 600;
    margin-top: 0.5rem;
    padding: 0.2rem 0.5rem;
    border-radius: 100px;
}

.trend-up {
    background: rgba(239, 68, 68, 0.1);
    color: #f87171;
}

.trend-down {
    background: rgba(16, 185, 129, 0.1);
    color: #34d399;
}

.trend-flat {
    background: rgba(107, 114, 128, 0.1);
    color: #9ca3af;
}

/* Glass Cards */
.glass-premium {
    background: rgba(26, 29, 39, 0.8);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-xl);
    padding: 1.5rem;
    transition: all var(--transition-normal);
}

.glass-premium:hover {
    background: rgba(26, 29, 39, 0.9);
    border-color: var(--border-medium);
}

/* Buttons */
.btn-premium {
    background: linear-gradient(135deg, var(--primary), var(--primary-dark));
    border: none;
    border-radius: var(--radius-lg);
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    font-size: 0.875rem;
    color: white;
    cursor: pointer;
    transition: all var(--transition-fast);
    position: relative;
    overflow: hidden;
}

.btn-premium:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(0, 212, 170, 0.3);
}

.btn-premium:active {
    transform: translateY(0);
}

.btn-outline-premium {
    background: transparent;
    border: 1px solid var(--border-medium);
    border-radius: var(--radius-lg);
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    font-size: 0.875rem;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all var(--transition-fast);
}

.btn-outline-premium:hover {
    border-color: var(--primary);
    color: var(--primary-light);
    background: rgba(0, 212, 170, 0.05);
}

/* Status Badges */
.badge-premium {
    display: inline-flex;
    align-items: center;
    gap: 0.375rem;
    padding: 0.25rem 0.75rem;
    border-radius: 100px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}

.badge-low {
    background: rgba(16, 185, 129, 0.1);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.2);
}

.badge-moderate {
    background: rgba(245, 158, 11, 0.1);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.2);
}

.badge-high {
    background: rgba(249, 115, 22, 0.1);
    color: #fdba74;
    border: 1px solid rgba(249, 115, 22, 0.2);
}

.badge-critical {
    background: rgba(239, 68, 68, 0.1);
    color: #fca5a5;
    border: 1px solid rgba(239, 68, 68, 0.2);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-light) !important;
}

[data-testid="stSidebar"] .stButton button {
    background: transparent;
    border: none;
    color: var(--text-secondary);
    font-weight: 500;
    border-radius: var(--radius-md);
    transition: all var(--transition-fast);
    text-align: left;
}

[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(0, 212, 170, 0.1);
    color: var(--primary-light);
    transform: translateX(4px);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: transparent;
    border-bottom: 1px solid var(--border-light);
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: var(--radius-md) var(--radius-md) 0 0;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    color: var(--text-tertiary);
    border: none;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-secondary);
}

.stTabs [aria-selected="true"] {
    background: transparent;
    color: var(--primary-light);
    border-bottom: 2px solid var(--primary);
}

/* Input Fields */
.stTextInput input, 
.stNumberInput input, 
.stTextArea textarea,
.stSelectbox [data-baseweb="select"] > div {
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    transition: all var(--transition-fast) !important;
}

.stTextInput input:focus, 
.stNumberInput input:focus, 
.stTextArea textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px var(--primary-glow) !important;
}

/* Metrics */
[data-testid="stMetricValue"] {
    font-weight: 700 !important;
    color: var(--primary-light) !important;
}

[data-testid="stMetricLabel"] {
    color: var(--text-tertiary) !important;
}

/* Dataframes */
.stDataFrame {
    border-radius: var(--radius-lg) !important;
    overflow: hidden !important;
}

/* Expanders */
.streamlit-expanderHeader {
    background: var(--bg-tertiary) !important;
    border-radius: var(--radius-md) !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
}

/* Divider */
.divider-premium {
    margin: 1.5rem 0;
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-light), transparent);
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}

::-webkit-scrollbar-track {
    background: var(--bg-tertiary);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: var(--primary);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--primary-light);
}

/* Animations */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.fade-in-up {
    animation: fadeInUp 0.5s ease forwards;
}

@keyframes pulse {
    0%, 100% {
        opacity: 1;
    }
    50% {
        opacity: 0.5;
    }
}

.pulse {
    animation: pulse 2s ease-in-out infinite;
}

/* Responsive */
@media (max-width: 768px) {
    .hero-title {
        font-size: 2rem;
    }
    
    .hero-premium {
        padding: 1.5rem;
    }
    
    .kpi-value {
        font-size: 1.5rem;
    }
    
    .card-premium {
        padding: 1rem;
    }
    
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
}

/* Print Styles */
@media print {
    .stApp header,
    .stApp footer,
    [data-testid="stSidebar"],
    .btn-premium {
        display: none !important;
    }
    
    .card-premium {
        break-inside: avoid;
        border: 1px solid #ccc;
    }
}
</style>
"""


def hero_premium(title: str, subtitle: str = None, badge: str = None):
    """Premium hero section"""
    badge_html = f'<div class="hero-badge">✨ {badge}</div>' if badge else ''
    sub_html = f'<p class="hero-subtitle">{subtitle}</p>' if subtitle else ''
    
    return f"""
    <div class="hero-premium fade-in-up">
        {badge_html}
        <h1 class="hero-title">{title}</h1>
        {sub_html}
    </div>
    """


def card_premium(title: str = None, icon: str = None, content: str = None):
    """Premium card component"""
    icon_html = f'<div class="card-icon">{icon}</div>' if icon else ''
    title_html = f'<div class="card-title">{title}</div>' if title else ''
    
    return f"""
    <div class="card-premium fade-in-up">
        <div class="card-header">
            {icon_html}
            {title_html}
        </div>
        <div>{content or ''}</div>
    </div>
    """


def kpi_premium(icon: str, label: str, value: str, trend: str = None, trend_dir: str = None):
    """Premium KPI card"""
    trend_html = ""
    if trend and trend_dir:
        arrow = "↑" if trend_dir == "up" else "↓" if trend_dir == "down" else "→"
        trend_class = "trend-up" if trend_dir == "up" else "trend-down" if trend_dir == "down" else "trend-flat"
        trend_html = f'<div class="kpi-trend {trend_class}">{arrow} {trend}</div>'
    
    return f"""
    <div class="kpi-premium fade-in-up">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {trend_html}
    </div>
    """


def badge_premium(risk_level: str):
    """Premium risk badge"""
    badge_map = {
        'Low': ('badge-low', '🟢'),
        'Moderate': ('badge-moderate', '🟡'),
        'High': ('badge-high', '🟠'),
        'Critical': ('badge-critical', '🔴')
    }
    badge_class, icon = badge_map.get(risk_level, ('badge-moderate', '⚪'))
    return f'<span class="badge-premium {badge_class}">{icon} {risk_level}</span>'


def glass_premium(content: str, padding: str = "1.5rem"):
    """Glass morphism container"""
    return f'<div class="glass-premium" style="padding: {padding};">{content}</div>'


def divider_premium():
    """Premium divider"""
    return '<div class="divider-premium"></div>'


def kpi_row_premium(metrics: list):
    """Row of KPI cards"""
    cards = ""
    for m in metrics:
        cards += f"""
        <div style="flex: 1; min-width: 0;">
            {kpi_premium(m['icon'], m['label'], m['value'], m.get('trend'), m.get('trend_dir'))}
        </div>
        """
    
    return f"""
    <div style="display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap;">
        {cards}
    </div>
    """


# Compatibility aliases
def kpi_modern(icon: str, label: str, value: str, trend: str = None, trend_dir: str = None, color: str = None):
    """Alias for kpi_premium"""
    return kpi_premium(icon, label, value, trend, trend_dir)


def badge_modern(risk_level: str):
    """Alias for badge_premium"""
    return badge_premium(risk_level)


def kpi_card_improved(label: str, value: str, icon: str, color: str = None):
    """Simple KPI card for compatibility"""
    return f"""
    <div class="kpi-premium">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value" style="color: {color or 'var(--primary)'};">{value}</div>
            </div>
            <div style="font-size: 2rem;">{icon}</div>
        </div>
    </div>
    """


# CSS variable for compatibility
ENHANCED_CSS = PREMIUM_DARK_CSS