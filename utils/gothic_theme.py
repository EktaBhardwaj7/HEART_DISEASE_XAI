"""
CardioVue AI — Gothic Dark Academia Theme
Anatomical Surrealism meets Vintage Medical Aesthetic
"""

GOTHIC_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700;800;900&family=IM+Fell+English:ital@0;1&family=Special+Elite&family=Crimson+Text:wght@400;600;700&family=La+Belle+Aurore&display=swap');

/* ========================================
   GOTHIC DARK ACADEMIA THEME
   "Floral Necrosis" - Vintage Medical Aesthetic
   ======================================== */

:root {
    /* Parchment & Bone Colors */
    --parchment-light: #F4ECD8;
    --parchment-dark: #E8E2D0;
    --parchment-ink: #D4C9A8;
    --aged-paper: #DFD7BF;
    --antique-bone: #C4B896;
    
    /* Ink Colors */
    --charcoal-ink: #2C2C2B;
    --faded-ink: #4A4A48;
    --sepia-ink: #5C5346;
    --midnight-blue: #1A2A36;
    
    /* Anatomical Accents */
    --dried-blood: #8B0000;
    --deep-terracotta: #A52A2A;
    --crimson-stain: #6B1A1A;
    --heart-crimson: #9B1D2C;
    
    /* Organic Accents */
    --forest-moss: #4A5D4E;
    --deep-leaf: #2D5A27;
    --withered-vine: #6B5B4B;
    --poison-ivy: #3A5C3C;
    
    /* Botanical Gradients */
    --gradient-blood-moss: linear-gradient(135deg, #8B0000 0%, #2D5A27 100%);
    --gradient-parchment: linear-gradient(135deg, #F4ECD8 0%, #E8E2D0 50%, #DFD7BF 100%);
    --gradient-antique: linear-gradient(135deg, #E8E2D0 0%, #C4B896 100%);
    
    /* Textures */
    --paper-texture: repeating-linear-gradient(45deg, rgba(0,0,0,0.02) 0px, rgba(0,0,0,0.02) 2px, transparent 2px, transparent 8px);
    --grain-noise: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200"><filter id="noise"><feTurbulence type="fractalNoise" baseFrequency="0.65" numOctaves="3" stitchTiles="stitch"/></filter><rect width="100%" height="100%" filter="url(%23noise)" opacity="0.04"/></svg>');
    
    /* Shadows */
    --shadow-vintage: 0 4px 20px rgba(0, 0, 0, 0.08);
    --shadow-inset: inset 0 2px 4px rgba(0, 0, 0, 0.04);
}

/* Base Gothic Styles */
.stApp {
    background: var(--parchment-light);
    background-image: var(--paper-texture), var(--grain-noise);
    font-family: 'Crimson Text', serif;
    color: var(--charcoal-ink);
}

/* Hide default Streamlit elements */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* Gothic Typography */
h1, h2, h3, h4, .gothic-title, .hero-title {
    font-family: 'Playfair Display', serif;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--charcoal-ink);
}

.gothic-subtitle {
    font-family: 'IM Fell English', serif;
    font-style: italic;
    color: var(--sepia-ink);
}

.gothic-body {
    font-family: 'Crimson Text', serif;
    font-size: 1rem;
    line-height: 1.6;
    color: var(--faded-ink);
}

.gothic-caption {
    font-family: 'La Belle Aurore', cursive;
    font-size: 0.85rem;
    color: var(--withered-vine);
}

.gothic-mono {
    font-family: 'Special Elite', monospace;
    font-size: 0.8rem;
    color: var(--dried-blood);
}

/* Vintage Card */
.card-gothic {
    background: var(--parchment-dark);
    border: 1px solid var(--antique-bone);
    border-radius: 0px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow-vintage);
    position: relative;
    transition: all 0.3s ease;
}

.card-gothic::before {
    content: '';
    position: absolute;
    top: 8px;
    left: 8px;
    right: 8px;
    bottom: 8px;
    border: 1px dashed rgba(139, 0, 0, 0.15);
    pointer-events: none;
}

.card-gothic:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
}

/* Hero Section - Vintage Book Style */
.hero-gothic {
    background: var(--gradient-parchment);
    border: 2px solid var(--antique-bone);
    padding: 3rem;
    margin-bottom: 2rem;
    position: relative;
    text-align: center;
}

.hero-gothic::before {
    content: '⚚';
    position: absolute;
    top: -15px;
    left: 50%;
    transform: translateX(-50%);
    background: var(--parchment-light);
    padding: 0 1rem;
    color: var(--dried-blood);
    font-family: 'Special Elite', monospace;
    font-size: 1.2rem;
}

.hero-gothic::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 10%;
    right: 10%;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--dried-blood), var(--forest-moss), transparent);
}

.hero-title-gothic {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 900;
    color: var(--charcoal-ink);
    letter-spacing: -0.03em;
}

.hero-title-gothic em {
    font-style: italic;
    color: var(--dried-blood);
    font-family: 'IM Fell English', serif;
}

/* Image Containers - Vintage Specimen Style */
.image-specimen {
    background: var(--parchment-dark);
    border: 1px solid var(--antique-bone);
    padding: 1rem;
    position: relative;
    margin: 1rem 0;
    box-shadow: var(--shadow-vintage);
}

.image-specimen::after {
    content: 'Specimen — Fig. ' attr(data-fig);
    position: absolute;
    bottom: -10px;
    right: 10px;
    background: var(--parchment-light);
    padding: 0 0.5rem;
    font-family: 'Special Elite', monospace;
    font-size: 0.7rem;
    color: var(--dried-blood);
}

.image-container {
    width: 100%;
    overflow: hidden;
    position: relative;
}

.image-container img {
    width: 100%;
    height: auto;
    display: block;
    filter: sepia(0.2) contrast(1.05) brightness(0.98);
    transition: all 0.5s ease;
}

.image-container:hover img {
    filter: sepia(0) contrast(1.1);
    transform: scale(1.02);
}

.image-caption {
    font-family: 'La Belle Aurore', cursive;
    font-size: 0.75rem;
    color: var(--withered-vine);
    text-align: center;
    margin-top: 0.5rem;
    font-style: italic;
}

/* Vintage Badges */
.badge-gothic {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.25rem 1rem;
    background: var(--parchment-ink);
    border: 1px solid var(--antique-bone);
    font-family: 'Special Elite', monospace;
    font-size: 0.7rem;
    color: var(--dried-blood);
    letter-spacing: 1px;
}

/* KPI Cards - Vintage Ledger Style */
.kpi-gothic {
    background: var(--parchment-dark);
    border: 1px solid var(--antique-bone);
    padding: 1rem;
    text-align: center;
    position: relative;
    transition: all 0.3s ease;
}

.kpi-gothic:hover {
    background: var(--parchment-light);
    border-color: var(--dried-blood);
}

.kpi-icon-gothic {
    font-size: 2rem;
    margin-bottom: 0.5rem;
    filter: drop-shadow(2px 2px 4px rgba(0,0,0,0.1));
}

.kpi-value-gothic {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 800;
    color: var(--dried-blood);
    letter-spacing: -0.02em;
}

.kpi-label-gothic {
    font-family: 'Special Elite', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--withered-vine);
}

/* Botanical Dividers */
.divider-gothic {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1rem;
    margin: 2rem 0;
    color: var(--dried-blood);
    font-family: 'Special Elite', monospace;
    font-size: 0.8rem;
}

.divider-gothic::before,
.divider-gothic::after {
    content: '✧';
    color: var(--forest-moss);
    font-size: 0.8rem;
}

.divider-gothic-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--antique-bone), transparent);
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background: var(--parchment-dark) !important;
    border-right: 1px solid var(--antique-bone) !important;
    background-image: var(--paper-texture);
}

[data-testid="stSidebar"] .stButton button {
    background: transparent;
    border: 1px solid var(--antique-bone);
    color: var(--charcoal-ink);
    font-family: 'Special Elite', monospace;
    transition: all 0.3s ease;
}

[data-testid="stSidebar"] .stButton button:hover {
    border-color: var(--dried-blood);
    color: var(--dried-blood);
    background: rgba(139, 0, 0, 0.05);
}

/* Buttons */
.btn-gothic {
    background: transparent;
    border: 2px solid var(--dried-blood);
    color: var(--dried-blood);
    padding: 0.5rem 1.5rem;
    font-family: 'Special Elite', monospace;
    font-size: 0.8rem;
    letter-spacing: 2px;
    cursor: pointer;
    transition: all 0.3s ease;
}

.btn-gothic:hover {
    background: var(--dried-blood);
    color: var(--parchment-light);
}

/* Input Fields */
.stTextInput input, .stNumberInput input, .stTextArea textarea {
    background: var(--parchment-light) !important;
    border: 1px solid var(--antique-bone) !important;
    border-radius: 0px !important;
    font-family: 'Crimson Text', serif !important;
    color: var(--charcoal-ink) !important;
}

.stTextInput input:focus {
    border-color: var(--dried-blood) !important;
    box-shadow: 0 0 0 2px rgba(139, 0, 0, 0.1) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid var(--antique-bone);
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 0;
    padding: 0.5rem 1.5rem;
    font-family: 'Special Elite', monospace;
    color: var(--withered-vine);
    border: none;
}

.stTabs [aria-selected="true"] {
    color: var(--dried-blood);
    border-bottom: 2px solid var(--dried-blood);
}

/* Alert Boxes - Gothic Style */
.alert-gothic {
    border-left: 4px solid;
    padding: 1rem;
    margin: 1rem 0;
    background: var(--parchment-light);
    font-family: 'Crimson Text', serif;
}

.alert-info-gothic {
    border-left-color: var(--forest-moss);
}

.alert-warning-gothic {
    border-left-color: var(--dried-blood);
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 6px;
}

::-webkit-scrollbar-track {
    background: var(--parchment-ink);
}

::-webkit-scrollbar-thumb {
    background: var(--dried-blood);
    border-radius: 0;
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.fade-in {
    animation: fadeIn 0.5s ease forwards;
}

/* Responsive */
@media (max-width: 768px) {
    .hero-title-gothic {
        font-size: 2rem;
    }
    .hero-gothic {
        padding: 1.5rem;
    }
}
</style>
"""


def image_specimen(image_url: str, caption: str = None, fig_num: int = None):
    """Create a vintage specimen image container"""
    fig_attr = f'data-fig="{fig_num}"' if fig_num else ''
    caption_html = f'<div class="image-caption">{caption}</div>' if caption else ''
    return f"""
    <div class="image-specimen" {fig_attr}>
        <div class="image-container">
            <img src="{image_url}" alt="{caption or 'Anatomical specimen'}">
        </div>
        {caption_html}
    </div>
    """


def image_placeholder_gothic(title: str, description: str = None, fig_num: int = None):
    """Placeholder for user images - Gothic specimen style"""
    fig_attr = f'data-fig="{fig_num}"' if fig_num else ''
    desc_html = f'<div class="gothic-caption" style="margin-top: 0.5rem;">{description}</div>' if description else ''
    return f"""
    <div class="image-specimen" {fig_attr}>
        <div class="image-container" style="min-height: 200px; background: var(--parchment-ink); display: flex; align-items: center; justify-content: center; flex-direction: column; padding: 2rem;">
            <span style="font-size: 3rem; opacity: 0.5;">⚚</span>
            <p style="font-family: 'Special Elite', monospace; color: var(--dried-blood); margin-top: 1rem; text-align: center;">
                {title}
            </p>
        </div>
        {desc_html}
    </div>
    """


def kpi_gothic(icon: str, label: str, value: str, trend: str = None):
    """Gothic-style KPI card"""
    trend_html = f'<div style="font-size: 0.7rem; margin-top: 0.25rem; color: var(--forest-moss);">{trend}</div>' if trend else ''
    return f"""
    <div class="kpi-gothic fade-in">
        <div class="kpi-icon-gothic">{icon}</div>
        <div class="kpi-value-gothic">{value}</div>
        <div class="kpi-label-gothic">{label}</div>
        {trend_html}
    </div>
    """


def badge_gothic(text: str, type: str = "default"):
    """Gothic-style badge"""
    return f'<span class="badge-gothic">{text}</span>'


def divider_gothic():
    """Botanical-style divider"""
    return """
    <div class="divider-gothic">
        <span>✧</span>
        <span class="divider-gothic-line"></span>
        <span>⚚</span>
        <span class="divider-gothic-line"></span>
        <span>✧</span>
    </div>
    """


def hero_gothic(title: str, subtitle: str = None):
    """Gothic hero section"""
    sub_html = f'<p class="gothic-subtitle" style="font-size: 1.1rem; margin-top: 0.5rem;">{subtitle}</p>' if subtitle else ''
    return f"""
    <div class="hero-gothic fade-in">
        <h1 class="hero-title-gothic">{title}</h1>
        {sub_html}
    </div>
    """


def alert_gothic(message: str, type: str = "info"):
    """Gothic-style alert"""
    type_class = "alert-info-gothic" if type == "info" else "alert-warning-gothic"
    return f"""
    <div class="alert-gothic {type_class}">
        <span style="font-family: 'Special Elite', monospace;">⚚</span> {message}
    </div>
    """


# Placeholder image URLs for different categories (replace with your actual images)
IMAGE_PLACEHOLDERS = {
    "heart_anatomy": "https://images.unsplash.com/photo-1576091160550-2173dba999ef?w=400&q=80",
    "ecg_waveform": "https://images.unsplash.com/photo-1579684385127-1ef15d508118?w=400&q=80",
    "medical_research": "https://images.unsplash.com/photo-1518152006812-edab29b069ac?w=400&q=80",
    "patient_monitoring": "https://images.unsplash.com/photo-1628348068343-c6a848d2b6dd?w=400&q=80",
    "doctor_consultation": "https://images.unsplash.com/photo-1584036561566-baf8f5f1b144?w=400&q=80",
    "lab_equipment": "https://images.unsplash.com/photo-1576086213369-97a306d36557?w=400&q=80",
    "heart_3d": "https://images.unsplash.com/photo-1628177142898-93e36e4e0a50?w=400&q=80",
    "stethoscope": "https://images.unsplash.com/photo-1584466977773-e625c92cddbe?w=400&q=80",
}

# Compatibility aliases
ENHANCED_CSS = GOTHIC_CSS
kpi_modern = kpi_gothic
badge_modern = badge_gothic