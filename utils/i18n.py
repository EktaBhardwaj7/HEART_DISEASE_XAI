"""
CardioVue AI — Internationalization
Multi-language support with RTL for heart health content
"""

import streamlit as st
from typing import Dict

class I18nManager:
    """Internationalization manager with cardiovascular focus"""
    
    TRANSLATIONS = {
        'en': {
            # Navigation
            'dashboard': 'Your Heart Health',
            'risk_assessment': 'See Your Future Risk',
            'whatif': 'Try Lifestyle Changes',
            'ecg': 'Live Heart Activity',
            'profile': 'My Heart Profile',
            
            # Metrics
            'risk_score': 'Heart Risk Score',
            'blood_pressure': 'Blood Pressure',
            'cholesterol': 'Cholesterol',
            'heart_rate': 'Heart Rate',
            'bmi': 'BMI',
            
            # Actions
            'analyze': 'Analyze My Heart',
            'save': 'Save to My Heart Record',
            'share': 'Share with Doctor',
            'download': 'Download Heart Report',
            
            # Status
            'low_risk': 'Healthy Heart',
            'moderate_risk': 'Watch Your Heart',
            'high_risk': 'Protect Your Heart',
            'critical_risk': 'Heart Needs Attention',
            
            # Insights
            'ai_insight': 'Heart Health Insight',
            'recommendation': 'For Your Heart',
        },
        'es': {
            'dashboard': 'Tu Salud Cardíaca',
            'risk_assessment': 'Tu Riesgo Futuro',
            'whatif': 'Prueba Cambios de Vida',
            'ecg': 'Actividad Cardíaca en Vivo',
            'profile': 'Mi Perfil Cardíaco',
            'risk_score': 'Puntuación de Riesgo Cardíaco',
            'blood_pressure': 'Presión Arterial',
            'cholesterol': 'Colesterol',
            'analyze': 'Analizar Mi Corazón',
            'low_risk': 'Corazón Saludable',
            'moderate_risk': 'Cuida tu Corazón',
            'high_risk': 'Protege tu Corazón',
            'ai_insight': 'Información de Salud Cardíaca',
        },
        'hi': {  # Hindi
            'dashboard': 'आपका हृदय स्वास्थ्य',
            'risk_assessment': 'अपना भविष्य का जोखिम देखें',
            'whatif': 'जीवनशैली बदलाव आजमाएं',
            'ecg': 'लाइव हृदय गतिविधि',
            'profile': 'मेरी हृदय प्रोफाइल',
            'risk_score': 'हृदय जोखिम स्कोर',
            'blood_pressure': 'रक्तचाप',
            'cholesterol': 'कोलेस्ट्रॉल',
            'analyze': 'मेरे दिल का विश्लेषण करें',
            'low_risk': 'स्वस्थ हृदय',
            'moderate_risk': 'अपने दिल का ख्याल रखें',
            'high_risk': 'अपने दिल की रक्षा करें',
            'ai_insight': 'हृदय स्वास्थ्य अंतर्दृष्टि',
        },
        'zh': {  # Chinese
            'dashboard': '您的心脏健康',
            'risk_assessment': '查看您的未来风险',
            'whatif': '尝试生活方式改变',
            'ecg': '实时心脏活动',
            'profile': '我的心脏档案',
            'risk_score': '心脏风险评分',
            'blood_pressure': '血压',
            'cholesterol': '胆固醇',
            'analyze': '分析我的心脏',
            'low_risk': '健康心脏',
            'moderate_risk': '关注您的心脏',
            'high_risk': '保护您的心脏',
            'ai_insight': '心脏健康洞察',
        }
    }
    
    RTL_LANGUAGES = ['ar', 'he', 'fa', 'ur']
    
    def __init__(self):
        self.current_lang = st.session_state.get('language', 'en')
    
    def render_selector(self):
        """Render language selector with heart icons"""
        languages = {
            'en': '🇺🇸 English (Heart Health)',
            'es': '🇪🇸 Español (Salud del Corazón)',
            'hi': '🇮🇳 हिन्दी (हृदय स्वास्थ्य)',
            'zh': '🇨🇳 中文 (心脏健康)',
            'ar': '🇸🇦 العربية (صحة القلب)',
        }
        
        with st.expander("🌐 Language / भाषा / 语言", expanded=False):
            col1, col2 = st.columns([3, 1])
            with col1:
                selected = st.selectbox(
                    "Choose your language",
                    options=list(languages.keys()),
                    format_func=lambda x: languages[x],
                    index=list(languages.keys()).index(self.current_lang),
                    key="lang_selector"
                )
            with col2:
                st.markdown("❤️", unsafe_allow_html=True)
            
            if selected != self.current_lang:
                st.session_state.language = selected
                self.current_lang = selected
                st.rerun()
    
    def get(self, key: str) -> str:
        """Get translated text"""
        translations = self.TRANSLATIONS.get(self.current_lang, self.TRANSLATIONS['en'])
        return translations.get(key, key)
    
    def get_rtl_css(self) -> str:
        """Get RTL CSS for right-to-left languages"""
        if self.current_lang in self.RTL_LANGUAGES:
            return """
            <style>
            body, .stApp, .stMarkdown {
                direction: rtl;
                text-align: right;
            }
            .stButton button, .stSelectbox, .stTextInput {
                direction: rtl;
            }
            .kpi-card, .glass-card {
                text-align: right;
            }
            </style>
            """
        return ""
    
    def format_date(self, date_obj) -> str:
        """Format date according to locale"""
        formats = {
            'en': '%B %d, %Y',
            'es': '%d de %B de %Y',
            'hi': '%d %B %Y',
            'zh': '%Y年%m月%d日',
        }
        fmt = formats.get(self.current_lang, '%B %d, %Y')
        return date_obj.strftime(fmt)

def translate_ui():
    """Apply translations to UI elements"""
    i18n = I18nManager()
    
    # Apply RTL if needed
    st.markdown(i18n.get_rtl_css(), unsafe_allow_html=True)
    
    return i18n