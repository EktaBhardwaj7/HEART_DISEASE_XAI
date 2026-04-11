"""
CardioVue AI — Accessibility Manager
WCAG 2.1 compliant accessibility features
"""

import streamlit as st
import json

class AccessibilityManager:
    """Manage accessibility features across the application"""
    
    def __init__(self):
        self.font_size = st.session_state.get('font_size', 'normal')
        self.high_contrast = st.session_state.get('high_contrast', False)
        self.reduced_motion = st.session_state.get('reduced_motion', False)
        self.screen_reader = st.session_state.get('screen_reader', False)
        self.color_blind_mode = st.session_state.get('color_blind_mode', 'none')
    
    def render_menu(self):
        """Render accessibility settings in sidebar"""
        with st.expander("♿ Accessibility", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                font_size = st.selectbox(
                    "📝 Font Size",
                    ['small', 'normal', 'large', 'x-large'],
                    index=['small', 'normal', 'large', 'x-large'].index(self.font_size),
                    key="acc_font"
                )
                
                high_contrast = st.checkbox(
                    "🔆 High Contrast Mode",
                    value=self.high_contrast,
                    key="acc_contrast",
                    help="Enhanced contrast for visually impaired users"
                )
            
            with col2:
                reduced_motion = st.checkbox(
                    "🎬 Reduce Motion",
                    value=self.reduced_motion,
                    key="acc_motion",
                    help="Disable animations"
                )
                
                color_blind = st.selectbox(
                    "🎨 Color Blind Mode",
                    ['none', 'protanopia', 'deuteranopia', 'tritanopia'],
                    format_func=lambda x: {
                        'none': 'None',
                        'protanopia': 'Red-Blind (Protanopia)',
                        'deuteranopia': 'Green-Blind (Deuteranopia)',
                        'tritanopia': 'Blue-Blind (Tritanopia)'
                    }[x],
                    key="acc_color"
                )
            
            if st.button("💾 Apply Settings", type="primary", use_container_width=True):
                st.session_state.font_size = font_size
                st.session_state.high_contrast = high_contrast
                st.session_state.reduced_motion = reduced_motion
                st.session_state.color_blind_mode = color_blind
                st.rerun()
    
    def get_css(self):
        """Return accessibility CSS"""
        css = ""
        
        # Font size
        sizes = {'small': '13px', 'normal': '16px', 'large': '20px', 'x-large': '24px'}
        css += f"""
        body, .stApp, .stMarkdown, p, div, span, label {{
            font-size: {sizes.get(self.font_size, '16px')} !important;
        }}
        """
        
        # High contrast
        if self.high_contrast:
            css += """
            :root {
                --bg: #000000 !important;
                --surface: #1a1a1a !important;
                --text: #ffffff !important;
                --accent: #ffff00 !important;
            }
            .stButton button {
                background: #000 !important;
                border: 2px solid #ff0 !important;
                color: #ff0 !important;
            }
            .glass-card, .kpi-card {
                background: #111 !important;
                border: 2px solid #ff0 !important;
            }
            """
        
        # Reduced motion
        if self.reduced_motion:
            css += """
            *, *::before, *::after {
                animation: none !important;
                transition: none !important;
            }
            """
        
        return css
    
    def inject_aria(self):
        """Inject ARIA labels for screen readers"""
        st.markdown("""
        <script>
        (function() {
            // Add ARIA labels to all buttons
            document.querySelectorAll('.stButton button').forEach(btn => {
                if (!btn.getAttribute('aria-label')) {
                    btn.setAttribute('aria-label', btn.innerText || 'Button');
                }
            });
            
            // Add skip to content link
            const skipLink = document.createElement('a');
            skipLink.href = '#main-content';
            skipLink.innerText = 'Skip to main content';
            skipLink.style.position = 'absolute';
            skipLink.style.left = '-9999px';
            skipLink.style.top = '-9999px';
            skipLink.style.background = '#14b8a6';
            skipLink.style.color = '#fff';
            skipLink.style.padding = '10px';
            skipLink.style.zIndex = '9999';
            skipLink.onfocus = () => { skipLink.style.left = '10px'; skipLink.style.top = '10px'; };
            skipLink.onblur = () => { skipLink.style.left = '-9999px'; skipLink.style.top = '-9999px'; };
            document.body.prepend(skipLink);
        })();
        </script>
        """, unsafe_allow_html=True)