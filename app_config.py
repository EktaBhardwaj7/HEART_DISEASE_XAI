# app_config.py - Enhanced configuration with PWA support

import streamlit as st
import json

def setup_pwa():
    """Add Progressive Web App support"""
    
    pwa_config = """
    <link rel="manifest" href="/manifest.json">
    <meta name="theme-color" content="#0f172a">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="CardioVue AI">
    <link rel="apple-touch-icon" href="/apple-touch-icon.png">
    
    <script>
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('/sw.js')
            .then(reg => console.log('SW registered:', reg))
            .catch(err => console.log('SW registration failed:', err));
    }
    </script>
    """
    st.markdown(pwa_config, unsafe_allow_html=True)

def responsive_css():
    """Enhanced responsive CSS for all devices"""
    return """
    <style>
    /* Responsive breakpoints */
    @media (max-width: 768px) {
        /* Mobile styles */
        .main .block-container {
            padding: 0.75rem !important;
        }
        
        .page-header {
            flex-direction: column;
            align-items: flex-start !important;
            gap: 0.5rem;
        }
        
        .kpi-grid {
            grid-template-columns: repeat(2, 1fr) !important;
            gap: 0.75rem;
        }
        
        .card, .card-sm {
            padding: 0.875rem !important;
        }
        
        h1.page-title {
            font-size: 1.5rem !important;
        }
        
        .top-nav {
            padding: 0.5rem 0.75rem !important;
        }
        
        .top-nav-brand {
            font-size: 0.9rem;
        }
        
        .top-nav-right {
            gap: 0.5rem !important;
        }
        
        .avatar {
            width: 28px;
            height: 28px;
            font-size: 0.7rem;
        }
        
        /* Make tables scrollable on mobile */
        .stDataFrame {
            overflow-x: auto;
        }
        
        /* Touch-friendly buttons */
        .stButton button {
            min-height: 44px;
            min-width: 44px;
        }
        
        /* Stack columns on mobile */
        .row-widget.stHorizontal {
            flex-direction: column !important;
        }
        
        /* Responsive charts */
        .js-plotly-plot {
            height: 300px !important;
        }
    }
    
    @media (min-width: 769px) and (max-width: 1024px) {
        /* Tablet styles */
        .kpi-grid {
            grid-template-columns: repeat(2, 1fr) !important;
        }
        
        .main .block-container {
            padding: 1rem !important;
        }
    }
    
    @media (min-width: 1025px) {
        /* Desktop styles */
        .main .block-container {
            padding: 1.5rem !important;
        }
    }
    
    /* Touch device optimizations */
    @media (hover: none) and (pointer: coarse) {
        .stButton button, 
        .element-container button,
        [data-testid="stSelectbox"] select {
            min-height: 48px;
        }
        
        .card, .card-sm {
            transition: none;
        }
        
        /* Disable hover effects on touch */
        .card:hover {
            transform: none;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        :root {
            --bg: #0f172a;
            --surface: #1e293b;
            --text-primary: #f1f5f9;
            --text-secondary: #cbd5e1;
        }
    }
    
    /* Light mode */
    @media (prefers-color-scheme: light) {
        :root {
            --bg: #f8fafc;
            --surface: #ffffff;
            --text-primary: #0f172a;
            --text-secondary: #475569;
        }
    }
    
    /* Reduced motion for accessibility */
    @media (prefers-reduced-motion: reduce) {
        * {
            animation-duration: 0.01ms !important;
            transition-duration: 0.01ms !important;
        }
    }
    </style>
    """