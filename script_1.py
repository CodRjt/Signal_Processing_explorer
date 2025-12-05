# Create the main app.py file

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import use
use('Agg')  # Use non-interactive backend
import plotly.graph_objects as go
import plotly.express as px
from scipy import signal
from scipy.fft import fft, fftfreq
import time

# Page configuration
st.set_page_config(
    page_title="Interactive Signal Processing Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .module-description {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .footer {
        margin-top: 2rem;
        width: 100%;
        background-color: #0e1117;
        color: #fafafa;
        text-align: center;
        padding: 10px 0;
        font-size: 0.8rem;
    }
    .stSelectbox > div > div {
        background-color: #262730;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Interactive Signal Processing Explorer</h1>', 
                unsafe_allow_html=True)
    
    # Theme toggle in sidebar
    with st.sidebar:
        st.markdown("## 🎨 Theme")
        theme = st.selectbox("Choose Theme", ["Light", "Dark"], index=1)
        if theme == "Dark":
            st.markdown(""" 
                <style>
                    .main-header { color: #ffffff; }
                    .module-description { background-color: #262730; }
                    body { background-color: #0e1117; color: #fafafa; }
                </style>
                """, unsafe_allow_html=True)
        else:
            st.markdown(""" 
                <style>
                    .main-header { color: #1f77b4; }
                    .module-description { background-color: #f0f2f6; }
                    body { background-color: #ffffff; color: #000000; }
                </style>
                """, unsafe_allow_html=True)
        
        st.markdown("## 📚 Navigation")
        st.markdown("Select a module to explore different aspects of signal processing:")
        
        # Module descriptions
        modules_info = {
            "Home": "Overview of all modules and signal processing concepts",
            "DTMF Checker": "Interactive dual-tone multi-frequency keypad simulation",
            "Fingerprint Scanning": "Biometric signal processing visualization",
            "Sampling Rate Demo": "Nyquist theorem and aliasing demonstration",
            "Human Eye Sampling": "Visual perception as temporal sampling analogy",
            "Real-life DSP": "Practical digital signal processing examples"
        }
        
        for module, description in modules_info.items():
            st.markdown(f"**{module}**: {description}")
    
    # Main content area
    st.markdown("""
    ## 🎯 Welcome to Interactive Signal Processing
    
    This application provides hands-on exploration of fundamental digital signal processing concepts
    through interactive visualizations and real-world examples.
    
    ### 🚀 Featured Modules:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 📞 DTMF Checker
        - Interactive telephone keypad simulation
        - Dual-tone frequency generation and analysis
        - Time and frequency domain visualization
        - Real-time tone detection
        """)
        
        st.markdown("""
        #### 🔍 Fingerprint Scanning
        - Biometric signal acquisition simulation
        - Analog-to-digital conversion demonstration
        - Quantization effects visualization
        """)
        
        st.markdown("""
        #### 📊 Sampling Rate Visualization
        - Nyquist theorem demonstration
        - Aliasing effects in under-sampling
        - Reconstruction filter visualization
        """)
    
    with col2:
        st.markdown("""
        #### 👁️ Human Eye Sampling
        - Visual perception as discrete sampling
        - Frame rate effects on motion perception
        - Temporal sampling analogy
        """)
        
        st.markdown("""
        #### 🌍 Real-life DSP Applications
        - Audio quantization noise
        - Image processing effects
        - Signal reconstruction methods
        """)
    
  # Mathematical foundation
st.markdown("""
## 🧮 Mathematical Foundation

Signal processing is built on several key mathematical concepts:
""")

math_col1, math_col2 = st.columns(2)

with math_col1:
    st.markdown("**Discrete Fourier Transform:**")
    st.latex(r"X[k] = \sum_{n=0}^{N-1} x[n] e^{-j2\pi kn/N}")

    st.markdown("**Sampling Theorem (Nyquist):**")
    st.latex(r"f_s \geq 2f_{max}")

with math_col2:
    st.markdown("**Quantization:**")
    st.latex(r"x_q[n] = Q(x[n]) = \Delta \left\lfloor \frac{x[n]}{\Delta} + \frac{1}{2} \right\rfloor")

    st.markdown("**Z-Transform:**")
    st.latex(r"X(z) = \sum_{n=0}^{\infty} x[n] z^{-n}")

    
 # Getting started
st.markdown("""
## 🎮 Getting Started

1. **Navigate** using the sidebar to select any module  
2. **Interact** with the controls to modify parameters  
3. **Observe** how changes affect the visualizations  
4. **Learn** from the mathematical explanations provided  

Each module is designed to be self-contained while building upon core DSP concepts.
""")

# Footer
st.markdown("""
<div class="footer">
    Developed using Streamlit and Python by AI-generated design | © 2024 Interactive Signal Processing Explorer
</div>
""", unsafe_allow_html=True)


# Main entry point
if __name__ == "__main__":
    main()
    