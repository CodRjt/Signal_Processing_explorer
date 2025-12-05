import streamlit as st
import numpy as np
from matplotlib import use
use('Agg')

# Page configuration
st.set_page_config(
    page_title="Interactive Signal Processing Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with modern design
st.markdown("""
<style>
    /* Main title with animated gradient */
    .hero-title {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        animation: gradient-shift 3s ease infinite;
        background-size: 200% 200%;
    }
    
    @keyframes gradient-shift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .hero-subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.3rem;
        margin-bottom: 3rem;
        font-weight: 400;
    }
    
    /* Module cards with hover effects */
    .module-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        border: 2px solid transparent;
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    .module-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        transform: scaleX(0);
        transition: transform 0.4s ease;
    }
    
    .module-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 24px rgba(102,126,234,0.2);
        border-color: #667eea;
    }
    
    .module-card:hover::before {
        transform: scaleX(1);
    }
    
    .module-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .module-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.75rem;
    }
    
    .module-description {
        color: #6b7280;
        line-height: 1.6;
        font-size: 1rem;
    }
    
    .module-tags {
        margin-top: 1rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    
    .module-tag {
        background: linear-gradient(135deg, #667eea15, #764ba215);
        color: #667eea;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid #667eea30;
    }
    
    /* Feature boxes */
    .feature-box {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-left: 4px solid #3b82f6;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .feature-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1e40af;
        margin-bottom: 0.5rem;
    }
    
    .feature-text {
        color: #1e3a8a;
        line-height: 1.6;
    }
    
    /* Math boxes */
    .math-card {
        background: white;
        border: 2px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        margin: 1rem 0;
    }
    
    .math-card:hover {
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102,126,234,0.15);
        transform: translateY(-4px);
    }
    
    .math-label {
        color: #6b7280;
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Info sections */
    .info-section {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 2px solid #f59e0b;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 2rem 0;
    }
    
    .info-section strong {
        color: #92400e;
    }
    
    .info-section ul {
        color: #78350f;
        margin-top: 0.5rem;
    }
    
    /* Quick start guide */
    .quick-start {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 2rem 0;
    }
    
    .quick-start-title {
        color: #065f46;
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .quick-start-steps {
        color: #064e3b;
        line-height: 2;
    }
    
    .step-number {
        display: inline-block;
        width: 32px;
        height: 32px;
        background: #10b981;
        color: white;
        border-radius: 50%;
        text-align: center;
        line-height: 32px;
        font-weight: 700;
        margin-right: 0.5rem;
    }
    
    /* Stats section */
    .stats-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .stat-box {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 2px solid #e5e7eb;
        transition: all 0.3s ease;
    }
    
    .stat-box:hover {
        border-color: #667eea;
        transform: scale(1.05);
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-label {
        color: #6b7280;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Footer */
    .footer {
        margin-top: 4rem;
        padding: 2rem;
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        color: #d1d5db;
        text-align: center;
        border-radius: 12px;
    }
    
    .footer-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #f3f4f6;
        margin-bottom: 0.5rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
        margin: 2rem 0 1.5rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid #667eea;
        position: relative;
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        bottom: -3px;
        left: 0;
        width: 100px;
        height: 3px;
        background: #764ba2;
    }
    
    /* Sidebar styling */
    .sidebar-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1f2937;
        margin: 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }
    
    .sidebar-module {
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        background: #f9fafb;
        border-left: 3px solid #667eea;
        transition: all 0.2s ease;
    }
    
    .sidebar-module:hover {
        background: #f3f4f6;
        padding-left: 1rem;
    }
    
    .sidebar-module-title {
        font-weight: 700;
        color: #374151;
    }
    
    .sidebar-module-desc {
        font-size: 0.85rem;
        color: #6b7280;
        margin-top: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Hero Section
    st.markdown('<h1 class="hero-title">📊 Interactive Signal Processing Explorer</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Master Digital Signal Processing Through Interactive Visualization</p>', 
                unsafe_allow_html=True)
    
    # Quick stats
    st.markdown("""
    <div class="stats-container">
        <div class="stat-box">
            <div class="stat-value">6</div>
            <div class="stat-label">Interactive Modules</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">∞</div>
            <div class="stat-label">Learning Possibilities</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">100%</div>
            <div class="stat-label">Hands-On Experience</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">Real-Time</div>
            <div class="stat-label">Interactive Feedback</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Welcome message
    st.markdown("""
    <div class="info-section">
        <strong>🎯 Welcome!</strong><br>
        This platform provides an immersive learning experience for <strong>Digital Signal Processing (DSP)</strong> 
        concepts through interactive visualizations and real-world applications. Whether you're a student, 
        educator, or professional, explore fundamental concepts at your own pace with immediate visual feedback.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar content
    with st.sidebar:
        st.markdown('<div class="sidebar-header">📚 Module Navigator</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: #eff6ff; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border: 2px solid #3b82f6;">
            <strong style="color: #1e40af;">💡 Quick Tip:</strong><br>
            <span style="color: #1e3a8a; font-size: 0.9rem;">
            Click on any module card below to navigate. Each module is self-contained and interactive!
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        # Module quick links
        modules_info = [
            ("📞", "DTMF Checker", "Telephone tone generation & analysis"),
            ("🔍", "Fingerprint Scanning", "Biometric signal processing"),
            ("📊", "Sampling Rate Demo", "Nyquist theorem & aliasing"),
            ("👁️", "Human Eye Sampling", "Visual perception as sampling"),
            ("🌍", "Real-life DSP", "Practical applications"),
            ("🛝", "Image Playhouse", "Image processing playground"),
        ]
        
        for icon, name, desc in modules_info:
            st.markdown(f"""
            <div class="sidebar-module">
                <div class="sidebar-module-title">{icon} {name}</div>
                <div class="sidebar-module-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        <div style="background: #fef3c7; padding: 1rem; border-radius: 8px; border: 2px solid #f59e0b;">
            <strong style="color: #92400e;">🎨 Platform Features:</strong><br>
            <span style="color: #78350f; font-size: 0.85rem;">
            ✓ Real-time parameter adjustment<br>
            ✓ Interactive visualizations<br>
            ✓ Mathematical foundations<br>
            ✓ Theory & practical examples<br>
            ✓ Mobile-responsive design
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content - Module cards
    st.markdown('<div class="section-header">🚀 Explore Interactive Modules</div>', 
                unsafe_allow_html=True)
    
    # Row 1 - DTMF and Fingerprint
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="module-card">
            <span class="module-icon">📞</span>
            <div class="module-title">DTMF Checker</div>
            <div class="module-description">
                Experience how touch-tone telephones work! Generate and analyze dual-tone 
                multi-frequency signals used in telephony. Visualize both time and frequency 
                domains, detect tones in real-time, and understand the mathematics behind 
                phone keypads.
            </div>
            <div class="module-tags">
                <span class="module-tag">Frequency Analysis</span>
                <span class="module-tag">FFT</span>
                <span class="module-tag">Tone Generation</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="module-card">
            <span class="module-icon">🔍</span>
            <div class="module-title">Fingerprint Scanning</div>
            <div class="module-description">
                Simulate the complete biometric signal processing pipeline from analog capture 
                to digital enhancement. Explore analog-to-digital conversion, quantization 
                effects, spatial sampling, and signal enhancement techniques used in modern 
                fingerprint scanners.
            </div>
            <div class="module-tags">
                <span class="module-tag">ADC</span>
                <span class="module-tag">Quantization</span>
                <span class="module-tag">Image Processing</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Row 2 - Sampling and Eye
    col3, col4 = st.columns(2, gap="large")
    
    with col3:
        st.markdown("""
        <div class="module-card">
            <span class="module-icon">📊</span>
            <div class="module-title">Sampling Rate Visualization</div>
            <div class="module-description">
                Master the Nyquist-Shannon sampling theorem through interactive demonstrations. 
                Observe aliasing effects, experiment with different reconstruction methods, 
                and understand why sampling rate matters in digital signal processing.
            </div>
            <div class="module-tags">
                <span class="module-tag">Nyquist Theorem</span>
                <span class="module-tag">Aliasing</span>
                <span class="module-tag">Reconstruction</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="module-card">
            <span class="module-icon">👁️</span>
            <div class="module-title">Human Eye Sampling</div>
            <div class="module-description">
                Discover the fascinating parallel between human vision and digital sampling! 
                Explore how our eyes process motion at discrete intervals, understand the 
                Critical Flicker Fusion frequency, and see why frame rates matter in displays 
                and cinema.
            </div>
            <div class="module-tags">
                <span class="module-tag">Visual Perception</span>
                <span class="module-tag">Frame Rate</span>
                <span class="module-tag">Temporal Sampling</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Row 3 - Real-life DSP (centered)
    col5, col6 = st.columns(2 , gap="large")
    with col5:
        st.markdown("""
        <div class="module-card">
            <span class="module-icon">🌍</span>
            <div class="module-title">Real-life DSP Applications</div>
            <div class="module-description">
                Connect theory to practice with real-world DSP examples. Explore audio 
                quantization noise, image processing effects, signal reconstruction methods, 
                and see how DSP is used in everyday technology from smartphones to medical devices.
            </div>
            <div class="module-tags">
                <span class="module-tag">Audio Processing</span>
                <span class="module-tag">Image Effects</span>
                <span class="module-tag">Practical DSP</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        st.markdown("""
        <div class="module-card">
            <span class="module-icon">🛝</span>
            <div class="module-title">Image Playhouse</div>
            <div class="module-description">
                Dive into the world of image processing! Upload your own images and experiment 
                with edge detection, frequency filtering, and transformations. Understand how 
                DSP techniques enhance and manipulate visual data.
            </div>
            <div class="module-tags">
                <span class="module-tag">Image Upload</span>
                <span class="module-tag">Edge Detection</span>
                <span class="module-tag">Frequency Filtering</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    # Mathematical Foundation Section
    st.markdown('<div class="section-header">🧮 Mathematical Foundation</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-box">
        <div class="feature-title">Core DSP Concepts</div>
        <div class="feature-text">
            Digital Signal Processing is built on rigorous mathematical principles. 
            Below are the fundamental equations that power every module in this platform. 
            Understanding these will give you deep insight into how digital systems process signals.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    math_col1, math_col2 = st.columns(2, gap="large")
    
    with math_col1:
        st.markdown('<div class="math-card">', unsafe_allow_html=True)
        st.latex(r"X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j2\pi kn/N}")
        st.markdown('<div class="math-label">Discrete Fourier Transform (DFT)</div>', 
                   unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption("Converts time-domain signals to frequency domain")
        
        st.markdown('<div class="math-card">', unsafe_allow_html=True)
        st.latex(r"f_s \geq 2f_{max}")
        st.markdown('<div class="math-label">Nyquist-Shannon Sampling Theorem</div>', 
                   unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption("Minimum sampling rate to avoid information loss")
    
    with math_col2:
        st.markdown('<div class="math-card">', unsafe_allow_html=True)
        st.latex(r"x_q[n] = Q(x[n]) = \Delta \left\lfloor \frac{x[n]}{\Delta} + \frac{1}{2} \right\rfloor")
        st.markdown('<div class="math-label">Quantization Formula</div>', 
                   unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption("Converts continuous amplitude to discrete levels")
        
        st.markdown('<div class="math-card">', unsafe_allow_html=True)
        st.latex(r"X(z) = \sum_{n=0}^{\infty} x[n] \cdot z^{-n}")
        st.markdown('<div class="math-label">Z-Transform</div>', 
                   unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption("Analyzes discrete-time systems in complex domain")
    
    # Getting Started Guide
    st.markdown('<div class="section-header">🎮 Getting Started</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="quick-start">
        <div class="quick-start-title">Your Learning Journey Begins Here</div>
        <div class="quick-start-steps">
            <div><span class="step-number">1</span> <strong>Choose a Module</strong> from the cards above or sidebar navigation</div>
            <div><span class="step-number">2</span> <strong>Interact with Controls</strong> using sliders, buttons, and input fields</div>
            <div><span class="step-number">3</span> <strong>Observe Real-time Changes</strong> in visualizations and metrics</div>
            <div><span class="step-number">4</span> <strong>Read Theory Sections</strong> to understand the mathematics behind the magic</div>
            <div><span class="step-number">5</span> <strong>Experiment Freely</strong> - there's no wrong way to explore!</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    feature_col1, feature_col2, feature_col3 = st.columns(3)
    
    with feature_col1:
        st.markdown("""
        <div class="feature-box">
            <div class="feature-title">🎯 Interactive Learning</div>
            <div class="feature-text">
                Every parameter is adjustable in real-time. See immediate results and 
                build intuition through experimentation.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_col2:
        st.markdown("""
        <div class="feature-box">
            <div class="feature-title">📈 Visual Feedback</div>
            <div class="feature-text">
                High-quality plots and animations help you understand complex concepts 
                through visualization.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_col3:
        st.markdown("""
        <div class="feature-box">
            <div class="feature-title">🧠 Deep Understanding</div>
            <div class="feature-text">
                Mathematical foundations and practical applications are explained together 
                for complete comprehension.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <div class="footer-title">📊 Interactive Signal Processing Explorer</div>
        <div>Built with Streamlit, Python, NumPy, SciPy, and Plotly</div>
        <div style="margin-top: 1rem; font-size: 0.85rem;">
            Empowering learners to master Digital Signal Processing through interactive exploration
        </div>
        <div style="margin-top: 0.5rem; color: #9ca3af;">
            © 2024 | Open Source Educational Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()