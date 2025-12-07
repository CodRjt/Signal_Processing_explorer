import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.fft import fft, fftfreq
from matplotlib import use
use('Agg')

st.set_page_config(page_title="DTMF Checker", page_icon="📞", layout="wide")

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Keypad button styling */
    div[data-testid="column"] button {
        font-size: 2rem !important;
        font-weight: bold !important;
        height: 80px !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
        border: 2px solid #e0e0e0 !important;
    }
    
    div[data-testid="column"] button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.2) !important;
        border-color: #667eea !important;
    }
    
    /* Info box styling */
    .stAlert {
        border-radius: 12px;
        border-left: 5px solid #667eea;
        animation: slideIn 0.5s ease;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Card-like sections */
    .section-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Frequency badge */
    .freq-badge {
        display: inline-block;
        padding: 8px 16px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        font-weight: bold;
        margin: 5px;
        font-size: 1.1rem;
    }
    
    /* Stats container */
    .stats-container {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
        margin: 1rem 0;
    }
    
    .stat-box {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        min-width: 150px;
        margin: 0.5rem;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    
    .stat-label {
        color: #666;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def generate_dtmf_tone(key, duration=0.5, sample_rate=8000):
    """Generate DTMF tone for given key"""
    dtmf_freqs = {
        '1': (697, 1209), '2': (697, 1336), '3': (697, 1477), 'A': (697, 1633),
        '4': (770, 1209), '5': (770, 1336), '6': (770, 1477), 'B': (770, 1633),
        '7': (852, 1209), '8': (852, 1336), '9': (852, 1477), 'C': (852, 1633),
        '*': (941, 1209), '0': (941, 1336), '#': (941, 1477), 'D': (941, 1633)
    }
    
    if key not in dtmf_freqs:
        return None, None, None
    
    low_freq, high_freq = dtmf_freqs[key]
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    tone = (np.sin(2 * np.pi * low_freq * t) + 
            np.sin(2 * np.pi * high_freq * t)) / 2
    
    return t, tone, (low_freq, high_freq)

def analyze_spectrum(signal, sample_rate):
    """Perform FFT analysis of the signal"""
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1 / sample_rate)
    
    xf = xf[:N//2]
    yf = np.abs(yf[:N//2])
    
    return xf, yf

def detect_dtmf_key(spectrum_freqs, spectrum_magnitude):
    """Detect DTMF key from frequency spectrum"""
    threshold = np.max(spectrum_magnitude) * 0.3
    peak_indices = np.where(spectrum_magnitude > threshold)[0]
    peak_freqs = spectrum_freqs[peak_indices]
    
    low_freqs = [697, 770, 852, 941]
    high_freqs = [1209, 1336, 1477, 1633]
    
    detected_low = None
    detected_high = None
    
    for freq in peak_freqs:
        for lf in low_freqs:
            if abs(freq - lf) < 50:
                detected_low = lf
                break
        for hf in high_freqs:
            if abs(freq - hf) < 50:
                detected_high = hf
                break
    
    if detected_low and detected_high:
        dtmf_map = {
            (697, 1209): '1', (697, 1336): '2', (697, 1477): '3', (697, 1633): 'A',
            (770, 1209): '4', (770, 1336): '5', (770, 1477): '6', (770, 1633): 'B',
            (852, 1209): '7', (852, 1336): '8', (852, 1477): '9', (852, 1633): 'C',
            (941, 1209): '*', (941, 1336): '0', (941, 1477): '#', (941, 1633): 'D'
        }
        return dtmf_map.get((detected_low, detected_high), 'Unknown')
    
    return 'No key detected'

def main():
    # Header
    st.markdown('<h1 class="main-title">📞 DTMF Signal Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Dual-Tone Multi-Frequency Decoder & Visualizer</p>', unsafe_allow_html=True)
    
    # Info banner
    with st.expander("ℹ️ What is DTMF?", expanded=False):
        st.markdown("""
        **DTMF (Dual-Tone Multi-Frequency)** is the signaling system used by touch-tone telephones. 
        Each key press generates **two simultaneous tones** at specific frequencies:
        - 🔊 One from a **low frequency group** (697-941 Hz)
        - 🔊 One from a **high frequency group** (1209-1633 Hz)
        
        This system allows reliable transmission of dialing information over telephone lines!
        """)
    
    st.markdown("---")
    
    # Initialize session state
    if 'dtmf_key' not in st.session_state:
        st.session_state.dtmf_key = '5'
    if 'key_history' not in st.session_state:
        st.session_state.key_history = []
    
    # Main layout
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        st.markdown("### 🎹 Virtual Keypad")
        st.markdown("*Click any button to generate its DTMF signal*")
        
        # Keypad layout
        keypad_layout = [
            ['1', '2', '3'],
            ['4', '5', '6'], 
            ['7', '8', '9'],
            ['*', '0', '#']
        ]
        
        selected_key = None
        
        for row in keypad_layout:
            cols = st.columns(3)
            for i, key in enumerate(row):
                with cols[i]:
                    if st.button(key, key=f"btn_{key}", use_container_width=True):
                        selected_key = key
                        st.session_state.key_history.append(key)
                        if len(st.session_state.key_history) > 10:
                            st.session_state.key_history.pop(0)
        
        if selected_key:
            st.session_state.dtmf_key = selected_key
        
        st.markdown("---")
        
        # Parameters section
        st.markdown("### ⚙️ Signal Parameters")
        duration = st.slider("🕐 Tone Duration (seconds)", 0.1, 2.0, 0.5, 0.1, 
                           help="Duration of the generated tone")
        sample_rate = st.selectbox("📊 Sample Rate (Hz)", 
                                   [4000, 8000, 16000, 44100], 
                                   index=1,
                                   help="Higher sample rate = better frequency resolution")
        
        # Key history
        if st.session_state.key_history:
            st.markdown("### 📝 Key History")
            history_text = " → ".join(st.session_state.key_history[-5:])
            st.code(history_text)
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.key_history = []
                st.rerun()
    
    with col2:
        current_key = st.session_state.dtmf_key
        
        if current_key:
            # Generate and analyze
            t, tone, freqs = generate_dtmf_tone(current_key, duration, sample_rate)
            audio_data = (tone * 32767).astype(np.int16) 
            st.audio(audio_data, sample_rate=sample_rate)
            if tone is not None:
                spectrum_freqs, spectrum_magnitude = analyze_spectrum(tone, sample_rate)
                detected_key = detect_dtmf_key(spectrum_freqs, spectrum_magnitude)
                
                # Stats display
                st.markdown("### 📊 Signal Information")
                
                stats_html = f"""
                <div class="stats-container">
                    <div class="stat-box">
                        <div class="stat-value">{current_key}</div>
                        <div class="stat-label">Current Key</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{detected_key}</div>
                        <div class="stat-label">Detected Key</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{freqs[0]} Hz</div>
                        <div class="stat-label">Low Frequency</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{freqs[1]} Hz</div>
                        <div class="stat-label">High Frequency</div>
                    </div>
                </div>
                """
                st.markdown(stats_html, unsafe_allow_html=True)
                
                # Validation indicator
                if detected_key == current_key:
                    st.success("✅ **Detection Successful!** The signal matches the expected key.")
                else:
                    st.warning("⚠️ **Detection Mismatch** - Check signal quality or parameters.")
                
                st.markdown("---")
                
                # Visualizations
                tab1, tab2, tab3 = st.tabs(["📈 Time Domain", "🎵 Frequency Spectrum", "📚 Theory"])
                
                with tab1:
                    st.markdown("### Time Domain Representation")
                    fig_time = plt.figure(figsize=(12, 5))
                    
                    # Show first 2000 samples or all if less
                    samples_to_show = min(2000, len(t))
                    plt.plot(t[:samples_to_show], tone[:samples_to_show], 'b-', linewidth=1.5, alpha=0.8)
                    plt.fill_between(t[:samples_to_show], tone[:samples_to_show], alpha=0.3)
                    
                    plt.title(f'DTMF Signal - Key "{current_key}" ({freqs[0]} Hz + {freqs[1]} Hz)', 
                             fontsize=14, fontweight='bold', pad=20)
                    plt.xlabel('Time (seconds)', fontsize=12)
                    plt.ylabel('Amplitude', fontsize=12)
                    plt.grid(True, alpha=0.3, linestyle='--')
                    plt.tight_layout()
                    st.pyplot(fig_time)
                    
                    st.info(f"💡 The waveform shows the combined signal of two sine waves at {freqs[0]} Hz and {freqs[1]} Hz")
                
                with tab2:
                    st.markdown("### Frequency Domain Analysis (FFT)")
                    
                    # Interactive Plotly chart
                    plotly_fig = go.Figure()
                    
                    # Main spectrum
                    plotly_fig.add_trace(go.Scatter(
                        x=spectrum_freqs, 
                        y=spectrum_magnitude,
                        mode='lines',
                        name='FFT Spectrum',
                        line=dict(color='#667eea', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(102, 126, 234, 0.2)'
                    ))
                    
                    # Peak markers
                    for i, freq in enumerate(freqs):
                        # Find the peak value
                        peak_idx = np.argmin(np.abs(spectrum_freqs - freq))
                        peak_val = spectrum_magnitude[peak_idx]
                        
                        plotly_fig.add_trace(go.Scatter(
                            x=[freq],
                            y=[peak_val],
                            mode='markers',
                            name=f'{"Low" if i==0 else "High"} Freq: {freq} Hz',
                            marker=dict(size=12, color='red' if i==0 else 'green', 
                                      symbol='diamond')
                        ))
                        
                        # Vertical line
                        plotly_fig.add_vline(
                            x=freq, 
                            line=dict(color='red' if i==0 else 'green', dash='dash', width=2),
                            annotation_text=f"{freq} Hz",
                            annotation_position="top"
                        )
                    
                    plotly_fig.update_layout(
                        title=f"Interactive FFT Spectrum - Key '{current_key}'",
                        xaxis_title="Frequency (Hz)",
                        yaxis_title="Magnitude",
                        height=500,
                        hovermode='x unified',
                        template='plotly_white'
                    )
                    
                    plotly_fig.update_xaxes(range=[0, 2000])
                    st.plotly_chart(plotly_fig, use_container_width=True)
                    
                    st.info("💡 Hover over the chart to see exact frequency values. The two peaks correspond to the DTMF frequencies.")
                
                with tab3:
                    st.markdown("### 📚 DTMF Theory & Mathematics")
                    
                    theory_col1, theory_col2 = st.columns(2)
                    
                    with theory_col1:
                        st.markdown("**DTMF Frequency Table:**")
                        st.markdown("""
                        | Hz  | 1209 | 1336 | 1477 | 1633 |
                        |-----|------|------|------|------|
                        | 697 |  1   |  2   |  3   |  A   |
                        | 770 |  4   |  5   |  6   |  B   |
                        | 852 |  7   |  8   |  9   |  C   |
                        | 941 |  *   |  0   |  #   |  D   |
                        """)
                        
                        st.markdown("**Applications:**")
                        st.markdown("""
                        - ☎️ Telephone dialing
                        - 🏧 ATM machines
                        - 📻 Two-way radio systems
                        - 🎛️ Industrial control systems
                        """)
                    
                    with theory_col2:
                        st.markdown("**Signal Generation:**")
                        st.latex(r"s(t) = A_1 \sin(2\pi f_1 t) + A_2 \sin(2\pi f_2 t)")
                        
                        st.markdown("**Where:**")
                        st.markdown(f"""
                        - $f_1 = {freqs[0]}$ Hz (Low frequency)
                        - $f_2 = {freqs[1]}$ Hz (High frequency)
                        - $A_1, A_2$ = 0.5 (Equal amplitudes)
                        """)
                        
                        st.markdown("**Detection Process:**")
                        st.markdown("""
                        1. 📊 Apply FFT to received signal
                        2. 🔍 Identify peak frequencies
                        3. 🎯 Match peaks to DTMF table
                        4. ✅ Decode corresponding key
                        """)

if __name__ == "__main__":
    main()