import streamlit as st
import numpy as np
import scipy.signal as signal
from scipy.fft import fft, fftfreq
import plotly.graph_objects as go

st.set_page_config(page_title="Audio DSP Workbench", page_icon="🎵", layout="wide")

# --- INITIALIZE SESSION STATE FOR PLAYLIST ---
if 'playlist' not in st.session_state:
    st.session_state.playlist = []

# --- CSS STYLING ---
st.markdown("""
<style>
    .main-title {
        font-family: 'Helvetica Neue', sans-serif;
        background: linear-gradient(90deg, #FF4B2B 0%, #FF416C 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0;
    }
    .rack-mount {
        background-color: #2b2b2b;
        border-radius: 10px;
        padding: 20px;
        border: 2px solid #444;
        color: #ddd;
        margin-bottom: 20px;
        box-shadow: inset 0 0 20px #000;
    }
    .rack-title {
        color: #FF416C;
        font-family: monospace;
        font-weight: bold;
        border-bottom: 1px solid #555;
        margin-bottom: 15px;
        text-transform: uppercase;
        font-size: 0.9rem;
    }
    .tape-label {
        background: #eee;
        color: #333;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: 'Courier New', monospace;
        font-weight: bold;
        font-size: 0.8rem;
        border: 1px solid #999;
    }
    /* Make audio players fit nicely */
    .stAudio {
        margin-top: -10px;
    }
</style>
""", unsafe_allow_html=True)

# --- DSP ENGINE ---

def generate_waveform(wave_type, freq, duration, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    if wave_type == "Sine (Pure)":
        y = np.sin(2 * np.pi * freq * t)
    elif wave_type == "Square (Harsh)":
        y = signal.square(2 * np.pi * freq * t)
    elif wave_type == "Sawtooth (Buzzy)":
        y = signal.sawtooth(2 * np.pi * freq * t)
    elif wave_type == "Triangle (Soft)":
        y = signal.sawtooth(2 * np.pi * freq * t, 0.5)
    return t, y

def add_noise(y, noise_type, level):
    if level == 0: return y
    np.random.seed(42) # Consistent noise for fair A/B testing
    if noise_type == "White Noise":
        noise = np.random.normal(0, 1, len(y))
    elif noise_type == "50Hz Hum":
        t = np.linspace(0, len(y)/44100, len(y), endpoint=False)
        noise = np.sin(2 * np.pi * 50 * t)
    elif noise_type == "High Hiss (8k)":
        t = np.linspace(0, len(y)/44100, len(y), endpoint=False)
        noise = np.sin(2 * np.pi * 8000 * t)
    return y + (noise * level)

def apply_filter(y, filter_type, cutoff, sample_rate=44100):
    if filter_type == "No Filter": return y
    nyquist = 0.5 * sample_rate
    norm_cutoff = np.clip(cutoff / nyquist, 0.01, 0.99)
    btype = 'low' if "Low Pass" in filter_type else 'high'
    b, a = signal.butter(4, norm_cutoff, btype=btype, analog=False)
    return signal.lfilter(b, a, y)

def compute_fft(y, sample_rate):
    N = len(y)
    yf = fft(y)
    xf = fftfreq(N, 1 / sample_rate)
    return xf[:N//2], 2.0/N * np.abs(yf[0:N//2])

# --- MAIN APP ---

def main():
    st.markdown('<h1 class="main-title"> Audio DSP Workbench</h1>', unsafe_allow_html=True)

    # --- TOP: THE CONTROL RACK ---
    # We use a container to visually group the controls
    with st.container():
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown('<div class="rack-mount"><div class="rack-title">1. GENERATOR</div>', unsafe_allow_html=True)
            wave = st.selectbox("Waveform", ["Sine (Pure)", "Square (Harsh)", "Sawtooth (Buzzy)", "Triangle (Soft)"])
            freq = st.slider("Freq (Hz)", 100, 1000, 440, 10)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c2:
            st.markdown('<div class="rack-mount"><div class="rack-title">2. NOISE CHANNEL</div>', unsafe_allow_html=True)
            noise_type = st.selectbox("Noise Type", ["White Noise", "50Hz Hum", "High Hiss (8k)"])
            noise_lvl = st.slider("Level", 0.0, 1.0, 0.0, 0.1)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c3:
            st.markdown('<div class="rack-mount"><div class="rack-title">3. FILTER UNIT</div>', unsafe_allow_html=True)
            filt_type = st.selectbox("Mode", ["No Filter", "Low Pass (Cut High)", "High Pass (Cut Low)"])
            cutoff = st.slider("Cutoff (Hz)", 50, 5000, 1000, 50)
            st.markdown('</div>', unsafe_allow_html=True)

    # --- PROCESSING ---
    sr = 44100
    t, y_clean = generate_waveform(wave, freq, 1.0, sr)
    y_noisy = add_noise(y_clean, noise_type, noise_lvl)
    y_final = apply_filter(y_noisy, filt_type, cutoff, sr)
    
    # Normalize volume
    y_audio = y_final / np.max(np.abs(y_final) + 1e-6)

    # --- MIDDLE: VISUALIZATION ---
    st.markdown("### 📊 Monitor")
    
    tab_osc, tab_spec = st.tabs(["Oscilloscope (Time)", "Spectrum (Frequency)"])
    
    with tab_osc:
        fig_time = go.Figure()
        # Plot only 0.02 seconds for visibility
        limit = int(0.02 * sr)
        fig_time.add_trace(go.Scatter(x=t[:limit], y=y_final[:limit], line=dict(color='#00ff00', width=2), name='Output'))
        fig_time.update_layout(height=250, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark", xaxis_title="Time (s)")
        st.plotly_chart(fig_time, use_container_width=True)
        
    with tab_spec:
        xf, yf = compute_fft(y_final, sr)
        fig_freq = go.Figure()
        limit_idx = np.searchsorted(xf, 10000)
        fig_freq.add_trace(go.Scatter(x=xf[:limit_idx], y=yf[:limit_idx], fill='tozeroy', line=dict(color='#00ccff'), name='Spectrum'))
        if "Filter" in filt_type:
            fig_freq.add_vline(x=cutoff, line_dash="dash", line_color="yellow", annotation_text="Cutoff")
        fig_freq.update_layout(height=250, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark", xaxis_title="Frequency (Hz)")
        st.plotly_chart(fig_freq, use_container_width=True)

    # --- BOTTOM: TAPE LIBRARY (THE COMPARISON TOOL) ---
    st.markdown("---")
    st.markdown("### 📼 Tape Loop Library (Comparison Tool)")
    
    # The "Add to Library" Action
    col_add, col_info = st.columns([1, 4])
    with col_add:
        # Create a descriptive name automatically
        noise_desc = f"+ {noise_type}" if noise_lvl > 0 else ""
        filt_desc = f"-> {filt_type}@{cutoff}Hz" if filt_type != "No Filter" else ""
        track_name = f"{wave} {freq}Hz {noise_desc} {filt_desc}"
        
        if st.button("🔴 REC / SAVE", use_container_width=True, type="primary"):
            st.session_state.playlist.append({
                "name": track_name,
                "data": y_audio,
                "sr": sr
            })
            st.toast(f"Saved: {track_name}")

    with col_info:
        st.caption("👈 **Click REC** to save the current sound settings. Save multiple versions to compare them side-by-side below.")

    # Render the Library
    if len(st.session_state.playlist) > 0:
        st.write("")
        # Clear button
        if st.button("Clear Tapes", type="secondary"):
            st.session_state.playlist = []
            st.rerun()

        # Display tapes in a grid or list
        for i, track in enumerate(reversed(st.session_state.playlist)):
            # Use a container for visual grouping
            with st.container():
                c_play, c_desc = st.columns([1, 3])
                with c_play:
                    st.audio(track["data"], sample_rate=track["sr"])
                with c_desc:
                    st.markdown(f'<span class="tape-label">TAPE #{len(st.session_state.playlist)-i}</span> **{track["name"]}**', unsafe_allow_html=True)
                st.markdown("---")
    else:
        st.info("The tape library is empty. Tweak the knobs above and click 'REC' to save your first sound.")
    st.markdown("---")
    with st.expander("📚 Further Reading"):
        st.markdown("""
        * **[Digital Signal Processing (Wikipedia)](https://en.wikipedia.org/wiki/Digital_signal_processing):** The foundation of this tool.
        * **[White Noise vs Pink Noise](https://en.wikipedia.org/wiki/Colors_of_noise):** Learn why some static sounds harsh and others soothing.
        * **[Fast Fourier Transform (FFT)](https://en.wikipedia.org/wiki/Fast_Fourier_transform):** The math behind the "Spectrum" view.
        """)

if __name__ == "__main__":
    main()