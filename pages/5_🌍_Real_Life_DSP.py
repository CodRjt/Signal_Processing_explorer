import streamlit as st
import numpy as np
import scipy.signal as signal
from scipy.fft import fft, fftfreq
import plotly.graph_objects as go
from PIL import Image
import io

st.set_page_config(page_title="The DSP Arcade", page_icon="🕹️", layout="wide")

# --- CUSTOM CSS FOR RETRO VIBE ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');
    
    .main-title {
        font-family: 'Press Start 2P', cursive;
        color: #FF0055;
        text-align: center;
        text-shadow: 4px 4px #000000;
        font-size: 2.5rem;
        margin-bottom: 20px;
    }
    .level-header {
        font-family: 'Press Start 2P', cursive;
        color: #00CCFF;
        font-size: 1.2rem;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .arcade-box {
        background-color: #222;
        border: 4px solid #00CCFF;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 10px 10px 0px #000;
        color: #fff;
    }
    .manual-box {
        background-color: #333;
        border-left: 5px solid #FF0055;
        padding: 15px;
        margin-top: 10px;
        font-family: monospace;
        font-size: 0.9rem;
        color: #ddd;
    }
    .score-box {
        background-color: #000;
        color: #00FF00;
        font-family: monospace;
        padding: 10px;
        border: 2px solid #00FF00;
        text-align: center;
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

def generate_lofi_beat(bpm=90, duration=4):
    """Generates a simple synthwave-style beat."""
    fs = 44100
    t = np.linspace(0, duration, int(fs * duration), False)
    
    # Bass (Sine wave)
    bass_freq = 55.0
    bass = np.sin(2 * np.pi * bass_freq * t)
    bass += 0.5 * np.sin(2 * np.pi * (bass_freq * 2) * t) 
    
    # Simple melody
    melody = np.zeros_like(t)
    notes = [220, 261.63, 329.63, 392.00] 
    note_len = int(fs * (60/bpm))
    
    for i in range(len(t)):
        idx = (i // note_len) % 4
        melody[i] = np.sin(2 * np.pi * notes[idx] * t[i]) * np.exp(-((i % note_len)/(note_len/4)))

    mix = (bass * 0.4) + (melody * 0.4)
    return t, mix, fs

def bitcrush(audio, bit_depth, sample_rate_reduction):
    """The 'Lo-Fi' Effect: Quantization + Downsampling."""
    # 1. Downsample
    audio_down = audio[::sample_rate_reduction]
    audio_down = np.repeat(audio_down, sample_rate_reduction)
    
    # Fix length mismatch
    if len(audio_down) > len(audio):
        audio_down = audio_down[:len(audio)]
    elif len(audio_down) < len(audio):
        audio_down = np.pad(audio_down, (0, len(audio) - len(audio_down)))
        
    # 2. Quantize
    levels = 2**bit_depth
    audio_crushed = np.round(audio_down * (levels/2)) / (levels/2)
    
    return audio_crushed

def pixelate_image(img_array, pixel_size):
    """Turn high-res image into pixel art."""
    h, w, c = img_array.shape
    small = img_array[::pixel_size, ::pixel_size]
    pixelated = np.repeat(np.repeat(small, pixel_size, axis=0), pixel_size, axis=1)
    return pixelated[:h, :w]

# --- MAIN APP ---

def main():
    st.markdown('<h1 class="main-title">🕹️ THE DSP ARCADE 🕹️</h1>', unsafe_allow_html=True)
    st.markdown("### Learn Audio Processing by Destroying Signals")
    
    tab1, tab2, tab3 = st.tabs(["Level 1: The Bitcrusher", "Level 2: Pixel Art Factory", "Level 3: Ghost Signal Hunter"])
    
    # --- LEVEL 1: BITCRUSHER ---
    with tab1:
        st.markdown('<div class="level-header">LEVEL 1: MAKE IT RETRO</div>', unsafe_allow_html=True)
        
        # --- MANUAL / CONTEXT ---
        with st.expander("📖 READ MANUAL: What are Bits and Samples?"):
            st.markdown("""
            <div class="manual-box">
            <strong>1. Sample Rate (Time):</strong> Imagine a movie. A smooth movie has 60 frames per second. A choppy stop-motion animation has 5 frames per second. <br>
            <em>Downsampling</em> removes frames, making the audio sound "muffled" or "robotic."
            <br><br>
            <strong>2. Bit Depth (Resolution):</strong> Imagine a staircase. High bit depth means tiny steps (smooth ramp). Low bit depth means giant steps (blocky staircase). <br>
            <em>Quantization</em> forces the smooth sound wave onto these blocky steps, adding "fizz" or "crunch."
            </div>
            """, unsafe_allow_html=True)
            st.caption("Notice how 'low bit depth' turns a smooth curve into a blocky staircase.")

        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.markdown('<div class="arcade-box">', unsafe_allow_html=True)
            st.write("**Mission:** Make this song sound like a 1989 GameBoy track.")
            
            bits = st.slider("Bit Depth (Crunchiness)", 1, 16, 16, help="Lower = More fizz/noise")
            downsample = st.slider("Sample Rate (Muffledness)", 1, 50, 1, help="Higher number = Lower quality")
            
            st.markdown("---")
            st.info("💡 Tip: Try **Bits=4** and **Sample Rate=20** for pure 8-bit nostalgia.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            t, clean_audio, fs = generate_lofi_beat()
            crushed_audio = bitcrush(clean_audio, bits, downsample)
            crushed_audio = np.clip(crushed_audio, -1.0, 1.0)
            
            st.markdown("#### 🎧 Output Monitor")
            st.audio(crushed_audio, sample_rate=fs)
            
            fig = go.Figure()
            limit = 500 * downsample 
            fig.add_trace(go.Scatter(y=clean_audio[:limit], name="Smooth (Original)", line=dict(color='#555', width=1)))
            fig.add_trace(go.Scatter(y=crushed_audio[:limit], name="Blocky (Result)", line=dict(color='#00CCFF', width=3)))
            fig.update_layout(height=250, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark", title="Waveform Zoom")
            st.plotly_chart(fig, use_container_width=True)

    # --- LEVEL 2: PIXEL ART FACTORY ---
    with tab2:
        st.markdown('<div class="level-header">LEVEL 2: PIXEL ARTIST</div>', unsafe_allow_html=True)
        
        # --- MANUAL / CONTEXT ---
        with st.expander("📖 READ MANUAL: Images are Signals too!"):
            st.markdown("""
            <div class="manual-box">
            Did you know an image is just a 2D signal?<br>
            <strong>1. Spatial Downsampling:</strong> Instead of keeping every single pixel, we pick one pixel to represent a whole group (e.g., a 10x10 block). This creates the "Mosaic" effect.<br>
            <strong>2. Color Quantization:</strong> Instead of millions of colors, we force the image to use only a few (e.g., 4 colors). This is exactly like "Bit Depth" in audio!
            </div>
            """, unsafe_allow_html=True)

        col_ctrl, col_art = st.columns([1, 2])
        
        with col_ctrl:
            st.markdown('<div class="arcade-box">', unsafe_allow_html=True)
            st.write("**Mission:** Turn a photo into a retro video game sprite.")
            
            pixel_size = st.slider("Pixel Block Size", 1, 30, 1, help="Combines neighboring pixels into one big block.")
            color_bits = st.select_slider("Color Palette Size", options=[1, 2, 4, 8], value=8, help="1 bit = Black & White only. 8 bits = Full Color.")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_art:
            uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'png'])
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
            else:
                # Default Gradient Pattern
                x = np.linspace(0, 1, 256)
                y = np.linspace(0, 1, 256)
                X, Y = np.meshgrid(x, y)
                img_data = (np.sin(10*X) + np.cos(10*Y)) * 255
                img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min()) * 255
                image = Image.fromarray(np.uint8(np.stack((img_data, img_data*0.5, img_data*0.2), axis=2)))

            img_array = np.array(image)
            pixelated = pixelate_image(img_array, pixel_size)
            
            # Color Quantization
            levels = 2**color_bits
            pixelated = np.floor(pixelated / (256/levels)) * (256/levels)
            pixelated = pixelated.astype(np.uint8)
            
            st.image(pixelated, caption=f"Result: {img_array.shape[1]//pixel_size}x{img_array.shape[0]//pixel_size} virtual pixels", use_container_width=True)

    # --- LEVEL 3: GHOST SIGNAL HUNTER ---
    with tab3:
        st.markdown('<div class="level-header">LEVEL 3: GHOST SIGNAL HUNTER</div>', unsafe_allow_html=True)
        
        # --- MANUAL / CONTEXT ---
        with st.expander("📖 READ MANUAL: How to find a hidden signal"):
            st.markdown("""
            <div class="manual-box">
            <strong>The Problem:</strong> The "Static" you hear is White Noise. It contains <em>all</em> frequencies mixed together.<br>
            <strong>The Solution:</strong> A <strong>Bandpass Filter</strong> acts like a narrow window. It blocks everything except a specific "slice" of sound.<br>
            <strong>The Game:</strong> Slide the window (Frequency) until you find the hidden tone. If the window is too wide, you let too much noise in!
            </div>
            """, unsafe_allow_html=True)

            st.caption("A Bandpass filter only lets frequencies in the 'Pass Band' (the green zone) through.")

        st.markdown("""
        <div class="arcade-box">
        <strong>Mission:</strong> A secret sine wave is hiding in the static. 
        Tune the radio frequency to finding it. Watch the <strong>Green Zone</strong> in the graph!
        </div>
        """, unsafe_allow_html=True)
        
        if 'target_freq' not in st.session_state:
            st.session_state.target_freq = np.random.randint(200, 1000)
            
        col_game, col_vis = st.columns([1, 1])
        
        with col_game:
            # Generate Hidden Signal
            duration = 1.0
            fs = 44100
            t = np.linspace(0, duration, int(fs*duration), False)
            target = st.session_state.target_freq
            signal_hidden = np.sin(2 * np.pi * target * t)
            noise = np.random.normal(0, 2, len(t))
            raw_input = signal_hidden + noise
            
            # Controls
            user_freq = st.slider("Tuner Frequency (Hz)", 100, 1200, 100)
            bandwidth = st.slider("Focus / Precision", 10, 200, 50, help="Smaller = Less Noise, but harder to find the signal!")
            
            # Apply Filter
            nyq = 0.5 * fs
            low = (user_freq - bandwidth/2) / nyq
            high = (user_freq + bandwidth/2) / nyq
            if low <= 0: low = 0.001
            if high >= 1: high = 0.999
            
            b, a = signal.butter(4, [low, high], btype='band')
            filtered_signal = signal.lfilter(b, a, raw_input)
            
            # Scoring
            energy = np.sum(filtered_signal**2)
            max_possible_energy = np.sum(signal_hidden**2) * 0.8 
            score = int((energy / max_possible_energy) * 100)
            score = min(score, 100)
            
            st.markdown(f'<div class="score-box">SIGNAL STRENGTH: {score}%</div>', unsafe_allow_html=True)
            
            if score > 80:
                st.balloons()
                st.success(f"LOCKED ON! Target found at {target} Hz.")
                if st.button("Find Next Signal"):
                    st.session_state.target_freq = np.random.randint(200, 1000)
                    st.rerun()
            
            st.markdown("#### 🎧 Audio Feed")
            st.audio(filtered_signal / np.max(np.abs(filtered_signal)), sample_rate=fs)

        with col_vis:
            st.markdown("**Spectral Scanner**")
            N = len(raw_input)
            yf = fft(raw_input)
            xf = fftfreq(N, 1/fs)[:N//2]
            skip = 100
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xf[::skip], y=2.0/N * np.abs(yf[0:N//2])[::skip], 
                                     line=dict(color='gray', width=1), name="Noise Floor"))
            
            # Hidden Target (Faint Red Line)
            fig.add_vline(x=target, line_color="#FF0055", line_width=2, opacity=0.5, annotation_text="Target")
            
            # User Filter (Green Zone)
            fig.add_vrect(x0=user_freq - bandwidth/2, x1=user_freq + bandwidth/2, 
                          fillcolor="#00FF00", opacity=0.2, line_width=0, annotation_text="Your Filter")
            
            fig.update_layout(xaxis_title="Frequency (Hz)", yaxis_title="Amplitude", xaxis_range=[0, 1500], template="plotly_dark", height=300)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()