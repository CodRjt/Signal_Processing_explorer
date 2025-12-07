import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.fft import fft, fftfreq
from scipy import signal as sp_signal
from matplotlib import use
use('Agg')

import textwrap

st.set_page_config(page_title="Sampling Rate Visualization", page_icon="📊", layout="wide")

# Enhanced CSS styling
st.markdown(textwrap.dedent("""
<style>
    /* Page-level dark theme adjustments for better contrast */
    body, .stApp {
        background-color: #0b1220;
        color: #e6eef8;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }
    /* Main title */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #cbd5e1;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Status indicators */
    .status-excellent {
        background: linear-gradient(135deg, #08331a 0%, #0b4426 100%);
        color: #c7f9d3;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
        font-size: 1.1rem;
        box-shadow: 0 6px 18px rgba(2,6,23,0.6);
        border: 1px solid rgba(255,255,255,0.03);
    }
    
    .status-warning {
        background: linear-gradient(135deg, #3a2308 0%, #512f0b 100%);
        color: #ffe8c2;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
        font-size: 1.1rem;
        box-shadow: 0 6px 18px rgba(2,6,23,0.6);
        animation: pulse 2s infinite;
        border: 1px solid rgba(255,255,255,0.03);
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    /* Metric cards */
    .metric-card {
        background: #0f1724;
        padding: 1.25rem;
        border-radius: 12px;
        box-shadow: 0 6px 18px rgba(2,6,23,0.6);
        text-align: center;
        transition: transform 0.3s ease;
        border-top: 4px solid #2196f3;
        border: 1px solid rgba(255,255,255,0.03);
        color: #cbd5e1;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(102,126,234,0.2);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #e6eef8;
        background: linear-gradient(135deg, #7dd3fc 0%, #60a5fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        color: #9aa6b8;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-sublabel {
        color: #98a6b8;
        font-size: 0.78rem;
        margin-top: 0.3rem;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #071226 0%, #092036 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
        color: #cbd5e1;
        border: 1px solid rgba(255,255,255,0.03);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #2b1a08 0%, #3a220b 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
        color: #ffe8c2;
        border: 1px solid rgba(255,255,255,0.03);
    }
    
    .success-box {
        background: linear-gradient(135deg, #08331a 0%, #0b4426 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
        color: #c7f9d3;
        border: 1px solid rgba(255,255,255,0.03);
    }
    
    .danger-box {
        background: linear-gradient(135deg, #3a0a0a 0%, #4b0f0f 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #f44336;
        margin: 1rem 0;
        color: #ffd6d6;
        border: 1px solid rgba(255,255,255,0.03);
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e6eef8;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    /* Frequency badges */
    .freq-badge {
        display: inline-block;
        padding: 6px 12px;
        background: linear-gradient(135deg, #0b2a44 0%, #14385a 100%);
        color: #e6eef8;
        border-radius: 20px;
        font-weight: bold;
        margin: 3px;
        font-size: 0.9rem;
        border: 1px solid rgba(255,255,255,0.03);
    }
    
    /* Preset buttons styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    /* Custom radio button styling */
    .preset-option {
        background: #071226;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.03);
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
        color: #cbd5e1;
    }
    
    .preset-option:hover {
        border-color: #667eea;
        box-shadow: 0 6px 18px rgba(2,6,23,0.6);
    }
</style>
"""), unsafe_allow_html=True)

def generate_test_signal(t, freq1=5, freq2=15, freq3=25):
    """Generate a composite test signal with multiple frequency components"""
    signal = (np.sin(2 * np.pi * freq1 * t) + 
             0.5 * np.sin(2 * np.pi * freq2 * t) + 
             0.3 * np.sin(2 * np.pi * freq3 * t))
    return signal

def sample_signal(t_continuous, signal_continuous, fs):
    """Sample the continuous signal at given sampling frequency"""
    dt = 1 / fs
    t_max = t_continuous[-1]
    t_sampled = np.arange(0, t_max, dt)
    signal_sampled = np.interp(t_sampled, t_continuous, signal_continuous)
    return t_sampled, signal_sampled

def reconstruct_signal(t_sampled, signal_sampled, t_continuous, method='linear'):
    """Reconstruct signal from samples using different interpolation methods"""
    if method == 'linear':
        return np.interp(t_continuous, t_sampled, signal_sampled)
    elif method == 'cubic':
        from scipy import interpolate
        f = interpolate.interp1d(t_sampled, signal_sampled, kind='cubic', 
                               bounds_error=False, fill_value='extrapolate')
        return f(t_continuous)
    elif method == 'sinc':
        dt = t_sampled[1] - t_sampled[0] if len(t_sampled) > 1 else 0.01
        reconstructed = np.zeros_like(t_continuous)
        for i, sample in enumerate(signal_sampled):
            sinc_kernel = np.sinc((t_continuous - t_sampled[i]) / dt)
            reconstructed += sample * sinc_kernel
        return reconstructed

def analyze_spectrum(signal, fs):
    """Perform FFT analysis"""
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1/fs)
    xf = xf[:N//2]
    yf = np.abs(yf[:N//2]) * 2 / N
    return xf, yf

def get_aliasing_status(sampling_ratio, aliased_count):
    """Determine aliasing status and styling"""
    if sampling_ratio >= 2.0:
        return "🟢 Excellent - High Oversampling", "status-excellent", "No aliasing, perfect reconstruction possible"
    elif sampling_ratio >= 1.0:
        return "🟡 Good - Above Nyquist", "status-excellent", "Sufficient sampling, minimal distortion"
    else:
        return f"🔴 Warning - Aliasing! ({aliased_count} freq aliased)", "status-warning", "Below Nyquist rate, information loss occurring"

def main():
    # Header
    st.markdown('<h1 class="main-title"> Interactive Sampling & Aliasing Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Visualize the Nyquist-Shannon Sampling Theorem in Real-Time</p>', unsafe_allow_html=True)
    
    # Info banner
    with st.expander("ℹ️ What is the Nyquist Sampling Theorem?", expanded=False):
        st.markdown("""
        ### The Fundamental Principle of Digital Signal Processing
        
        The **Nyquist-Shannon Sampling Theorem** states that to perfectly reconstruct a continuous signal, 
        you must sample it at **at least twice the highest frequency** present in the signal.
        
        **Key Formula:** $f_s ≥ 2f_{max}$ (Nyquist Rate)
        
        **Why it matters:**
        - 🎵 Audio CDs sample at 44.1 kHz to capture frequencies up to 22.05 kHz (human hearing limit)
        - 📱 Cell phone voice: 8 kHz sampling for 4 kHz bandwidth
        - 📷 Digital cameras use pixel spacing based on this principle
        - 🔬 Oscilloscopes follow this for accurate waveform display
        
        **What happens below Nyquist?** → **Aliasing**: High frequencies masquerade as low frequencies!
        """)
    
    st.markdown("---")
    
    # Initialize session state
    if 'preset_applied' not in st.session_state:
        st.session_state.preset_applied = None
    
    # Main layout
    col1, col2 = st.columns([1, 2.5], gap="large")
    
    with col1:
        st.markdown('<div class="section-header">🎛️ Signal Configuration</div>', unsafe_allow_html=True)
        
        # Quick presets
        st.markdown("#### 🎯 Quick Presets")
        preset_cols = st.columns(2)
        
        with preset_cols[0]:
            if st.button("🎵 Audio Signal", use_container_width=True, help="Simulates audio signal sampling"):
                st.session_state.preset_applied = 'audio'
                st.rerun()
        
        with preset_cols[1]:
            if st.button("📡 Low-pass Signal", use_container_width=True, help="Simple low-frequency signal"):
                st.session_state.preset_applied = 'lowpass'
                st.rerun()
        
        preset_cols2 = st.columns(2)
        with preset_cols2[0]:
            if st.button("⚠️ Aliasing Demo", use_container_width=True, help="Demonstrates aliasing effect"):
                st.session_state.preset_applied = 'alias'
                st.rerun()
        
        with preset_cols2[1]:
            if st.button("🔬 High Resolution", use_container_width=True, help="High sampling rate example"):
                st.session_state.preset_applied = 'highres'
                st.rerun()
        
        st.markdown("---")
        
        # Apply presets
        if st.session_state.preset_applied == 'audio':
            freq1, freq2, freq3, sampling_rate = 440, 880, 1320, 8000
        elif st.session_state.preset_applied == 'lowpass':
            freq1, freq2, freq3, sampling_rate = 2, 5, 8, 50
        elif st.session_state.preset_applied == 'alias':
            freq1, freq2, freq3, sampling_rate = 5, 25, 45, 40
        elif st.session_state.preset_applied == 'highres':
            freq1, freq2, freq3, sampling_rate = 10, 20, 30, 200
        else:
            freq1, freq2, freq3, sampling_rate = 5, 15, 25, 50
        
        # Reset preset after applying
        if st.session_state.preset_applied:
            st.session_state.preset_applied = None
        
        # Signal composition
        st.markdown("#### 🎼 Signal Composition")
        
        freq1 = st.slider("🔵 Frequency 1 (Hz)", 1, 50, freq1, 1, 
                         help="Primary frequency component")
        freq2 = st.slider("🟡 Frequency 2 (Hz)", 1, 50, freq2, 1,
                         help="Secondary frequency component (0.5x amplitude)")
        freq3 = st.slider("🟣 Frequency 3 (Hz)", 1, 50, freq3, 1,
                         help="Tertiary frequency component (0.3x amplitude)")
        
        max_freq = max(freq1, freq2, freq3)
        nyquist_rate = 2 * max_freq
        
        # Display frequency info with badges
        st.markdown(textwrap.dedent(f"""
        <div class="info-box">
            <strong>Signal Components:</strong><br>
            <span class="freq-badge">{freq1} Hz</span>
            <span class="freq-badge">{freq2} Hz</span>
            <span class="freq-badge">{freq3} Hz</span>
            <br><br>
            <strong>Max Frequency:</strong> {max_freq} Hz<br>
            <strong>Nyquist Rate:</strong> {nyquist_rate} Hz (minimum required)
        </div>
        """), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Sampling parameters
        st.markdown("#### 📡 Sampling Configuration")
        
        sampling_rate = st.slider(
            "Sampling Frequency (Hz)",
            10, 200, sampling_rate, 5,
            help="Rate at which the signal is sampled"
        )
        
        # Calculate and display sampling ratio
        sampling_ratio = sampling_rate / nyquist_rate
        
        # Reconstruction method
        reconstruction_method = st.selectbox(
            "🔧 Reconstruction Method", 
            ['linear', 'cubic', 'sinc'],
            index=0,
            help="Algorithm used to reconstruct the continuous signal"
        )
        
        method_info = {
            'linear': '📐 Fast, simple, moderate accuracy',
            'cubic': '🎨 Smooth curves, good visual quality',
            'sinc': '🎯 Theoretically perfect for band-limited signals'
        }
        st.caption(method_info[reconstruction_method])
        
        st.markdown("---")
        
        # Display options
        st.markdown("#### 👁️ Visualization Options")
        show_samples = st.checkbox("Show Sample Points", value=True)
        show_reconstruction = st.checkbox("Show Reconstructed Signal", value=True)
        show_error = st.checkbox("Show Reconstruction Error", value=True)
    
    with col2:
        # Generate signals
        t_continuous = np.linspace(0, 2, 2000)
        signal_continuous = generate_test_signal(t_continuous, freq1, freq2, freq3)
        t_sampled, signal_sampled = sample_signal(t_continuous, signal_continuous, sampling_rate)
        
        if show_reconstruction:
            signal_reconstructed = reconstruct_signal(
                t_sampled, signal_sampled, t_continuous, reconstruction_method
            )
        else:
            signal_reconstructed = signal_continuous
        
        # Calculate aliasing
        aliased_freqs = []
        for f in [freq1, freq2, freq3]:
            if f > sampling_rate / 2:
                alias_freq = abs(f - sampling_rate * round(f / sampling_rate))
                if alias_freq > sampling_rate / 2:
                    alias_freq = sampling_rate - alias_freq
                aliased_freqs.append((f, alias_freq))
        
        # Status display
        status_text, status_class, status_desc = get_aliasing_status(sampling_ratio, len(aliased_freqs))
        
        st.markdown(textwrap.dedent(f"""
        <div class="{status_class}">
            <div style="font-size: 1.3rem;">{status_text}</div>
            <div style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.9;">{status_desc}</div>
        </div>
        """), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Metrics dashboard
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.markdown(textwrap.dedent(f"""
            <div class="metric-card">
                <div class="metric-value">{sampling_rate}</div>
                <div class="metric-label">Sampling Rate</div>
                <div class="metric-sublabel">Hz (samples/sec)</div>
            </div>
            """), unsafe_allow_html=True)
        
        with metric_col2:
            st.markdown(textwrap.dedent(f"""
            <div class="metric-card">
                <div class="metric-value">{sampling_ratio:.2f}x</div>
                <div class="metric-label">Nyquist Ratio</div>
                <div class="metric-sublabel">fs / (2·fmax)</div>
            </div>
            """), unsafe_allow_html=True)
        
        with metric_col3:
            st.markdown(textwrap.dedent(f"""
            <div class="metric-card">
                <div class="metric-value">{len(t_sampled)}</div>
                <div class="metric-label">Total Samples</div>
                <div class="metric-sublabel">in 2 seconds</div>
            </div>
            """), unsafe_allow_html=True)
        
        with metric_col4:
            nyquist_freq = sampling_rate / 2
            st.markdown(textwrap.dedent(f"""
            <div class="metric-card">
                <div class="metric-value">{nyquist_freq:.0f}</div>
                <div class="metric-label">Nyquist Freq</div>
                <div class="metric-sublabel">fs / 2 (Hz)</div>
            </div>
            """), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Tabbed visualization
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Time Domain", "🌊 Frequency Domain", "📊 Error Analysis", "📚 Theory"])
        
        with tab1:
            st.markdown("### Signal in Time Domain")
            
            # Create time domain plot
            fig_time = go.Figure()
            
            # Original signal
            fig_time.add_trace(go.Scatter(
                x=t_continuous,
                y=signal_continuous,
                mode='lines',
                name='Original (Continuous)',
                line=dict(color='#2196f3', width=3),
                hovertemplate='Time: %{x:.3f}s<br>Amplitude: %{y:.3f}<extra></extra>'
            ))
            
            # Sampled points
            if show_samples:
                fig_time.add_trace(go.Scatter(
                    x=t_sampled,
                    y=signal_sampled,
                    mode='markers',
                    name=f'Samples ({len(t_sampled)} points)',
                    marker=dict(color='#f44336', size=10, symbol='circle',
                              line=dict(color='white', width=2)),
                    hovertemplate='Sample at %{x:.3f}s<br>Value: %{y:.3f}<extra></extra>'
                ))
                
                # Add stems for better visualization
                for t, s in zip(t_sampled, signal_sampled):
                    fig_time.add_trace(go.Scatter(
                        x=[t, t],
                        y=[0, s],
                        mode='lines',
                        line=dict(color='#f44336', width=1, dash='dot'),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
            
            # Reconstructed signal
            if show_reconstruction:
                fig_time.add_trace(go.Scatter(
                    x=t_continuous,
                    y=signal_reconstructed,
                    mode='lines',
                    name=f'Reconstructed ({reconstruction_method})',
                    line=dict(color='#4caf50', width=2, dash='dash'),
                    hovertemplate='Time: %{x:.3f}s<br>Reconstructed: %{y:.3f}<extra></extra>'
                ))
            
            fig_time.update_layout(
                title=dict(
                    text=f"Sampling at {sampling_rate} Hz (Nyquist: {nyquist_rate} Hz)",
                    font=dict(size=16, color='#333')
                ),
                xaxis_title="Time (seconds)",
                yaxis_title="Amplitude",
                height=500,
                hovermode='x unified',
                template='plotly_white',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig_time, use_container_width=True)
            
            # Explanation
            if sampling_ratio < 1:
                st.markdown(textwrap.dedent("""
                <div class="danger-box">
                    <strong>⚠️ Aliasing Visible:</strong> Notice how the reconstructed signal (green dashed line) 
                    differs significantly from the original. High frequency components appear as lower frequencies!
                </div>
                """), unsafe_allow_html=True)
            else:
                st.markdown(textwrap.dedent("""
                <div class="success-box">
                    <strong>✅ Good Reconstruction:</strong> The reconstructed signal closely matches the original. 
                    The sampling rate is sufficient to capture all frequency components.
                </div>
                """), unsafe_allow_html=True)
    
        with tab2:
            st.markdown("### Frequency Spectrum Analysis")
            
            # Analyze spectra
            freq_orig, spectrum_orig = analyze_spectrum(signal_continuous, 1000)
            freq_sampled, spectrum_sampled = analyze_spectrum(signal_sampled, sampling_rate)
            
            fig_freq = go.Figure()
            
            # Original spectrum
            fig_freq.add_trace(go.Scatter(
                x=freq_orig,
                y=spectrum_orig,
                mode='lines',
                name='Original Spectrum',
                line=dict(color='#2196f3', width=3),
                fill='tozeroy',
                fillcolor='rgba(33, 150, 243, 0.1)'
            ))
            
            # Sampled spectrum
            fig_freq.add_trace(go.Scatter(
                x=freq_sampled,
                y=spectrum_sampled,
                mode='lines+markers',
                name='Sampled Spectrum',
                line=dict(color='#f44336', width=2),
                marker=dict(size=8)
            ))
            
            # Mark Nyquist frequency
            nyquist_freq = sampling_rate / 2
            fig_freq.add_vline(
                x=nyquist_freq,
                line=dict(color='#ff9800', dash='dash', width=3),
                annotation=dict(
                    text=f"Nyquist Frequency<br>{nyquist_freq} Hz",
                    showarrow=True,
                    arrowhead=2
                )
            )
            
            # Mark original frequency components
            colors = ['#4caf50', '#8bc34a', '#cddc39']
            for i, (f, name) in enumerate([(freq1, 'f₁'), (freq2, 'f₂'), (freq3, 'f₃')]):
                fig_freq.add_vline(
                    x=f,
                    line=dict(color=colors[i], dash='dot', width=2),
                    annotation=dict(
                        text=f"{name}={f}Hz",
                        showarrow=False,
                        yshift=10 * (i - 1)
                    )
                )
            
            fig_freq.update_layout(
                title="FFT Spectrum: Original vs Sampled",
                xaxis_title="Frequency (Hz)",
                yaxis_title="Magnitude",
                height=500,
                template='plotly_white',
                showlegend=True
            )
            
            fig_freq.update_xaxes(range=[0, min(100, sampling_rate)])
            st.plotly_chart(fig_freq, use_container_width=True)
            
            # Aliasing information
            if aliased_freqs:
                st.markdown(textwrap.dedent(f"""
                <div class="warning-box">
                    <strong>🔴 Aliased Frequencies Detected:</strong><br>
                    {"".join([f"• <strong>{orig} Hz</strong> appears as <strong>{alias:.1f} Hz</strong> (folded around Nyquist)<br>" 
                              for orig, alias in aliased_freqs])}
                </div>
                """), unsafe_allow_html=True)
            else:
                st.markdown(textwrap.dedent("""
                <div class="success-box">
                    <strong>✅ No Aliasing:</strong> All frequency components are below the Nyquist frequency.
                    The spectrum is accurately represented.
                </div>
                """), unsafe_allow_html=True)
        
        with tab3:
            st.markdown("### Reconstruction Quality Metrics")
            
            if show_reconstruction:
                # Calculate errors
                error = signal_continuous - signal_reconstructed
                mse = np.mean(error**2)
                rmse = np.sqrt(mse)
                max_error = np.max(np.abs(error))
                correlation = np.corrcoef(signal_continuous, signal_reconstructed)[0,1]
                
                # Error metrics
                error_col1, error_col2, error_col3, error_col4 = st.columns(4)
                
                with error_col1:
                    st.metric("MSE", f"{mse:.6f}", help="Mean Squared Error")
                
                with error_col2:
                    st.metric("RMSE", f"{rmse:.4f}", help="Root Mean Squared Error")
                
                with error_col3:
                    st.metric("Max Error", f"{max_error:.4f}", help="Maximum absolute error")
                
                with error_col4:
                    st.metric("Correlation", f"{correlation:.4f}", help="Pearson correlation coefficient")
                
                if show_error:
                    # Error visualization
                    fig_error = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=("Reconstruction Error Over Time", "Error Histogram"),
                        vertical_spacing=0.15,
                        row_heights=[0.6, 0.4]
                    )
                    
                    # Error over time
                    fig_error.add_trace(
                        go.Scatter(
                            x=t_continuous,
                            y=error,
                            mode='lines',
                            name='Error',
                            line=dict(color='#f44336', width=2),
                            fill='tozeroy',
                            fillcolor='rgba(244, 67, 54, 0.2)'
                        ),
                        row=1, col=1
                    )
                    
                    # Add zero line
                    fig_error.add_hline(y=0, line=dict(color='black', dash='dash', width=1), row=1, col=1)
                    
                    # Error histogram
                    fig_error.add_trace(
                        go.Histogram(
                            x=error,
                            nbinsx=50,
                            name='Error Distribution',
                            marker=dict(color='#9c27b0')
                        ),
                        row=2, col=1
                    )
                    
                    fig_error.update_xaxes(title_text="Time (s)", row=1, col=1)
                    fig_error.update_yaxes(title_text="Error", row=1, col=1)
                    fig_error.update_xaxes(title_text="Error Value", row=2, col=1)
                    fig_error.update_yaxes(title_text="Frequency", row=2, col=1)
                    
                    fig_error.update_layout(
                        height=600,
                        showlegend=False,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig_error, use_container_width=True)
                
                # Quality interpretation
                if correlation > 0.99 and mse < 0.01:
                    quality_msg = "🌟 Excellent reconstruction quality! Nearly perfect match."
                    quality_class = "success-box"
                elif correlation > 0.95 and mse < 0.05:
                    quality_msg = "✅ Good reconstruction quality. Minor distortions present."
                    quality_class = "info-box"
                elif correlation > 0.85:
                    quality_msg = "⚠️ Fair reconstruction. Noticeable distortion in high frequencies."
                    quality_class = "warning-box"
                else:
                    quality_msg = "❌ Poor reconstruction. Significant information loss due to aliasing."
                    quality_class = "danger-box"
                
                st.markdown(textwrap.dedent(f'<div class="{quality_class}"><strong>Quality Assessment:</strong> {quality_msg}</div>'), 
                          unsafe_allow_html=True)
            else:
                st.info("Enable 'Show Reconstructed Signal' to see error analysis.")
        
        with tab4:
            st.markdown("### 📚 Mathematical Foundation")
            
            theory_col1, theory_col2 = st.columns(2)
            
            with theory_col1:
                st.markdown("#### Nyquist-Shannon Sampling Theorem")
                st.latex(r"f_s \geq 2f_{max}")
                st.markdown("""
                **Where:**
                - $f_s$ = Sampling frequency (Hz)
                - $f_{max}$ = Maximum frequency in signal (Hz)
                - $2f_{max}$ = Nyquist rate (minimum sampling rate)
                """)
                
                st.markdown("---")
                
                st.markdown("#### Aliasing Formula")
                st.latex(r"f_{alias} = |f - n \cdot f_s|")
                st.markdown("""
                Where $n$ is chosen to place $f_{alias}$ in the range $[0, f_s/2]$
                
                **Example:** If $f = 30$ Hz and $f_s = 20$ Hz:
                - $f_{alias} = |30 - 1×20| = 10$ Hz
                - The 30 Hz component appears as 10 Hz!
                """)
                
                st.markdown("---")
                
                st.markdown("#### Signal Reconstruction (Whittaker-Shannon)")
                st.latex(r"x(t) = \sum_{n=-\infty}^{\infty} x[n] \cdot \text{sinc}\left(\frac{t-nT}{T}\right)")
                st.markdown("""
                **Where:**
                - $x[n]$ = Sampled values
                - $T = 1/f_s$ = Sampling period
                - $\\text{sinc}(x) = \\sin(\\pi x)/(\\pi x)$
                
                This is the **theoretically perfect** reconstruction method!
                """)
            
            with theory_col2:
                st.markdown("#### Sampling Categories")
                
                st.markdown(textwrap.dedent("""
                <div class="danger-box">
                    <strong>🔴 Undersampling:</strong> $f_s < 2f_{max}$<br>
                    • Aliasing occurs<br>
                    • Information is permanently lost<br>
                    • High frequencies fold back into low frequency range
                </div>
                """), unsafe_allow_html=True)
                
                st.markdown(textwrap.dedent("""
                <div class="warning-box">
                    <strong>🟡 Critical Sampling:</strong> $f_s = 2f_{max}$<br>
                    • Theoretical minimum (rarely used)<br>
                    • No aliasing but sensitive to noise<br>
                    • Requires perfect anti-aliasing filter
                </div>
                """), unsafe_allow_html=True)
                
                st.markdown(textwrap.dedent("""
                    <div class="success-box">
                        <strong>🟢 Oversampling:</strong> $f_s > 2f_{max}$<br>
                        • Perfect reconstruction possible<br>
                        • Provides margin for practical filters<br>
                        • More robust to noise and timing errors
                    </div>
                    """), unsafe_allow_html=True)
                
                st.markdown("---")
                
                st.markdown("#### Reconstruction Methods Compared")
                st.markdown("""
                | Method | Accuracy | Speed | Best For |
                |--------|----------|-------|----------|
                | **Linear** | ⭐⭐ | ⚡⚡⚡ | Real-time, low compute |
                | **Cubic** | ⭐⭐⭐ | ⚡⚡ | Smooth visuals |
                | **Sinc** | ⭐⭐⭐⭐⭐ | ⚡ | Perfect band-limited signals |
                """)
                
                st.markdown("---")
                
                st.markdown("#### Real-World Examples")
                st.markdown("""
                **Audio Applications:**
                - 🎵 CD Audio: 44.1 kHz (captures up to 22.05 kHz)
                - 📞 Phone: 8 kHz (captures up to 4 kHz)
                - 🎙️ Studio: 96-192 kHz (high fidelity)
                
                **Why 44.1 kHz for music?**
                Human hearing: ~20 Hz to 20 kHz. Nyquist says we need >40 kHz. 
                44.1 kHz provides margin for practical filters.
                
                **Image Sampling:**
                - 📷 Camera sensors: Pixel spacing = spatial sampling
                - 🖥️ Displays: Refresh rate must exceed 2× content frequency
                """)
    
    # Interactive demonstration section
    st.markdown("---")
    st.markdown('<div class="section-header">🎮 Interactive Demonstrations</div>', unsafe_allow_html=True)
    
    demo_col1, demo_col2 = st.columns(2)
    
    with demo_col1:
        st.markdown(textwrap.dedent("""
    <div class="info-box">
        <strong>🧪 Try These Experiments:</strong><br><br>
        <strong> Aliasing Effect:</strong><br>
        • Set frequencies to 5, 25, 45 Hz<br>
        • Set sampling to 40 Hz<br>
        • Watch 25 Hz and 45 Hz alias!<br>
<br>
        <strong> Perfect Reconstruction:</strong><br>
        • Keep default frequencies (5, 15, 25 Hz)<br>
        • Set sampling to 100+ Hz<br>
        • Try different reconstruction methods<br>
   <br>     
        <strong>3. Critical Sampling:</strong><br>
        • Set sampling = 2 × max frequency<br>
        • Observe the edge case behavior
    </div>
    """), unsafe_allow_html=True)
    
    with demo_col2:
        st.markdown(textwrap.dedent(f"""
        <div class="warning-box">
            <strong>⚠️ Current Configuration Analysis:</strong><br><br>
            <strong>Your Settings:</strong><br>
            • Max frequency: <strong>{max_freq} Hz</strong><br>
            • Sampling rate: <strong>{sampling_rate} Hz</strong><br>
            • Nyquist rate: <strong>{nyquist_rate} Hz</strong><br>
            • Ratio: <strong>{sampling_ratio:.2f}x</strong> Nyquist<br>
            <br>            
            <strong>Status:</strong><br>
            {"✅ All frequencies are properly sampled!" if not aliased_freqs else 
             f"⚠️ {len(aliased_freqs)} frequency component(s) will be aliased!"}
        </div>
        """), unsafe_allow_html=True)
    
    # Key takeaways
    st.markdown("---")
    st.markdown('<div class="section-header">💡 Key Takeaways</div>', unsafe_allow_html=True)
    
    takeaway_cols = st.columns(3)
    
    with takeaway_cols[0]:
        st.markdown(textwrap.dedent("""
        <div class="info-box">
            <strong>🎯 The Rule:</strong><br>
            Sample at <strong>at least 2× the highest frequency</strong> you want to capture. 
            In practice, use 2.5-4× for safety margin.
        </div>
        """), unsafe_allow_html=True)
    
    with takeaway_cols[1]:
        st.markdown(textwrap.dedent("""
        <div class="warning-box">
            <strong>⚠️ The Danger:</strong><br>
            Undersample and high frequencies will <strong>masquerade as low frequencies</strong>. 
            This information loss is permanent!
        </div>
        """), unsafe_allow_html=True)
    
    with takeaway_cols[2]:
        st.markdown(textwrap.dedent("""
        <div class="success-box">
            <strong>✅ The Solution:</strong><br>
            Use <strong>anti-aliasing filters</strong> before sampling to remove frequencies 
            above Nyquist limit.
        </div>
        """), unsafe_allow_html=True)
    
    # Footer with additional resources
    st.markdown("---")
    with st.expander("📖 Additional Resources & Further Reading"):
        st.markdown("""
        ### Learn More About Signal Processing
        
        **Recommended Topics:**
        - Anti-aliasing filters (Low-pass filters before ADC)
        - Quantization effects (bit depth impact)
        - Window functions in FFT analysis
        - Oversampling and decimation
        - Delta-Sigma modulation
                    

        * **[Nyquist–Shannon Sampling Theorem](https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem):** The core mathematical principle visualized here.
        * **[Aliasing](https://en.wikipedia.org/wiki/Aliasing):** Detailed explanation of the "wagon wheel" effect in data.
        * **[Anti-aliasing Filter](https://en.wikipedia.org/wiki/Anti-aliasing_filter):** How engineers prevent these artifacts in real life.
                    

        **Related Theorems:**
        - Whittaker-Shannon interpolation formula
        - Kotelnikov theorem (Russian equivalent)
        - Sampling theorem for bandpass signals
        
        **Practical Applications:**
        - Software Defined Radio (SDR)
        - Digital Audio Workstations (DAWs)
        - Oscilloscopes and data acquisition
        - Medical imaging (CT, MRI)
        - Seismic data processing
        """)
    st.markdown("---")
    with st.expander("🧠 Test Your Knowledge"):
        st.write("**Question:** If you have a signal with a maximum frequency of 50Hz, what is the minimum sampling rate required to avoid aliasing?")
        
        # Unique key is important so it doesn't conflict with other widgets
        answer = st.radio("Select answer:", ["25 Hz", "50 Hz", "100 Hz", "200 Hz"], key="quiz_1")
        
        if st.button("Check Answer", key="check_1"):
            if answer == "100 Hz":
                st.success("Correct! The Nyquist rate is 2 × f_max (2 × 50 = 100 Hz).")
                st.balloons()
            else:
                st.error("Not quite. Remember the Nyquist Theorem: f_s ≥ 2 × f_max.")
if __name__ == "__main__":
    main()