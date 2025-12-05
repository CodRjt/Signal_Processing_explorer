# Create Sampling Rate Visualization page
sampling_page_code = '''
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.fft import fft, fftfreq
from scipy import signal as sp_signal
from matplotlib import use
use('Agg')

st.set_page_config(page_title="Sampling Rate Visualization", page_icon="📊", layout="wide")

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
    
    # Interpolate the continuous signal at sampling points
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
        # Ideal sinc interpolation (simplified)
        dt = t_sampled[1] - t_sampled[0]
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
    
    # Only keep positive frequencies
    xf = xf[:N//2]
    yf = np.abs(yf[:N//2]) * 2 / N
    
    return xf, yf

def main():
    st.title("📊 Sampling Rate Visualization & Nyquist Theorem")
    
    st.markdown("""
    Explore the **Nyquist Sampling Theorem** and observe the effects of different sampling rates
    on signal reconstruction and aliasing. The theorem states that to perfectly reconstruct a 
    signal, the sampling frequency must be at least twice the highest frequency component.
    """)
    
    # Controls
    control_col, viz_col = st.columns([1, 2])
    
    with control_col:
        st.markdown("### 🎛️ Signal Parameters")
        
        # Signal parameters
        freq1 = st.slider("Frequency 1 (Hz)", 1, 20, 5, 1)
        freq2 = st.slider("Frequency 2 (Hz)", 10, 40, 15, 1)
        freq3 = st.slider("Frequency 3 (Hz)", 20, 50, 25, 1)
        
        max_freq = max(freq1, freq2, freq3)
        nyquist_rate = 2 * max_freq
        
        st.info(f"**Maximum Frequency:** {max_freq} Hz")
        st.info(f"**Nyquist Rate:** {nyquist_rate} Hz")
        
        # Sampling parameters
        st.markdown("### 📡 Sampling Parameters")
        sampling_rate = st.slider("Sampling Rate (Hz)", 10, 200, 50, 10)
        
        # Show sampling ratio
        sampling_ratio = sampling_rate / nyquist_rate
        if sampling_ratio >= 1:
            st.success(f"**Sampling Ratio:** {sampling_ratio:.2f} (Above Nyquist)")
        else:
            st.error(f"**Sampling Ratio:** {sampling_ratio:.2f} (Below Nyquist - Aliasing!)")
        
        # Reconstruction method
        reconstruction_method = st.selectbox(
            "Reconstruction Method", 
            ['linear', 'cubic', 'sinc'],
            help="Method used to reconstruct continuous signal from samples"
        )
        
        # Display options
        st.markdown("### 📊 Display Options")
        show_samples = st.checkbox("Show Sample Points", value=True)
        show_spectrum = st.checkbox("Show Frequency Spectrum", value=True)
        show_reconstruction = st.checkbox("Show Reconstructed Signal", value=True)
    
    with viz_col:
        # Generate signals
        t_continuous = np.linspace(0, 2, 2000)  # High resolution "continuous" signal
        signal_continuous = generate_test_signal(t_continuous, freq1, freq2, freq3)
        
        # Sample the signal
        t_sampled, signal_sampled = sample_signal(t_continuous, signal_continuous, sampling_rate)
        
        # Reconstruct signal
        if show_reconstruction:
            signal_reconstructed = reconstruct_signal(
                t_sampled, signal_sampled, t_continuous, reconstruction_method
            )
        
        # Time domain visualization
        st.markdown("### 📈 Time Domain Analysis")
        
        fig_time = go.Figure()
        
        # Original signal
        fig_time.add_trace(go.Scatter(
            x=t_continuous,
            y=signal_continuous,
            mode='lines',
            name='Original Signal',
            line=dict(color='blue', width=2)
        ))
        
        # Sampled points
        if show_samples:
            fig_time.add_trace(go.Scatter(
                x=t_sampled,
                y=signal_sampled,
                mode='markers',
                name='Sampled Points',
                marker=dict(color='red', size=8, symbol='circle')
            ))
        
        # Reconstructed signal
        if show_reconstruction:
            fig_time.add_trace(go.Scatter(
                x=t_continuous,
                y=signal_reconstructed,
                mode='lines',
                name=f'Reconstructed ({reconstruction_method})',
                line=dict(color='green', width=2, dash='dash')
            ))
        
        fig_time.update_layout(
            title=f"Signal Sampling at {sampling_rate} Hz",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            height=400,
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_time, use_container_width=True)
        
        # Frequency domain analysis
        if show_spectrum:
            st.markdown("### 🌊 Frequency Domain Analysis")
            
            # Analyze original and sampled spectra
            freq_orig, spectrum_orig = analyze_spectrum(signal_continuous, 1000)  # High fs for "continuous"
            freq_sampled, spectrum_sampled = analyze_spectrum(signal_sampled, sampling_rate)
            
            fig_freq = go.Figure()
            
            # Original spectrum
            fig_freq.add_trace(go.Scatter(
                x=freq_orig,
                y=spectrum_orig,
                mode='lines',
                name='Original Spectrum',
                line=dict(color='blue', width=2)
            ))
            
            # Sampled spectrum
            fig_freq.add_trace(go.Scatter(
                x=freq_sampled,
                y=spectrum_sampled,
                mode='lines+markers',
                name='Sampled Spectrum',
                line=dict(color='red', width=2),
                marker=dict(size=6)
            ))
            
            # Mark Nyquist frequency
            nyquist_freq = sampling_rate / 2
            fig_freq.add_vline(
                x=nyquist_freq,
                line=dict(color='orange', dash='dash', width=2),
                annotation_text=f"Nyquist: {nyquist_freq} Hz"
            )
            
            # Mark original frequency components
            for f, name in [(freq1, 'f1'), (freq2, 'f2'), (freq3, 'f3')]:
                fig_freq.add_vline(
                    x=f,
                    line=dict(color='green', dash='dot', width=1),
                    annotation_text=f"{name}: {f}Hz"
                )
            
            fig_freq.update_layout(
                title="Frequency Spectrum Comparison",
                xaxis_title="Frequency (Hz)",
                yaxis_title="Magnitude",
                height=400,
                showlegend=True
            )
            
            fig_freq.update_xaxes(range=[0, min(100, sampling_rate)])
            st.plotly_chart(fig_freq, use_container_width=True)
    
    # Error analysis
    st.markdown("---")
    st.markdown("### 📊 Reconstruction Error Analysis")
    
    if show_reconstruction:
        # Calculate reconstruction error
        error = signal_continuous - signal_reconstructed
        mse = np.mean(error**2)
        max_error = np.max(np.abs(error))
        
        error_col1, error_col2, error_col3 = st.columns(3)
        
        with error_col1:
            st.metric("Mean Squared Error", f"{mse:.6f}")
        
        with error_col2:
            st.metric("Maximum Error", f"{max_error:.4f}")
        
        with error_col3:
            correlation = np.corrcoef(signal_continuous, signal_reconstructed)[0,1]
            st.metric("Correlation", f"{correlation:.4f}")
        
        # Error visualization
        fig_error = go.Figure()
        fig_error.add_trace(go.Scatter(
            x=t_continuous,
            y=error,
            mode='lines',
            name='Reconstruction Error',
            line=dict(color='red', width=2)
        ))
        
        fig_error.update_layout(
            title="Reconstruction Error Over Time",
            xaxis_title="Time (s)",
            yaxis_title="Error",
            height=300
        )
        
        st.plotly_chart(fig_error, use_container_width=True)
    
    # Aliasing demonstration
    st.markdown("---")
    st.markdown("### ⚠️ Aliasing Demonstration")
    
    if sampling_ratio < 1:
        st.warning("**Aliasing Detected!** The sampling rate is below the Nyquist rate.")
        
        # Show which frequencies are aliased
        aliased_freqs = []
        for f in [freq1, freq2, freq3]:
            if f > sampling_rate / 2:
                # Calculate alias frequency
                alias_freq = abs(f - sampling_rate * round(f / sampling_rate))
                if alias_freq > sampling_rate / 2:
                    alias_freq = sampling_rate - alias_freq
                aliased_freqs.append((f, alias_freq))
        
        if aliased_freqs:
            st.markdown("**Aliased Frequencies:**")
            for orig_f, alias_f in aliased_freqs:
                st.markdown(f"- {orig_f} Hz → appears as {alias_f:.1f} Hz")
    else:
        st.success("No aliasing - sampling rate is sufficient!")
    
    # Theory section
    st.markdown("---")
    st.markdown("### 📚 Theory & Mathematics")
    
    theory_col1, theory_col2 = st.columns(2)
    
    with theory_col1:
        st.markdown("**Nyquist-Shannon Sampling Theorem:**")
        st.latex(r"f_s \\geq 2f_{max}")
        st.markdown("Where:")
        st.markdown("- $f_s$ = Sampling frequency")
        st.markdown("- $f_{max}$ = Maximum frequency in signal")
        
        st.markdown("**Aliasing Frequency:**")
        st.latex(r"f_{alias} = |f - nf_s|")
        st.markdown("Where n is chosen to minimize $f_{alias}$")
        
        st.markdown("**Reconstruction (Sinc Interpolation):**")
        st.latex(r"x(t) = \\sum_{n} x[n] \\cdot sinc\\left(\\frac{t-nT}{T}\\right)")
    
    with theory_col2:
        st.markdown("**Key Concepts:**")
        st.markdown("""
        - **Undersampling**: $f_s < 2f_{max}$ → Aliasing occurs
        - **Critical Sampling**: $f_s = 2f_{max}$ → Just sufficient
        - **Oversampling**: $f_s > 2f_{max}$ → Perfect reconstruction possible
        
        **Reconstruction Methods:**
        - **Linear**: Simple, fast, some distortion
        - **Cubic**: Smoother, better for visualization
        - **Sinc**: Theoretically perfect for band-limited signals
        
        **Applications:**
        - Audio sampling (44.1 kHz for 20 kHz audio)
        - Image sensors (pixel spacing)
        - Communication systems
        """)

if __name__ == "__main__":
    main()
'''

# Save Sampling page
with open('pages/3_📊_Sampling_Rate_Visualization.py', 'w') as f:
    f.write(sampling_page_code)

print("✅ Created Sampling Rate Visualization page")
print("File size:", len(sampling_page_code), "characters")