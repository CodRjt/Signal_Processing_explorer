
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy.fft import fft, fftfreq
from scipy import signal as sp_signal
from matplotlib import use
use('Agg')

st.set_page_config(page_title="Real-life DSP", page_icon="🌍", layout="wide")

def generate_audio_signal(duration=1.0, fs=44100, freq=440):
    """Generate audio signal with quantization noise"""
    t = np.linspace(0, duration, int(fs * duration), False)
    signal = np.sin(2 * np.pi * freq * t)
    # Add harmonics for richer sound
    signal += 0.3 * np.sin(2 * np.pi * 2 * freq * t)
    signal += 0.15 * np.sin(2 * np.pi * 3 * freq * t)
    return t, signal / np.max(np.abs(signal))  # Normalize

def quantize_audio(signal, bits):
    """Quantize audio signal to specified bit depth"""
    levels = 2**bits
    quantized = np.round(signal * (levels/2)) / (levels/2)
    quantized = np.clip(quantized, -1, 1)
    return quantized

def calculate_quantization_noise(original, quantized):
    """Calculate quantization error/noise"""
    return original - quantized

def generate_test_image(size=256):
    """Generate a test image with various features"""
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))

    # Create gradient
    gradient = (x + 1) / 2

    # Add circular patterns
    r = np.sqrt(x**2 + y**2)
    circles = np.sin(10 * r) * np.exp(-r**2)

    # Combine
    image = 0.6 * gradient + 0.4 * circles
    image = (image - image.min()) / (image.max() - image.min())

    return image

def downsample_image(image, factor):
    """Downsample image by given factor"""
    h, w = image.shape
    new_h, new_w = h // factor, w // factor

    # Simple decimation (pick every nth pixel)
    downsampled = image[::factor, ::factor]

    return downsampled

def upsample_image(image, target_shape, method='nearest'):
    """Upsample image to target shape"""
    from scipy.ndimage import zoom

    h_factor = target_shape[0] / image.shape[0]
    w_factor = target_shape[1] / image.shape[1]

    if method == 'nearest':
        order = 0
    elif method == 'bilinear':
        order = 1
    elif method == 'cubic':
        order = 3

    upsampled = zoom(image, (h_factor, w_factor), order=order)

    return upsampled

def apply_lowpass_filter(signal, cutoff_freq, fs, order=5):
    """Apply lowpass filter for anti-aliasing"""
    nyquist = fs / 2
    normal_cutoff = cutoff_freq / nyquist
    b, a = sp_signal.butter(order, normal_cutoff, btype='low', analog=False)
    filtered = sp_signal.filtfilt(b, a, signal)
    return filtered

def main():
    st.title("🌍 Real-life DSP Applications & Demonstrations")

    st.markdown("""
    Explore practical applications of digital signal processing in everyday technology.
    This module demonstrates audio quantization, image processing, and signal reconstruction
    techniques used in real-world systems.
    """)

    # Tabbed interface for different applications
    tab1, tab2, tab3 = st.tabs([
        "🎵 Audio Quantization",
        "🖼️ Image Processing",
        "🔄 Signal Reconstruction"
    ])

    # TAB 1: Audio Quantization
    with tab1:
        st.markdown("### 🎵 Audio Signal Quantization & Noise Analysis")

        st.markdown("""
        Digital audio requires quantization of continuous amplitude values. Lower bit depths
        result in **quantization noise** that can be heard as distortion or "graininess."
        """)

        audio_col1, audio_col2 = st.columns([1, 2])

        with audio_col1:
            st.markdown("#### ⚙️ Audio Parameters")

            # Audio parameters
            frequency = st.slider("Audio Frequency (Hz)", 100, 2000, 440, 50)
            sample_rate = st.selectbox("Sample Rate", [8000, 22050, 44100, 48000], index=2)
            bit_depth = st.selectbox("Bit Depth", [4, 8, 12, 16, 24], index=3)

            duration = st.slider("Signal Duration (s)", 0.1, 2.0, 0.5, 0.1)

            # Quality info
            st.markdown("#### 📊 Quality Metrics")
            theoretical_snr = 6.02 * bit_depth + 1.76
            st.info(f"**Theoretical SNR:** {theoretical_snr:.2f} dB")

            # Common standards
            st.markdown("""
            **Common Standards:**
            - Telephone: 8-bit, 8 kHz
            - CD Audio: 16-bit, 44.1 kHz
            - DVD Audio: 24-bit, 48-96 kHz
            """)

        with audio_col2:
            st.markdown("#### 📊 Waveform Analysis")

            # Generate audio
            t, signal_original = generate_audio_signal(duration, sample_rate, frequency)
            signal_quantized = quantize_audio(signal_original, bit_depth)
            quantization_noise = calculate_quantization_noise(signal_original, signal_quantized)

            # Create visualization
            fig_audio = go.Figure()

            # Show zoomed-in portion
            zoom_samples = min(500, len(t))

            fig_audio.add_trace(go.Scatter(
                x=t[:zoom_samples],
                y=signal_original[:zoom_samples],
                mode='lines',
                name='Original (Continuous)',
                line=dict(color='blue', width=2)
            ))

            fig_audio.add_trace(go.Scatter(
                x=t[:zoom_samples],
                y=signal_quantized[:zoom_samples],
                mode='lines+markers',
                name=f'Quantized ({bit_depth}-bit)',
                line=dict(color='red', width=1),
                marker=dict(size=3)
            ))

            fig_audio.update_layout(
                title=f"Audio Waveform: {frequency} Hz at {bit_depth}-bit Quantization",
                xaxis_title="Time (s)",
                yaxis_title="Amplitude",
                height=350,
                showlegend=True
            )

            st.plotly_chart(fig_audio, use_container_width=True)

            # Quantization noise analysis
            st.markdown("#### 🔊 Quantization Noise")

            fig_noise = go.Figure()

            fig_noise.add_trace(go.Scatter(
                x=t[:zoom_samples],
                y=quantization_noise[:zoom_samples],
                mode='lines',
                name='Quantization Error',
                line=dict(color='red', width=1),
                fill='tozeroy'
            ))

            fig_noise.update_layout(
                title="Quantization Noise (Error Signal)",
                xaxis_title="Time (s)",
                yaxis_title="Error Amplitude",
                height=250
            )

            st.plotly_chart(fig_noise, use_container_width=True)

            # Calculate actual SNR
            signal_power = np.mean(signal_original**2)
            noise_power = np.mean(quantization_noise**2)
            if noise_power > 0:
                actual_snr = 10 * np.log10(signal_power / noise_power)
            else:
                actual_snr = float('inf')

            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

            with metrics_col1:
                st.metric("Actual SNR", f"{actual_snr:.2f} dB")

            with metrics_col2:
                st.metric("RMS Noise", f"{np.sqrt(noise_power):.6f}")

            with metrics_col3:
                st.metric("Peak Error", f"{np.max(np.abs(quantization_noise)):.6f}")

    # TAB 2: Image Processing
    with tab2:
        st.markdown("### 🖼️ Image Downsampling & Upsampling")

        st.markdown("""
        Image resizing involves spatial sampling. Downsampling reduces resolution, while
        upsampling attempts to recreate detail through interpolation.
        """)

        image_col1, image_col2 = st.columns([1, 2])

        with image_col1:
            st.markdown("#### ⚙️ Image Parameters")

            # Image processing parameters
            downsample_factor = st.slider("Downsample Factor", 2, 8, 4, 1)
            interpolation_method = st.selectbox(
                "Upsampling Method",
                ['nearest', 'bilinear', 'cubic']
            )

            show_difference = st.checkbox("Show Difference Map", value=True)

            # Info
            st.markdown("#### 📊 Resolution Info")
            original_size = 256
            downsampled_size = original_size // downsample_factor

            st.info(f"**Original:** {original_size}×{original_size}")
            st.info(f"**Downsampled:** {downsampled_size}×{downsampled_size}")
            st.info(f"**Compression:** {downsample_factor**2}x fewer pixels")

        with image_col2:
            st.markdown("#### 🖼️ Image Processing Pipeline")

            # Generate test image
            original_image = generate_test_image(original_size)

            # Downsample
            downsampled = downsample_image(original_image, downsample_factor)

            # Upsample back
            upsampled = upsample_image(downsampled, original_image.shape, interpolation_method)

            # Create subplot
            fig_image, axes = plt.subplots(1, 4, figsize=(16, 4))

            images = [
                (original_image, "Original"),
                (downsampled, f"Downsampled ({downsample_factor}x)"),
                (upsampled, f"Upsampled ({interpolation_method})"),
                (np.abs(original_image - upsampled), "Difference")
            ]

            for i, (img, title) in enumerate(images):
                if i == 3 and not show_difference:
                    axes[i].axis('off')
                    continue

                im = axes[i].imshow(img, cmap='viridis' if i < 3 else 'hot', 
                                   interpolation='nearest')
                axes[i].set_title(title, fontsize=11, fontweight='bold')
                axes[i].axis('off')
                plt.colorbar(im, ax=axes[i], shrink=0.8)

            plt.tight_layout()
            st.pyplot(fig_image)

            # Quality metrics
            mse = np.mean((original_image - upsampled)**2)
            psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
            correlation = np.corrcoef(original_image.flatten(), upsampled.flatten())[0,1]

            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:
                st.metric("MSE", f"{mse:.6f}")

            with metric_col2:
                st.metric("PSNR", f"{psnr:.2f} dB")

            with metric_col3:
                st.metric("Correlation", f"{correlation:.4f}")

    # TAB 3: Signal Reconstruction
    with tab3:
        st.markdown("### 🔄 Signal Reconstruction & Anti-Aliasing")

        st.markdown("""
        Reconstruction filters are essential for converting discrete signals back to continuous form.
        **Anti-aliasing filters** prevent high-frequency artifacts before downsampling.
        """)

        recon_col1, recon_col2 = st.columns([1, 2])

        with recon_col1:
            st.markdown("#### ⚙️ Reconstruction Parameters")

            # Parameters
            signal_freq = st.slider("Signal Frequency (Hz)", 5, 50, 15, 5)
            original_fs = st.slider("Original Sample Rate (Hz)", 100, 500, 200, 50)
            target_fs = st.slider("Target Sample Rate (Hz)", 20, 150, 50, 10)

            use_antialiasing = st.checkbox("Use Anti-aliasing Filter", value=True)

            # Nyquist info
            nyquist_original = original_fs / 2
            nyquist_target = target_fs / 2

            st.markdown("#### 📊 Nyquist Frequencies")
            st.info(f"**Original:** {nyquist_original} Hz")
            st.info(f"**Target:** {nyquist_target} Hz")

            if signal_freq > nyquist_target:
                st.warning(f"⚠️ Signal frequency ({signal_freq} Hz) > Nyquist ({nyquist_target} Hz)")
                st.markdown("**Aliasing will occur without filtering!**")

        with recon_col2:
            st.markdown("#### 📊 Reconstruction Analysis")

            # Generate high-rate signal
            duration = 2.0
            t_original = np.linspace(0, duration, int(original_fs * duration), False)
            signal_original = np.sin(2 * np.pi * signal_freq * t_original)

            # Apply anti-aliasing filter if enabled
            if use_antialiasing:
                cutoff = target_fs / 2 * 0.8  # 80% of Nyquist
                signal_filtered = apply_lowpass_filter(signal_original, cutoff, 
                                                      original_fs, order=6)
            else:
                signal_filtered = signal_original

            # Downsample
            decimation_factor = int(original_fs / target_fs)
            signal_downsampled = signal_filtered[::decimation_factor]
            t_downsampled = t_original[::decimation_factor]

            # Reconstruction
            t_reconstructed = t_original
            signal_reconstructed = np.interp(t_reconstructed, t_downsampled, 
                                            signal_downsampled)

            # Visualization
            fig_recon = go.Figure()

            fig_recon.add_trace(go.Scatter(
                x=t_original,
                y=signal_original,
                mode='lines',
                name='Original Signal',
                line=dict(color='blue', width=2, dash='dash'),
                opacity=0.7
            ))

            if use_antialiasing:
                fig_recon.add_trace(go.Scatter(
                    x=t_original,
                    y=signal_filtered,
                    mode='lines',
                    name='After Anti-aliasing',
                    line=dict(color='green', width=2),
                    opacity=0.8
                ))

            fig_recon.add_trace(go.Scatter(
                x=t_downsampled,
                y=signal_downsampled,
                mode='markers',
                name=f'Samples ({target_fs} Hz)',
                marker=dict(color='red', size=8, symbol='circle')
            ))

            fig_recon.add_trace(go.Scatter(
                x=t_reconstructed,
                y=signal_reconstructed,
                mode='lines',
                name='Reconstructed Signal',
                line=dict(color='orange', width=2, dash='dot')
            ))

            fig_recon.update_layout(
                title="Signal Reconstruction with Anti-aliasing",
                xaxis_title="Time (s)",
                yaxis_title="Amplitude",
                height=400,
                showlegend=True
            )

            st.plotly_chart(fig_recon, use_container_width=True)

            # Frequency domain
            st.markdown("#### 🌊 Frequency Domain")

            # FFT of original and reconstructed
            from scipy.fft import fft, fftfreq

            N_orig = len(signal_original)
            freq_orig = fftfreq(N_orig, 1/original_fs)[:N_orig//2]
            fft_orig = np.abs(fft(signal_original))[:N_orig//2] * 2/N_orig

            N_recon = len(signal_reconstructed)
            freq_recon = fftfreq(N_recon, 1/original_fs)[:N_recon//2]
            fft_recon = np.abs(fft(signal_reconstructed))[:N_recon//2] * 2/N_recon

            fig_freq = go.Figure()

            fig_freq.add_trace(go.Scatter(
                x=freq_orig,
                y=fft_orig,
                mode='lines',
                name='Original Spectrum',
                line=dict(color='blue', width=2)
            ))

            fig_freq.add_trace(go.Scatter(
                x=freq_recon,
                y=fft_recon,
                mode='lines',
                name='Reconstructed Spectrum',
                line=dict(color='orange', width=2, dash='dash')
            ))

            # Mark Nyquist frequency
            fig_freq.add_vline(
                x=nyquist_target,
                line=dict(color='red', dash='dash', width=2),
                annotation_text=f"Nyquist: {nyquist_target} Hz"
            )

            fig_freq.update_layout(
                title="Frequency Spectrum Comparison",
                xaxis_title="Frequency (Hz)",
                yaxis_title="Magnitude",
                height=350
            )

            fig_freq.update_xaxes(range=[0, original_fs/2])
            st.plotly_chart(fig_freq, use_container_width=True)

    # Summary section
    st.markdown("---")
    st.markdown("### 📚 Summary of Real-World Applications")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.markdown("""
        **Audio Processing:**
        - MP3/AAC compression
        - Voice codecs
        - Audio effects (reverb, EQ)
        - Noise cancellation
        """)

    with summary_col2:
        st.markdown("""
        **Image Processing:**
        - JPEG compression
        - Photo resizing
        - Image enhancement
        - Computer vision
        """)

    with summary_col3:
        st.markdown("""
        **Signal Reconstruction:**
        - DAC (Digital-to-Analog)
        - Video upscaling
        - Interpolation filters
        - Bandwidth optimization
        """)

if __name__ == "__main__":
    main()
