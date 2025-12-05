# Create Fingerprint Scanning Simulation page
fingerprint_page_code = '''
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib import use
use('Agg')
from scipy import ndimage
import cv2

st.set_page_config(page_title="Fingerprint Scanning", page_icon="🔍", layout="wide")

def generate_fingerprint_pattern(size=200, noise_level=0.1):
    """Generate a simulated fingerprint pattern"""
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    
    # Create concentric circular pattern (simplified fingerprint)
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    # Create ridge pattern
    ridges = np.sin(20 * r + 5 * np.sin(3 * theta)) * np.exp(-r**2)
    
    # Add some spiral characteristics
    spiral = np.sin(10 * theta + 15 * r)
    
    # Combine patterns
    fingerprint = 0.7 * ridges + 0.3 * spiral
    
    # Add noise
    noise = np.random.normal(0, noise_level, fingerprint.shape)
    fingerprint += noise
    
    # Normalize to [0, 1]
    fingerprint = (fingerprint - fingerprint.min()) / (fingerprint.max() - fingerprint.min())
    
    return fingerprint

def simulate_analog_pickup(fingerprint, sensor_resolution=0.95):
    """Simulate analog sensor pickup with some degradation"""
    # Apply Gaussian blur to simulate sensor limitations
    blurred = ndimage.gaussian_filter(fingerprint, sigma=1.0)
    
    # Add sensor noise
    sensor_noise = np.random.normal(0, 0.05, blurred.shape)
    analog_signal = blurred + sensor_noise
    
    # Apply sensor resolution limitations
    analog_signal = analog_signal * sensor_resolution
    
    return np.clip(analog_signal, 0, 1)

def quantize_signal(signal, bits=8):
    """Quantize the analog signal to digital levels"""
    levels = 2**bits
    quantized = np.round(signal * (levels - 1)) / (levels - 1)
    return quantized

def sample_signal(signal, sampling_factor=1.0):
    """Simulate spatial sampling (downsampling/upsampling)"""
    if sampling_factor < 1.0:
        # Downsample
        h, w = signal.shape
        new_h, new_w = int(h * sampling_factor), int(w * sampling_factor)
        downsampled = cv2.resize(signal, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # Upsample back to original size for comparison
        return cv2.resize(downsampled, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        return signal

def main():
    st.title("🔍 Fingerprint Scanning Simulation")
    
    st.markdown("""
    This module simulates the process of fingerprint scanning, from analog signal acquisition 
    to digital processing. Observe how **sampling**, **quantization**, and **noise** affect 
    the quality of biometric signal processing.
    """)
    
    # Controls
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ Scanning Parameters")
        
        # Noise controls
        noise_level = st.slider("Sensor Noise Level", 0.0, 0.3, 0.1, 0.01)
        sensor_resolution = st.slider("Sensor Resolution", 0.5, 1.0, 0.95, 0.01)
        
        # Quantization
        quantization_bits = st.selectbox("Quantization Bits", [2, 4, 6, 8, 10, 12], index=3)
        
        # Sampling
        sampling_factor = st.slider("Spatial Sampling Factor", 0.2, 1.0, 1.0, 0.1)
        
        # Regenerate button
        if st.button("🔄 Generate New Fingerprint", use_container_width=True):
            st.session_state.fingerprint_seed = np.random.randint(0, 1000)
        
        # Enhancement
        st.markdown("### 🔧 Signal Enhancement")
        enhance_contrast = st.checkbox("Enhance Contrast", value=False)
        apply_filter = st.checkbox("Apply Median Filter", value=False)
    
    with col2:
        st.markdown("### 📊 Signal Processing Pipeline")
        
        # Set random seed for reproducibility
        if 'fingerprint_seed' not in st.session_state:
            st.session_state.fingerprint_seed = 42
        
        np.random.seed(st.session_state.fingerprint_seed)
        
        # Generate original fingerprint
        original = generate_fingerprint_pattern(noise_level=0.02)
        
        # Simulate analog pickup
        analog = simulate_analog_pickup(original, sensor_resolution)
        
        # Add noise
        noisy_analog = analog + np.random.normal(0, noise_level, analog.shape)
        noisy_analog = np.clip(noisy_analog, 0, 1)
        
        # Quantization
        quantized = quantize_signal(noisy_analog, quantization_bits)
        
        # Sampling
        sampled = sample_signal(quantized, sampling_factor)
        
        # Enhancement
        final_signal = sampled.copy()
        if enhance_contrast:
            final_signal = np.clip((final_signal - 0.5) * 2 + 0.5, 0, 1)
        
        if apply_filter:
            # Apply median filter for noise reduction
            final_signal = ndimage.median_filter(final_signal, size=3)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Fingerprint Signal Processing Pipeline", fontsize=16, fontweight='bold')
        
        stages = [
            (original, "1. Original Pattern"),
            (analog, "2. Analog Sensor"),
            (noisy_analog, "3. Noisy Analog"),
            (quantized, "4. Quantized Digital"),
            (sampled, "5. Sampled"),
            (final_signal, "6. Enhanced")
        ]
        
        for i, (signal, title) in enumerate(stages):
            ax = axes[i//3, i%3]
            im = ax.imshow(signal, cmap='gray', interpolation='nearest')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Signal quality metrics
        st.markdown("### 📈 Quality Metrics")
        
        # Calculate SNR and other metrics
        def calculate_snr(original, processed):
            signal_power = np.mean(original**2)
            noise_power = np.mean((original - processed)**2)
            if noise_power > 0:
                return 10 * np.log10(signal_power / noise_power)
            return float('inf')
        
        def calculate_mse(original, processed):
            return np.mean((original - processed)**2)
        
        snr = calculate_snr(original, final_signal)
        mse = calculate_mse(original, final_signal)
        correlation = np.corrcoef(original.flatten(), final_signal.flatten())[0,1]
        
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            st.metric("Signal-to-Noise Ratio", f"{snr:.2f} dB")
        
        with metrics_col2:
            st.metric("Mean Squared Error", f"{mse:.6f}")
        
        with metrics_col3:
            st.metric("Correlation", f"{correlation:.4f}")
    
    # Cross-section analysis
    st.markdown("---")
    st.markdown("### 📊 Cross-Section Analysis")
    
    # Take horizontal cross-section through middle
    middle_row = original.shape[0] // 2
    x_axis = np.arange(original.shape[1])
    
    cross_sections = {
        'Original': original[middle_row, :],
        'Analog': analog[middle_row, :],
        'Noisy': noisy_analog[middle_row, :],
        'Quantized': quantized[middle_row, :],
        'Final': final_signal[middle_row, :]
    }
    
    # Create interactive plot
    fig_cross = go.Figure()
    
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    for i, (name, signal) in enumerate(cross_sections.items()):
        fig_cross.add_trace(go.Scatter(
            x=x_axis,
            y=signal,
            mode='lines',
            name=name,
            line=dict(color=colors[i], width=2)
        ))
    
    fig_cross.update_layout(
        title="Signal Cross-Section Comparison",
        xaxis_title="Pixel Position",
        yaxis_title="Signal Amplitude",
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig_cross, use_container_width=True)
    
    # Theory section
    st.markdown("---")
    st.markdown("### 📚 Theory & Applications")
    
    theory_col1, theory_col2 = st.columns(2)
    
    with theory_col1:
        st.markdown("""
        **Signal Processing Chain:**
        
        1. **Optical/Capacitive Sensing**: Physical ridges create varying signal intensities
        2. **Analog-to-Digital Conversion**: Continuous signal → discrete samples
        3. **Quantization**: Amplitude discretization based on bit depth
        4. **Spatial Sampling**: Resolution determined by sensor pixel density
        5. **Enhancement**: Filtering and contrast adjustment
        6. **Feature Extraction**: Ridge patterns, minutiae detection
        """)
        
        st.markdown("""
        **Key Challenges:**
        - **Noise**: Environmental interference, sensor limitations
        - **Resolution**: Balance between detail and processing speed
        - **Dynamic Range**: Capturing both light and dark features
        - **Distortion**: Pressure, rotation, partial prints
        """)
    
    with theory_col2:
        st.markdown("**Quantization Effects:**")
        st.latex(r"Q(x) = \Delta \cdot \lfloor \frac{x}{\Delta} + 0.5 \\rfloor")
        st.markdown("Where $\\Delta = \\frac{1}{2^b - 1}$ for b-bit quantization")
        
        st.markdown("**Quality Metrics:**")
        st.latex(r"SNR = 10 \log_{10} \left( \frac{P_{signal}}{P_{noise}} \\right)")
        st.latex(r"MSE = \frac{1}{N} \sum_{i=1}^{N} (x_i - \hat{x}_i)^2")
        
        st.markdown("**Applications:**")
        st.markdown("- Biometric authentication")
        st.markdown("- Criminal identification (AFIS)")
        st.markdown("- Access control systems")
        st.markdown("- Mobile device security")

if __name__ == "__main__":
    main()
'''

# Save Fingerprint page
with open('pages/2_🔍_Fingerprint_Scanning.py', 'w') as f:
    f.write(fingerprint_page_code)

print("✅ Created Fingerprint Scanning page")
print("File size:", len(fingerprint_page_code), "characters")