import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib import use
use('Agg')
from scipy import ndimage
import cv2

st.set_page_config(page_title="Fingerprint Scanning", page_icon="🔍", layout="wide")

# Enhanced CSS styling
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Control panel styling */
    .control-section {
        background: linear-gradient(135deg, #f6f8fb 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #2a5298;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-top: 3px solid #2a5298;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(42,82,152,0.2);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        background: linear-gradient(135deg, #1e3c72 0%, #7e22ce 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        color: #666;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500;
    }
    
    .metric-description {
        color: #888;
        font-size: 0.75rem;
        margin-top: 0.5rem;
    }
    
    /* Stage indicators */
    .stage-indicator {
        display: inline-block;
        padding: 6px 12px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 5px;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e3c72;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #2a5298;
    }
    
    /* Pipeline step cards */
    .pipeline-step {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        border-left: 3px solid #667eea;
    }
    
    /* Button enhancements */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102,126,234,0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102,126,234,0.4);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    /* Quality indicator */
    .quality-excellent { color: #4caf50; font-weight: bold; }
    .quality-good { color: #8bc34a; font-weight: bold; }
    .quality-fair { color: #ffc107; font-weight: bold; }
    .quality-poor { color: #ff5722; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

def generate_fingerprint_pattern(size=200, noise_level=0.1):
    """Generate a simulated fingerprint pattern with ridge-valley structure"""
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    # Create ridge patterns - ridges are high intensity (reflect more light), valleys are low
    ridges = np.sin(20 * r + 5 * np.sin(3 * theta)) * np.exp(-r**2)
    spiral = np.sin(10 * theta + 15 * r)
    fingerprint = 0.7 * ridges + 0.3 * spiral
    
    # Normalize to 0-1 range (0 = valley/dark, 1 = ridge/bright)
    fingerprint = (fingerprint - fingerprint.min()) / (fingerprint.max() - fingerprint.min())
    
    # Add slight skin texture noise
    noise = np.random.normal(0, noise_level * 0.3, fingerprint.shape)
    fingerprint += noise
    fingerprint = np.clip(fingerprint, 0, 1)
    
    return fingerprint

def optical_light_reflection(fingerprint, light_intensity=1.0, glass_reflection=0.95):
    """
    Simulate optical reflection from finger on glass plate.
    Ridges (in contact with glass) reflect more light than valleys (air gaps).
    """
    # Ridges reflect more light due to direct contact with glass
    # Valleys have air gaps and reflect less (total internal reflection in glass)
    reflected_light = fingerprint * light_intensity * glass_reflection
    
    # Add optical scattering and imperfections
    optical_blur = ndimage.gaussian_filter(reflected_light, sigma=0.5)
    
    return optical_blur

def photodiode_array_capture(optical_signal, sensor_quality=0.95, ambient_noise=0.05):
    """
    Simulate photodiode array converting optical pattern to electrical signal.
    Each photodiode produces a voltage proportional to light intensity.
    """
    # Photodiode response (converts light to electrical current/voltage)
    electrical_signal = optical_signal * sensor_quality
    
    # Add sensor noise (thermal noise, dark current)
    sensor_noise = np.random.normal(0, ambient_noise, electrical_signal.shape)
    electrical_signal = electrical_signal + sensor_noise
    
    # Photodiodes have limited response range
    electrical_signal = np.clip(electrical_signal, 0, 1)
    
    return electrical_signal

def analog_to_digital_converter(analog_signal, bits=8):
    """
    Digitize the analog electrical signal into discrete pixel intensity values.
    Each point becomes a digital pixel value (0-255 for 8-bit).
    """
    levels = 2**bits
    # Quantize to discrete levels
    digital_image = np.round(analog_signal * (levels - 1))
    # Normalize back to 0-1 for processing
    digital_image = digital_image / (levels - 1)
    return digital_image

def fourier_ridge_analysis(digital_image):
    """
    Apply Fourier transform to analyze ridge frequency and orientation.
    Returns frequency domain representation and ridge information.
    """
    # 2D FFT to analyze spatial frequencies
    fft_image = np.fft.fft2(digital_image)
    fft_shift = np.fft.fftshift(fft_image)
    magnitude_spectrum = np.abs(fft_shift)
    
    # Log scale for better visualization
    magnitude_spectrum_log = np.log1p(magnitude_spectrum)
    
    return fft_shift, magnitude_spectrum_log

def bandpass_filter_enhancement(digital_image, low_freq=0.1, high_freq=0.4):
    """
    Apply band-pass filtering to enhance repeating ridge patterns and suppress noise.
    Low frequencies = overall finger shape
    Medium frequencies = ridge patterns (what we want)
    High frequencies = noise
    """
    # Get FFT
    fft_image = np.fft.fft2(digital_image)
    fft_shift = np.fft.fftshift(fft_image)
    
    # Create band-pass filter
    rows, cols = digital_image.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create frequency grid
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    distance = np.sqrt(x*x + y*y)
    
    # Normalize distance to 0-1 range
    max_distance = np.sqrt(crow**2 + ccol**2)
    distance_norm = distance / max_distance
    
    # Band-pass mask (keeps frequencies in the ridge pattern range)
    mask = np.zeros_like(distance_norm)
    mask[(distance_norm >= low_freq) & (distance_norm <= high_freq)] = 1
    
    # Apply Gaussian smoothing to mask edges
    mask = ndimage.gaussian_filter(mask, sigma=2)
    
    # Apply filter in frequency domain
    fft_filtered = fft_shift * mask
    
    # Inverse FFT back to spatial domain
    fft_ishift = np.fft.ifftshift(fft_filtered)
    enhanced_image = np.fft.ifft2(fft_ishift)
    enhanced_image = np.abs(enhanced_image)
    
    # Normalize
    enhanced_image = (enhanced_image - enhanced_image.min()) / (enhanced_image.max() - enhanced_image.min())
    
    return enhanced_image, mask

def edge_minutiae_detection(enhanced_image, threshold=0.5):
    """
    Extract edges and minutiae points (ridge endings, bifurcations).
    Simplified version showing edge detection as first step.
    """
    # Sobel edge detection
    sobel_x = ndimage.sobel(enhanced_image, axis=1)
    sobel_y = ndimage.sobel(enhanced_image, axis=0)
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalize
    edge_magnitude = (edge_magnitude - edge_magnitude.min()) / (edge_magnitude.max() - edge_magnitude.min())
    
    # Binary threshold to get clear edges
    edges_binary = (edge_magnitude > threshold).astype(float)
    
    return edge_magnitude, edges_binary

def get_quality_rating(snr, correlation):
    """Determine quality rating based on metrics"""
    if snr > 15 and correlation > 0.9:
        return "Excellent", "quality-excellent", "🟢"
    elif snr > 10 and correlation > 0.8:
        return "Good", "quality-good", "🟡"
    elif snr > 5 and correlation > 0.7:
        return "Fair", "quality-fair", "🟠"
    else:
        return "Poor", "quality-poor", "🔴"

def main():
    # Header
    st.markdown('<h1 class="main-title">🔍 Advanced Fingerprint Scanner</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Biometric Signal Processing & Analysis Simulator</p>', unsafe_allow_html=True)
    
    # Info banner
    with st.expander("ℹ️ About Optical Fingerprint Scanning Technology", expanded=False):
        st.markdown("""
        ### How Optical Fingerprint Scanners Work
        
        This simulator demonstrates the **authentic optical scanning workflow** used in fingerprint readers:
        
        **1. 📸 Optical Sensing (Light Reflection)**
        - Finger is placed on a flat glass plate (prism)
        - LED light illuminates the finger from the side
        - **Ridges** (in contact with glass) reflect light back to sensor
        - **Valleys** (air gaps) cause total internal reflection - less light reaches sensor
        - Creates a high-contrast image: bright ridges, dark valleys
        
        **2. 🔆 Photodiode Array Capture**
        - Array of photodiodes (like a CCD/CMOS sensor) detects reflected light
        - Each photodiode converts light intensity → electrical voltage
        - Creates an analog electrical signal (grayscale image)
        - Essentially a "striped" pattern of ridges and valleys
        
        **3. 🔢 Analog-to-Digital Conversion (ADC)**
        - Continuous electrical signals → discrete digital values
        - Each point becomes a pixel with intensity value (e.g., 0-255 for 8-bit)
        - Now we have a digital image ready for processing
        
        **4. 🌊 Fourier Analysis & Enhancement**
        - **FFT** analyzes ridge frequency (ridges per mm) and orientation
        - **Band-pass filtering** enhances repeating ridge patterns
        - Suppresses low-freq noise (finger shape) and high-freq noise (sensor artifacts)
        
        **5. 🎯 Edge & Minutiae Extraction**
        - Edge detection highlights ridge boundaries
        - Minutiae points identified: ridge endings, bifurcations
        - These unique features create the fingerprint "template"
        
        **Real Systems:**
        - Optical scanners: FBI, police stations (high resolution)
        - Capacitive: Smartphones (measure electrical capacitance)
        - Ultrasonic: Advanced phones (uses sound waves)
        """)

    
    st.markdown("---")
    
    # Initialize session state
    if 'fingerprint_seed' not in st.session_state:
        st.session_state.fingerprint_seed = 42
    
    # Main layout
    col1, col2 = st.columns([1, 2.5], gap="large")
    
    with col1:
        st.markdown('<div class="section-header">⚙️ Scanner Configuration</div>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown("#### 💡 Optical System Settings")
            
            light_intensity = st.slider(
                "LED Light Intensity",
                0.5, 1.0, 0.9, 0.05,
                help="Brightness of LED illuminating the finger"
            )
            
            glass_quality = st.slider(
                "Glass Plate Quality",
                0.8, 1.0, 0.95, 0.01,
                help="Light transmission quality of the glass prism (higher = clearer)"
            )
            
            ambient_noise = st.slider(
                "Environmental Noise",
                0.0, 0.2, 0.05, 0.01,
                help="Ambient light, thermal noise, electrical interference"
            )
            
            st.markdown("---")
            st.markdown("#### 🔢 Digitization Settings")
            
            adc_bits = st.selectbox(
                "ADC Bit Depth",
                [4, 6, 8, 10, 12],
                index=2,
                help="Bits for analog-to-digital conversion (8-bit = 256 levels)"
            )
            
            st.markdown("---")
            st.markdown("#### 🌊 Signal Processing")
            
            apply_fourier = st.checkbox("Apply Fourier Analysis", value=True,
                                       help="Analyze ridge frequencies in frequency domain")
            
            apply_bandpass = st.checkbox("Apply Band-pass Filter", value=True,
                                        help="Enhance ridge patterns, suppress noise")
            
            if apply_bandpass:
                low_freq = st.slider("Low Frequency Cutoff", 0.05, 0.3, 0.1, 0.05,
                                    help="Remove low-frequency components (finger shape)")
                high_freq = st.slider("High Frequency Cutoff", 0.2, 0.6, 0.4, 0.05,
                                     help="Remove high-frequency noise")
            else:
                low_freq, high_freq = 0.1, 0.4
            
            apply_edge_detection = st.checkbox("Extract Edges/Minutiae", value=False,
                                              help="Detect ridge edges and characteristic points")
            
            st.markdown("---")
            st.markdown("#### 🎨 Additional Enhancement")
            
            enhance_contrast = st.checkbox("🔆 Enhance Contrast", value=False)
            apply_morphology = st.checkbox("🔲 Morphological Filtering", value=False,
                                          help="Clean up binary edges")
            
            st.markdown("---")
            
            # Generate button
            if st.button("🔄 Generate New Fingerprint", use_container_width=True):
                st.session_state.fingerprint_seed = np.random.randint(0, 10000)
                st.rerun()
            
            # Quick presets
            st.markdown("#### 🎯 Quick Presets")
            preset_col1, preset_col2 = st.columns(2)
            with preset_col1:
                if st.button("🏆 FBI Quality", use_container_width=True):
                    st.session_state.preset = 'fbi'
                    st.rerun()
            with preset_col2:
                if st.button("📱 Phone Scanner", use_container_width=True):
                    st.session_state.preset = 'phone'
                    st.rerun()
    
    with col2:
        # Apply presets if selected
        if hasattr(st.session_state, 'preset'):
            if st.session_state.preset == 'fbi':
                light_intensity, glass_quality, ambient_noise, adc_bits = 0.95, 0.98, 0.02, 12
                apply_fourier, apply_bandpass, apply_edge_detection = True, True, True
                low_freq, high_freq = 0.08, 0.45
            elif st.session_state.preset == 'phone':
                light_intensity, glass_quality, ambient_noise, adc_bits = 0.85, 0.92, 0.08, 8
                apply_fourier, apply_bandpass, apply_edge_detection = True, True, False
                low_freq, high_freq = 0.1, 0.4
            delattr(st.session_state, 'preset')
        
        # Set seed
        np.random.seed(st.session_state.fingerprint_seed)
        
        # === OPTICAL FINGERPRINT SCANNING PIPELINE ===
        
        # Step 1: Generate finger ridge-valley pattern
        finger_pattern = generate_fingerprint_pattern(noise_level=0.02)
        
        # Step 2: Optical light reflection from glass plate
        optical_signal = optical_light_reflection(finger_pattern, light_intensity, glass_quality)
        
        # Step 3: Photodiode array captures reflected light → electrical signal
        electrical_signal = photodiode_array_capture(optical_signal, glass_quality, ambient_noise)
        
        # Step 4: Analog-to-Digital Conversion - discrete pixel values
        digital_image = analog_to_digital_converter(electrical_signal, adc_bits)
        
        # Step 5: Fourier analysis (optional)
        if apply_fourier:
            fft_result, magnitude_spectrum = fourier_ridge_analysis(digital_image)
        else:
            magnitude_spectrum = None
        
        # Step 6: Band-pass filtering to enhance ridges
        if apply_bandpass:
            enhanced_image, bandpass_mask = bandpass_filter_enhancement(digital_image, low_freq, high_freq)
        else:
            enhanced_image = digital_image
            bandpass_mask = None
        
        # Step 7: Edge/minutiae extraction (optional)
        if apply_edge_detection:
            edge_image, edges_binary = edge_minutiae_detection(enhanced_image, threshold=0.3)
        else:
            edge_image, edges_binary = None, None
        
        # Step 8: Additional post-processing
        final_output = enhanced_image.copy()
        if enhance_contrast:
            final_output = np.clip((final_output - 0.5) * 2 + 0.5, 0, 1)
        if apply_morphology and edges_binary is not None:
            # Apply morphological closing to clean up edges
            from scipy.ndimage import binary_closing
            edges_binary = binary_closing(edges_binary, structure=np.ones((3,3)))

        
        # Calculate metrics
        def calculate_snr(original, processed):
            signal_power = np.mean(original**2)
            noise_power = np.mean((original - processed)**2)
            return 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        
        def calculate_mse(original, processed):
            return np.mean((original - processed)**2)
        
        snr = calculate_snr(finger_pattern, final_output)
        mse = calculate_mse(finger_pattern, final_output)
        correlation = np.corrcoef(finger_pattern.flatten(), final_output.flatten())[0,1]
        quality, quality_class, quality_icon = get_quality_rating(snr, correlation)
        
        # Quality summary banner
        st.markdown(f"""
        <div class="success-box">
            <h3 style="margin:0; color: #2e7d32;">{quality_icon} Overall Quality: <span class="{quality_class}">{quality}</span></h3>
            <p style="margin: 0.5rem 0 0 0; color: #555;">Based on SNR and correlation analysis | ADC: {2**adc_bits} levels ({adc_bits}-bit)</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics dashboard
        st.markdown('<div class="section-header">📊 Quality Metrics</div>', unsafe_allow_html=True)
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">SNR</div>
                <div class="metric-value">{snr:.1f}</div>
                <div class="metric-description">dB (higher is better)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Correlation</div>
                <div class="metric-value">{correlation:.3f}</div>
                <div class="metric-description">Similarity to original</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">MSE</div>
                <div class="metric-value">{mse:.4f}</div>
                <div class="metric-description">Error magnitude</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col4:
            adc_levels = 2**adc_bits
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">ADC Levels</div>
                <div class="metric-value">{adc_levels}</div>
                <div class="metric-description">{adc_bits}-bit quantization</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Tabbed visualization
        tab1, tab2, tab3, tab4 = st.tabs(["📸 Optical Pipeline", "🌊 Fourier Analysis", "🎯 Enhancement & Edges", "📚 Theory"])
        
        with tab1:
            st.markdown("### Optical Fingerprint Scanning Pipeline")
            
            # Create visualization with correct workflow
            fig, axes = plt.subplots(2, 4, figsize=(18, 10))
            fig.patch.set_facecolor('#f8f9fa')
            fig.suptitle("Complete Optical Fingerprint Scanner Workflow", fontsize=18, fontweight='bold', y=0.98)
            
            stages = [
                (finger_pattern, "1. Finger on Glass\n(Ridge-Valley Pattern)", 'gray'),
                (optical_signal, "2. Light Reflection\n(Optical Signal)", 'viridis'),
                (electrical_signal, "3. Photodiode Array\n(Electrical Signal)", 'plasma'),
                (digital_image, f"4. ADC Output\n({adc_bits}-bit Digital)", 'gray'),
                (magnitude_spectrum if magnitude_spectrum is not None else digital_image, 
                 "5. Fourier Spectrum\n(Frequency Analysis)", 'hot'),
                (enhanced_image, "6. Band-pass Filtered\n(Enhanced Ridges)", 'gray'),
                (edge_image if edge_image is not None else enhanced_image,
                 "7. Edge Detection\n(Ridge Boundaries)", 'inferno'),
                (final_output, "8. Final Output\n(Processed Image)", 'gray')
            ]
            
            for i, (signal, title, cmap) in enumerate(stages):
                ax = axes[i//4, i%4]
                im = ax.imshow(signal, cmap=cmap, interpolation='nearest')
                ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
                ax.axis('off')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax, shrink=0.75)
                cbar.ax.tick_params(labelsize=7)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Pipeline description
            st.markdown("""
            <div class="info-box">
                <strong>💡 Authentic Optical Scanning Workflow:</strong><br>
                <strong>1-2:</strong> Finger ridges (touching glass) reflect light, valleys (air gaps) don't<br>
                <strong>3:</strong> Photodiode array converts light intensity → electrical voltage<br>
                <strong>4:</strong> ADC digitizes analog signal → pixel intensity values<br>
                <strong>5:</strong> FFT reveals ridge frequencies and orientations<br>
                <strong>6:</strong> Band-pass filter enhances periodic ridge patterns<br>
                <strong>7:</strong> Edge detection finds ridge boundaries and minutiae<br>
                <strong>8:</strong> Final processed image ready for feature extraction
            </div>
            """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("### Fourier Transform Analysis")
            
            if magnitude_spectrum is not None:
                # Display frequency domain
                fig_fft, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                fig_fft.patch.set_facecolor('#f8f9fa')
                
                # Spatial domain
                ax1.imshow(digital_image, cmap='gray')
                ax1.set_title('Spatial Domain (Digital Image)', fontsize=13, fontweight='bold')
                ax1.axis('off')
                
                # Frequency domain
                im2 = ax2.imshow(magnitude_spectrum, cmap='hot')
                ax2.set_title('Frequency Domain (FFT Magnitude)', fontsize=13, fontweight='bold')
                ax2.axis('off')
                plt.colorbar(im2, ax=ax2, label='Log Magnitude')
                
                # Add annotations
                center_y, center_x = magnitude_spectrum.shape[0]//2, magnitude_spectrum.shape[1]//2
                ax2.plot(center_x, center_y, 'g+', markersize=15, markeredgewidth=2)
                ax2.text(center_x + 10, center_y, 'DC (0 Hz)', color='lime', fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig_fft)
                
                st.markdown("""
                <div class="info-box">
                    <strong>🌊 Fourier Analysis Explained:</strong><br>
                    • <strong>Center (DC component):</strong> Average brightness<br>
                    • <strong>Bright patterns radiating out:</strong> Ridge periodicities<br>
                    • <strong>Distance from center:</strong> Spatial frequency (ridges per mm)<br>
                    • <strong>Angle:</strong> Ridge orientation/direction<br><br>
                    
                    Engineers use this to:<br>
                    1. Determine ridge frequency (typically 5-7 ridges/mm)<br>
                    2. Analyze ridge orientation fields<br>
                    3. Design optimal band-pass filters
                </div>
                """, unsafe_allow_html=True)
                
                # Show band-pass mask if applied
                if bandpass_mask is not None:
                    st.markdown("#### Band-pass Filter Mask")
                    fig_mask, ax = plt.subplots(1, 1, figsize=(8, 8))
                    im = ax.imshow(bandpass_mask, cmap='viridis')
                    ax.set_title(f'Band-pass Mask ({low_freq:.2f} - {high_freq:.2f} normalized freq)', 
                               fontsize=12, fontweight='bold')
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, label='Filter Response (0=block, 1=pass)')
                    st.pyplot(fig_mask)
                    
                    st.caption("This mask is multiplied with the FFT to keep only ridge-frequency components")
            else:
                st.info("Enable 'Apply Fourier Analysis' to see frequency domain representation.")
        
        with tab3:
            st.markdown("### Enhancement & Feature Extraction")
            
            # Compare before/after enhancement
            fig_enh, axes_enh = plt.subplots(1, 3, figsize=(16, 5))
            fig_enh.patch.set_facecolor('#f8f9fa')
            
            axes_enh[0].imshow(digital_image, cmap='gray')
            axes_enh[0].set_title('Digital Image (After ADC)', fontsize=12, fontweight='bold')
            axes_enh[0].axis('off')
            
            axes_enh[1].imshow(enhanced_image, cmap='gray')
            axes_enh[1].set_title('After Band-pass Filter', fontsize=12, fontweight='bold')
            axes_enh[1].axis('off')
            
            if edge_image is not None:
                axes_enh[2].imshow(edge_image, cmap='hot')
                axes_enh[2].set_title('Edge/Minutiae Detection', fontsize=12, fontweight='bold')
                axes_enh[2].axis('off')
            else:
                axes_enh[2].imshow(final_output, cmap='gray')
                axes_enh[2].set_title('Final Output', fontsize=12, fontweight='bold')
                axes_enh[2].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig_enh)
            
            # Show binary edges if available
            if edges_binary is not None:
                st.markdown("#### Binary Edge Map (Minutiae Points)")
                fig_binary, ax = plt.subplots(1, 1, figsize=(8, 8))
                ax.imshow(edges_binary, cmap='gray')
                ax.set_title('Binary Edges - Ridge Endings & Bifurcations', fontsize=13, fontweight='bold')
                ax.axis('off')
                st.pyplot(fig_binary)
                
                st.markdown("""
                <div class="success-box">
                    <strong>🎯 Minutiae Extraction:</strong><br>
                    From these edges, the system identifies:<br>
                    • <strong>Ridge endings:</strong> Where a ridge terminates<br>
                    • <strong>Bifurcations:</strong> Where a ridge splits into two<br>
                    • <strong>Short ridges:</strong> Isolated ridge segments<br><br>
                    
                    These minutiae points form the unique "fingerprint template" stored for matching.
                    Typical fingerprint has 25-80 minutiae points.
                </div>
                """, unsafe_allow_html=True)
        
        with tab4:
            st.markdown("### 📚 Optical Scanning Technology")
            
            theory_col1, theory_col2 = st.columns(2)
            
            with theory_col1:
                st.markdown("""
                #### Complete Optical Workflow
                
                <div class="pipeline-step">
                    <strong>1. Optical Reflection</strong><br>
                    Finger placed on glass plate (prism). LED light illuminates from side.<br>
                    • Ridges (contact) → light reflects to sensor<br>
                    • Valleys (air gap) → total internal reflection in glass<br>
                    • Result: Bright ridges, dark valleys
                </div>
                
                <div class="pipeline-step">
                    <strong>2. Photodiode Array</strong><br>
                    CCD/CMOS sensor with array of photodiodes.<br>
                    • Each photodiode: light → electrical current<br>
                    • Creates analog voltage signal<br>
                    • Essentially a "striped" grayscale image
                </div>
                
                <div class="pipeline-step">
                    <strong>3. Analog-to-Digital Conversion</strong><br>
                    ADC samples and quantizes the electrical signal.<br>
                    • Continuous voltage → discrete levels<br>
                    • 8-bit = 256 levels (0-255)<br>
                    • Now a true digital image
                </div>
                
                <div class="pipeline-step">
                    <strong>4. Fourier Analysis</strong><br>
                    FFT transforms spatial image to frequency domain.<br>
                    • Reveals ridge periodicity (cycles/mm)<br>
                    • Shows orientation fields<br>
                    • Guides filter design
                </div>
                
                <div class="pipeline-step">
                    <strong>5. Band-pass Filtering</strong><br>
                    Keeps only ridge-pattern frequencies.<br>
                    • Low freq: finger shape (suppress)<br>
                    • Mid freq: ridges (enhance)<br>
                    • High freq: noise (suppress)
                </div>
                
                <div class="pipeline-step">
                    <strong>6. Minutiae Extraction</strong><br>
                    Edge detection → characteristic points.<br>
                    • Ridge endings<br>
                    • Bifurcations<br>
                    • Creates matching template
                </div>
                """, unsafe_allow_html=True)
            
            with theory_col2:
                st.markdown("#### Mathematical Foundations")
                
                st.markdown("**2D Fourier Transform:**")
                st.latex(r"F(u,v) = \sum_{x=0}^{M-1}\sum_{y=0}^{N-1} f(x,y) e^{-j2\pi(ux/M + vy/N)}")
                st.caption("Decomposes image into spatial frequency components")
                
                st.markdown("**Band-pass Filter (Frequency Domain):**")
                st.latex(r"H(u,v) = \begin{cases} 1 & \text{if } f_L \leq \sqrt{u^2+v^2} \leq f_H \\ 0 & \text{otherwise} \end{cases}")
                st.caption("Keeps frequencies between fL (low) and fH (high)")
                
                st.markdown("**Sobel Edge Detection:**")
                st.latex(r"G = \sqrt{G_x^2 + G_y^2}")
                st.caption("Gradient magnitude shows edge strength")
                
                st.markdown("**ADC Quantization:**")
                st.latex(r"Q(x) = \frac{\lfloor x \cdot (2^b - 1) \rfloor}{2^b - 1}")
                st.caption("b-bit quantization (e.g., 8-bit = 256 levels)")
                
                st.markdown("---")
                
                st.markdown("#### Scanner Technologies")
                st.markdown("""
                **Optical (This Simulator):**
                - LED + prism + CCD/CMOS
                - High resolution (500-1000 DPI)
                - Used: FBI, police, border control
                - Bulky but very accurate
                
                **Capacitive:**
                - Measures capacitance ridge-valley
                - Compact, lower cost
                - Used: Smartphones, laptops
                - Resolution: 300-500 DPI
                
                **Ultrasonic:**
                - Sound waves penetrate skin
                - Works through dirt, moisture
                - Used: High-end phones 
                - Can detect liveness
                
                **Thermal:**
                - Temperature difference
                - Less common
                - Vulnerable to ambient temp
                """)
                
                st.markdown("#### Ridge Characteristics")
                st.markdown("""
                - **Ridge frequency**: 5-7 ridges/mm typically
                - **Ridge width**: ~0.3-0.5 mm
                - **Valley width**: Similar to ridge width
                - **Minutiae density**: 25-80 points per finger
                - **Uniqueness**: ~1 in 64 billion probability
                """)

if __name__ == "__main__":
    main()