import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib import use
use('Agg')
from scipy import ndimage
from PIL import Image
import io

st.set_page_config(page_title="Image Playhouse", page_icon="🔍", layout="wide")

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
    
    /* Quality indicator */
    .quality-excellent { color: #4caf50; font-weight: bold; }
    .quality-good { color: #8bc34a; font-weight: bold; }
    .quality-fair { color: #ffc107; font-weight: bold; }
    .quality-poor { color: #ff5722; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

def load_image(uploaded_file):
    """Load and convert image to grayscale numpy array"""
    try:
        image = Image.open(uploaded_file)
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        # Normalize to 0-1 range
        image_array = np.array(image) / 255.0
        return image_array
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

def preprocess_print(image, contrast_enhancement=1.2):
    """Preprocess image: contrast enhancement and noise reduction"""
    # Contrast stretching
    min_val = image.min()
    max_val = image.max()
    if max_val > min_val:
        image = (image - min_val) / (max_val - min_val)
    
    # Gaussian blur for noise reduction
    processed = ndimage.gaussian_filter(image, sigma=0.8)
    
    # Contrast enhancement
    processed = np.power(processed, 1/contrast_enhancement)
    
    return np.clip(processed, 0, 1)

def fourier_ridge_analysis(digital_image):
    """Apply Fourier transform to analyze ridge frequency and orientation"""
    fft_image = np.fft.fft2(digital_image)
    fft_shift = np.fft.fftshift(fft_image)
    magnitude_spectrum = np.abs(fft_shift)
    magnitude_spectrum_log = np.log1p(magnitude_spectrum)
    
    return magnitude_spectrum_log, fft_shift

def bandpass_filter_enhancement(digital_image, low_freq=0.05, high_freq=0.4):
    """Apply bandpass filter and return both mask & filtered FFT magnitude."""
    fft_image = np.fft.fft2(digital_image)
    fft_shift = np.fft.fftshift(fft_image)

    # Create frequency grid and corresponding mask
    h, w = digital_image.shape
    x = np.linspace(-0.5, 0.5, w)
    y = np.linspace(-0.5, 0.5, h)
    X, Y = np.meshgrid(x, y)
    frequency = np.sqrt(X**2 + Y**2)
    bandpass = np.logical_and(frequency >= low_freq, frequency <= high_freq).astype(float)

    # Apply filter in frequency domain
    filtered_fft = fft_shift * bandpass
    filtered_image = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_fft)))

    # For frequency visualization, use log scale like FFT
    filtered_fft_magnitude = np.log1p(np.abs(filtered_fft))

    return filtered_image / filtered_image.max(), bandpass, filtered_fft_magnitude

def edge_minutiae_detection(enhanced_image, threshold=0.5):
    """Detect edges and potential minutiae points"""
    # Edge detection using Sobel
    sx = ndimage.sobel(enhanced_image, axis=0)
    sy = ndimage.sobel(enhanced_image, axis=1)
    edges = np.sqrt(sx**2 + sy**2)
    edges = edges / edges.max()
    
    # Binary threshold
    binary = edges > threshold
    
    return edges, binary

def get_quality_rating(image):
    """Calculate quality rating of image"""
    # Calculate contrast
    contrast = image.std()
    
    # Calculate sharpness (using Laplacian)
    laplacian = ndimage.laplace(image)
    sharpness = laplacian.std()
    
    # Overall quality score (0-100)
    quality_score = min(100, (contrast * 50 + sharpness * 50))
    
    if quality_score >= 80:
        return "Excellent", "✅", quality_score
    elif quality_score >= 60:
        return "Good", "👍", quality_score
    elif quality_score >= 40:
        return "Fair", "⚠️", quality_score
    else:
        return "Poor", "❌", quality_score

def main():
    st.markdown('<h1 class="main-title">🔍 Image Analysis Playhouse</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload and analyze images with digital signal processing</p>', 
                unsafe_allow_html=True)
    
    # File upload section
    with st.container():
        st.markdown('<div class="section-header">📁 Upload Image</div>', 
                    unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose an image file", 
                                         type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'])
        
        if uploaded_file is not None:
            # Load image
            image = load_image(uploaded_file)
            
            if image is not None:
                st.success("✅ Image loaded successfully!")
                
                # Sidebar controls
                with st.sidebar:
                    st.markdown("### ⚙️ Processing Parameters")
                    contrast_enhancement = st.slider("Contrast Enhancement", 1.0, 3.0, 1.2, 0.1)
                    low_freq = st.slider("Low Frequency Cutoff", 0.01, 0.2, 0.05, 0.01)
                    high_freq = st.slider("High Frequency Cutoff", 0.2, 0.6, 0.4, 0.05)
                    edge_threshold = st.slider("Edge Detection Threshold", 0.1, 0.9, 0.5, 0.05)
                
                # Create tabs for different processing stages
                tab1, tab2, tab3, tab4, tab5 = st.tabs(
                    ["📸 Original", "🎨 Preprocessed", "📊 Frequency Analysis", 
                     "✨ Enhanced", "🔍 Edge Detection"]
                )
                
                # Process image
# Image processing
                preprocessed = preprocess_print(image, contrast_enhancement)
                magnitude_spectrum, fft_shift = fourier_ridge_analysis(preprocessed)
                enhanced, mask, filtered_fft_magnitude = bandpass_filter_enhancement(preprocessed, low_freq, high_freq)
                edges, binary = edge_minutiae_detection(enhanced, edge_threshold)

                
                # Tab 1: Original Image
                with tab1:
                    st.markdown("### Original Image")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_orig = plt.figure(figsize=(6, 6))
                        plt.imshow(image, cmap='gray')
                        plt.title("Original Image")
                        plt.axis('off')
                        st.pyplot(fig_orig)
                    
                    with col2:
                        quality_level, quality_emoji, quality_score = get_quality_rating(image)
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Image Quality</div>
                            <div class="metric-value">{quality_emoji}</div>
                            <div class="metric-description">{quality_level}</div>
                            <div class="metric-description">Score: {quality_score:.1f}/100</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.info(f"**Dimensions:** {image.shape[0]} × {image.shape[1]} pixels")
                        st.info(f"**Contrast (Std Dev):** {image.std():.4f}")
                
                # Tab 2: Preprocessed Image
                with tab2:
                    st.markdown("### Preprocessed Image (Denoised & Enhanced)")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_pre = plt.figure(figsize=(6, 6))
                        plt.imshow(preprocessed, cmap='gray')
                        plt.title("Preprocessed Image")
                        plt.axis('off')
                        st.pyplot(fig_pre)
                    
                    with col2:
                        st.markdown("""
                        <div class="info-box">
                            <strong>Processing Steps:</strong>
                            <ul>
                                <li>Contrast stretching</li>
                                <li>Gaussian blur filtering</li>
                                <li>Noise reduction</li>
                                <li>Contrast enhancement</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Tab 3: Frequency Analysis
                with tab3:
                    st.markdown("### Fourier Transform Analysis")
                    # Three columns for: Magnitude Spectrum, Bandpass Mask, Filtered Frequency Spectrum
                    colA, colB, colC = st.columns(3)

                    # Original Frequency Magnitude
                    with colA:
                        fig_fft = plt.figure(figsize=(5, 5))
                        plt.imshow(magnitude_spectrum, cmap='hot')
                        plt.title("Magnitude Spectrum (Log Scale)")
                        plt.colorbar(label='Magnitude')
                        plt.axis('off')
                        st.pyplot(fig_fft)

                    # Bandpass Mask
                    with colB:
                        fig_mask = plt.figure(figsize=(5, 5))
                        plt.imshow(mask, cmap='viridis')
                        plt.title('Bandpass Frequency Mask')
                        plt.axis('off')
                        plt.colorbar(label='Mask Value')
                        st.pyplot(fig_mask)

                    # Filtered Frequency Spectrum (after mask applied)
                    with colC:
                        fig_filt = plt.figure(figsize=(5, 5))
                        plt.imshow(filtered_fft_magnitude, cmap='hot')
                        plt.title("Filtered Frequency Domain")
                        plt.colorbar(label='Filtered Magnitude')
                        plt.axis('off')
                        st.pyplot(fig_filt)

                    # Info box as before
                    st.markdown("""
                    <div class="info-box">
                        <strong>Ridge Frequency Analysis:</strong>
                        <ul>
                            <li>Left: Full frequency spectrum of the image</li>
                            <li>Center: Bandpass mask highlighting ridge-relevant frequencies</li>
                            <li>Right: Frequency response after applying the mask</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

                
                # Tab 4: Enhanced Image
                with tab4:
                    st.markdown("### Bandpass Filtered Enhancement")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_enh = plt.figure(figsize=(6, 6))
                        plt.imshow(enhanced, cmap='gray')
                        plt.title("Bandpass Filtered Image")
                        plt.axis('off')
                        st.pyplot(fig_enh)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="success-box">
                            <strong>Filter Parameters:</strong>
                            <ul>
                                <li>Low Frequency: {low_freq:.3f}</li>
                                <li>High Frequency: {high_freq:.3f}</li>
                                <li>Ridge enhancement active</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Tab 5: Edge Detection
                with tab5:
                    st.markdown("### Edge and Minutiae Detection")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_edge = plt.figure(figsize=(6, 6))
                        plt.imshow(edges, cmap='hot')
                        plt.title("Edge Magnitude")
                        plt.colorbar(label='Edge Strength')
                        st.pyplot(fig_edge)
                    
                    with col2:
                        fig_bin = plt.figure(figsize=(6, 6))
                        plt.imshow(binary, cmap='gray')
                        plt.title(f"Binary Edges (Threshold: {edge_threshold:.2f})")
                        plt.axis('off')
                        st.pyplot(fig_bin)
                        
                        # Statistics
                        edge_pixels = np.sum(binary)
                        total_pixels = binary.shape[0] * binary.shape[1]
                        edge_percentage = (edge_pixels / total_pixels) * 100
                        st.info(f"**Edge Pixels:** {edge_pixels} ({edge_percentage:.1f}%)")
        
        else:
            st.warning("👆 Please upload an image to begin analysis")
            st.markdown("""
            <div class="info-box">
                <strong>Supported Formats:</strong> PNG, JPG, JPEG, BMP, TIFF
                <br><br>
                <strong>Processing Pipeline:</strong>
                <ol>
                    <li>Image preprocessing (denoising, contrast enhancement)</li>
                    <li>Fourier analysis for frequency detection</li>
                    <li>Bandpass filtering for enhancement</li>
                    <li>Edge detection and feature analysis</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()