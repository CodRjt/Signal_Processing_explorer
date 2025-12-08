# 📊 Interactive Signal Processing Explorer

**Master Digital Signal Processing through Play and Visualization**

A comprehensive Streamlit web application designed to help students, engineers, and enthusiasts explore complex DSP concepts—from the Nyquist Theorem to Image Fourier Analysis—through interactive, real-time demonstrations and gamified challenges.

---

## 🚀 Features & Modules

The application is organized into **6 interactive modules**, each targeting specific DSP concepts with hands-on exploration:

### 1. 📞 DTMF Checker
**Concept:** Dual-Tone Multi-Frequency signaling (Phone dial tones)

- **Interactive Virtual Keypad:** Control buttons 0–9, *, and #
- **Real-Time Audio Generation & Playback:** Instantly synthesize DTMF tones
- **FFT Analysis:** Visualize the specific low and high-frequency pairs for each digit
- **Automated Tone Detection Algorithms:** Learn how phones decode your key presses

### 2. 🔍 Audio DSP Workbench
**Concept:** Waveform synthesis, noise injection, and filtering

- **Rack-Mount Interface:** Generate Sine, Square, Sawtooth, or Triangle waves with adjustable frequency and amplitude
- **Noise Channel:** Inject White Noise, 50 Hz Hum, or High-Frequency Hiss to test signal robustness
- **Filter Unit:** Interactive Low-Pass and High-Pass filters with real-time parameter adjustment
- **Tape Loop Library:** Record and save processed sounds to compare A/B snapshots side-by-side
- **Real-Time Visualization:** Oscilloscope view (time domain) and Spectrum Analyzer (frequency domain)

### 3. 📊 Sampling Rate Visualization
**Concept:** The Nyquist–Shannon Sampling Theorem & Aliasing

- **Adjustable Sampling Rates:** Vary sampling frequency relative to signal frequency to explore Nyquist limits
- **Reconstruction Methods:** Compare Linear, Cubic, and Ideal Sinc interpolation techniques
- **Error Analysis:** View Mean Squared Error (MSE) and reconstruction artifacts in real time
- **Visual Safety Indicators:** "Safe Oversampling" vs. "Aliasing Detected" alerts

### 4. 👁️ Vision & Perception (The Human Sampler)
**Concept:** Biological sampling, Temporal vs. Spatial Aliasing, and Chroma Subsampling

- **Wagon Wheel Effect:** Interactive demo of temporal aliasing and stroboscopic effects
- **Chroma Subsampling Lab:** Independently degrade Luma (Brightness) and Chroma (Color) to understand JPEG/Video compression
- **Custom Upload:** Test bandwidth compression on your own photos or use the built-in Astronaut test image
- **Theory Deep Dive:** Rods vs. Cones, the Retinal Mosaic, and Moiré pattern formation

### 5. 🕹️ The DSP Arcade (Real-Life Applications)
**Concept:** Gamified learning of practical DSP tasks

**Levels:**
- **The Bitcrusher:** Learn quantization (Bit Depth) and downsampling by making retro Lo-Fi audio
- **Pixel Art Factory:** Understand spatial downsampling and color bit-depth reduction
- **Ghost Signal Hunter:** Use Bandpass filters to isolate a hidden sine wave inside heavy noise

### 6. 🛝 Image Analysis Playhouse
**Concept:** Advanced 2D Signal Processing

- **Fourier Analysis:** View the 2D Magnitude Spectrum and phase of images
- **Frequency Filtering:** Apply Bandpass filters to isolate specific textures and ridge patterns
- **Edge Detection:** Apply Sobel operators to detect and enhance boundaries
- **Quality Metrics:** Automated scoring of image contrast, sharpness, and frequency content

---

## 🛠️ Installation

### Prerequisites

- **Python:** 3.8 or higher
- **pip:** Package manager for Python

### Step 1: Clone the Repository

```bash
git clone https://github.com/CodRjt/signal-processing-explorer.git
cd signal-processing-explorer
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt should include:**
```
streamlit>=1.28.0
numpy>=1.21.0
matplotlib>=3.5.0
plotly>=5.0.0
scipy>=1.7.0
Pillow>=8.0.0
requests>=2.26.0
scikit-image>=0.19.0
```

### Step 3: Run the Application

```bash
streamlit run app.py
```

The app will open in your default browser at **http://localhost:8501**.

---

## 📂 Project Structure

```
signal-processing-explorer/
│
├── app.py                                      # Main entry point (Sidebar & Navigation)
├── requirements.txt                            # Python dependencies
├── README.md                                   # This file
│
└── pages/                                      # Multi-page modules
    ├── 1_📞_DTMF_Checker.py                    # DTMF tone generation & analysis
    ├── 2_🔍_Audio_Tuner.py                     # Audio Workbench (waveforms & filters)
    ├── 3_📊_Sampling_Rate_Visualization.py     # Nyquist theorem & aliasing
    ├── 4_👁️_Human_Eye_Sampling.py              # Vision, aliasing & compression
    ├── 5_🌍_Real_Life_DSP.py                   # The DSP Arcade (games)
    └── 6_🛝_Image_Playhouse.py                 # Image FFT & filtering
```

---

## 💡 User Tips

**Precision Control**  
Click on any slider handle and use your **Left/Right Arrow Keys** on your keyboard to fine-tune values by 1 step for greater control.

**Audio Feedback**  
Ensure your system volume is on for **Modules 1, 2, and 5** (DTMF Checker, Audio Workbench, DSP Arcade) to hear generated tones and effects.

**Navigation**  
Use the sidebar to navigate between modules and access quick keyboard shortcuts. Each module is self-contained with its own interactive controls.

**Performance**  
For the best experience with real-time plots and audio, use a modern web browser (Chrome, Firefox, Safari) and ensure adequate system resources (RAM, GPU acceleration optional).

---

## 📚 References & Theory

This application implements foundational and advanced concepts from:

- **Oppenheim & Schafer.** *Discrete-Time Signal Processing.* 3rd ed. Prentice Hall, 2010.
- **Nyquist–Shannon Sampling Theorem:** The mathematical foundation for converting continuous signals to discrete samples without information loss.
- **Fast Fourier Transform (FFT) Algorithms:** Efficient computation of frequency-domain representations.
- **Chroma Subsampling (YUV Color Space):** Standards for image and video compression based on human perception.
- **Edge Detection & Image Gradients:** Sobel and Scharr operators for boundary enhancement.
- **Human Visual System:** Rods, cones, temporal resolution, and spatial aliasing artifacts.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. **Fork** the repository
2. **Create a feature branch:** `git checkout -b feature/your-feature-name`
3. **Commit your changes:** `git commit -m "Add your feature description"`
4. **Push to the branch:** `git push origin feature/your-feature-name`
5. **Open a Pull Request**

**For major changes**, please open an issue first to discuss what you would like to change and ensure alignment with the project's educational goals.

---

## 📄 License

This project is licensed under the **MIT License**—see the [LICENSE](https://choosealicense.com/licenses/mit/) file for details.

You are free to use, modify, and distribute this software in personal and commercial projects, provided you include the original copyright notice and license text.

---

## 🎓 Educational Use

This application is ideal for:

- **University Courses:** Digital Signal Processing, Image Processing, Audio Engineering
- **Self-Paced Learning:** Students exploring DSP concepts independently
- **Professional Training:** Engineers refreshing skills or exploring new tools
- **Research & Development:** Rapid prototyping of signal and image processing pipelines

---

## 🐛 Troubleshooting

**"ModuleNotFoundError" for streamlit or dependencies:**  
Ensure you have activated the correct Python environment and run `pip install -r requirements.txt`.

**Audio not playing in Modules 1, 2, or 5:**  
Check your system volume, browser permissions for audio, and that your audio device is properly configured.

**Slow performance or lag:**  
Close other applications, reduce the resolution of plots if available, or run on a machine with more available RAM.

**Plots not rendering:**  
Ensure your browser supports WebGL (for Plotly 3D plots) and that JavaScript is enabled.

---

## 📧 Contact & Support

For questions, suggestions, or issues, please:

- Open an issue on the [GitHub repository](https://github.com/CodRjt/signal-processing-explorer/issues)
- Review existing issues and discussions for quick answers

---

## 🙏 Acknowledgments

Special thanks to:

- **Streamlit** for the intuitive web app framework
- **NumPy, SciPy, Matplotlib, and Plotly** for numerical and visualization libraries
- **scikit-image** for advanced image processing tools
- The **DSP and computer vision communities** for foundational algorithms and theory

---

**Version:** 1.0.0  
**Last Updated:** December 2025  
**Maintainer:** CodRjt