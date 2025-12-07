# Interactive Signal Processing Explorer

A comprehensive Streamlit web application for exploring digital signal processing concepts through interactive visualizations and real-world demonstrations.

## 🎯 Features

### Modules Included:

1. **📞 DTMF Checker**
   - Interactive telephone keypad simulation
   - Dual-tone frequency generation and analysis
   - Time and frequency domain visualization
   - Real-time tone detection

2. **🔍 Audio Tuner Simulation**
   - Biometric signal acquisition visualization
   - Analog-to-digital conversion demonstration
   - Quantization effects analysis
   - Signal quality metrics

3. **📊 Sampling Rate Visualization**
   - Nyquist theorem demonstration
   - Aliasing effects in under-sampling
   - Multiple reconstruction methods
   - Frequency domain analysis

4. **👁️ Human Eye Sampling Analogy**
   - Visual perception as discrete sampling
   - Frame rate effects on motion perception
   - Motion blur and persistence of vision
   - Temporal sampling comparison

5. **🌍 Real-life DSP Applications**
   - Audio quantization noise visualization
   - Image downsampling/upsampling effects
   - Signal reconstruction techniques
   - Anti-aliasing demonstrations

  6. **🛝 Image Playhouse**
   - Upload Image
   - Fourier spectrum visualization
   - Interactive frequency filtering (low-pass/high-pass)
   - Edge detection and feature extraction  
## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download this repository

2. Install required packages:
```bash
pip install -r requirements.txt
```

## 📱 Running the Application

### Local Development

Run the Streamlit app:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Deployment on Streamlit Cloud

1. Push your code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select `app.py` as the main file
5. Deploy!

## 📁 Project Structure

```
signal-processing-app/
│
├── app.py                          # Main application entry point
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
└── pages/                          # Multi-page modules
    ├── 1_📞_DTMF_Checker.py
    ├── 2_🔍_Fingerprint_Scanning.py
    ├── 3_📊_Sampling_Rate_Visualization.py
    ├── 4_👁️_Human_Eye_Sampling.py
    └── 5_🌍_Real_Life_DSP.py
```

## 🎓 Educational Value

This application is designed for:
- **Students** learning digital signal processing
- **Educators** teaching DSP concepts
- **Engineers** reviewing fundamental concepts
- **Enthusiasts** exploring signal processing applications

## 🔧 Technologies Used

- **Streamlit**: Web application framework
- **NumPy**: Numerical computations
- **Matplotlib**: Static plotting
- **Plotly**: Interactive visualizations
- **SciPy**: Signal processing functions
- **OpenCV**: Image processing

## 📊 Key Concepts Covered

- Discrete Fourier Transform (DFT/FFT)
- Nyquist-Shannon Sampling Theorem
- Quantization and bit depth
- Aliasing and anti-aliasing
- Signal reconstruction
- Frequency domain analysis
- Time-frequency representations

## 🤝 Contributing

This is an educational project. Suggestions and improvements are welcome!

## 📄 License

This project is open-source and available for educational purposes.

## 👨‍💻 Credits

Developed using Streamlit and Python by AI-generated design.

## 📞 Support

For issues or questions:
- Open an issue in the repository
- Check Streamlit documentation: https://docs.streamlit.io

## 🎨 Customization

You can customize the application by:
- Modifying signal generation parameters
- Adding new DSP modules
- Adjusting visualization styles
- Implementing additional signal processing techniques

## 📚 References

Key DSP concepts are based on standard signal processing theory:
- Oppenheim & Schafer: "Discrete-Time Signal Processing"
- Proakis & Manolakis: "Digital Signal Processing"
- Lyons: "Understanding Digital Signal Processing"

---

**Enjoy exploring signal processing! 🎉**
