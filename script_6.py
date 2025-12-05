# Create Human Eye Sampling page
eye_sampling_page_code = '''
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from matplotlib import use
use('Agg')
import time

st.set_page_config(page_title="Human Eye Sampling", page_icon="👁️", layout="wide")

def generate_moving_object_data(t, motion_type='linear', speed=1.0):
    """Generate position data for moving object"""
    if motion_type == 'linear':
        x = speed * t
        y = np.zeros_like(t)
    elif motion_type == 'circular':
        x = speed * np.cos(2 * np.pi * 0.5 * t)
        y = speed * np.sin(2 * np.pi * 0.5 * t)
    elif motion_type == 'oscillating':
        x = speed * np.sin(2 * np.pi * t)
        y = np.zeros_like(t)
    elif motion_type == 'complex':
        x = speed * (t + 0.5 * np.sin(4 * np.pi * t))
        y = 0.3 * speed * np.cos(6 * np.pi * t)
    
    return x, y

def sample_motion(t, x, y, frame_rate):
    """Sample the continuous motion at given frame rate"""
    dt = 1 / frame_rate
    t_frames = np.arange(0, t[-1], dt)
    
    # Interpolate positions at frame times
    x_frames = np.interp(t_frames, t, x)
    y_frames = np.interp(t_frames, t, y)
    
    return t_frames, x_frames, y_frames

def simulate_motion_blur(positions, blur_factor=0.1):
    """Simulate motion blur effect at low frame rates"""
    if len(positions) < 2:
        return positions
    
    blurred = positions.copy()
    for i in range(1, len(positions)):
        velocity = positions[i] - positions[i-1]
        blur_offset = blur_factor * velocity
        blurred[i] = positions[i] + blur_offset
    
    return blurred

def calculate_perceived_smoothness(frame_rate):
    """Calculate smoothness perception based on frame rate"""
    if frame_rate >= 60:
        return "Very Smooth"
    elif frame_rate >= 30:
        return "Smooth"
    elif frame_rate >= 24:
        return "Acceptable"
    elif frame_rate >= 15:
        return "Noticeable Stuttering"
    elif frame_rate >= 10:
        return "Choppy"
    else:
        return "Very Choppy"

def main():
    st.title("👁️ Human Eye Sampling Analogy")
    
    st.markdown("""
    The human visual system can be understood as a **temporal sampling system**. Just like digital
    signal sampling, our eyes capture discrete "frames" of the world around us. This module explores
    the analogy between visual perception and digital signal processing concepts.
    """)
    
    # Controls
    control_col, viz_col = st.columns([1, 2])
    
    with control_col:
        st.markdown("### 👁️ Eye & Display Parameters")
        
        # Frame rate simulation
        frame_rate = st.slider("Visual Frame Rate (fps)", 5, 120, 30, 5)
        
        # Motion parameters
        st.markdown("### 🏃 Motion Parameters")
        motion_type = st.selectbox(
            "Motion Type",
            ['linear', 'circular', 'oscillating', 'complex']
        )
        
        motion_speed = st.slider("Motion Speed", 0.5, 3.0, 1.0, 0.1)
        
        # Visual effects
        st.markdown("### 🎬 Visual Effects")
        show_motion_blur = st.checkbox("Show Motion Blur", value=False)
        blur_intensity = st.slider("Blur Intensity", 0.0, 0.5, 0.1, 0.05)
        
        show_persistence = st.checkbox("Show Persistence of Vision", value=False)
        
        # Eye physiology info
        st.markdown("### 📊 Visual System Info")
        smoothness = calculate_perceived_smoothness(frame_rate)
        st.info(f"**Perceived Motion:** {smoothness}")
        
        # Critical flicker fusion frequency
        st.markdown(f"**Critical Flicker Fusion:** ~50 Hz")
        st.markdown(f"**Current Rate:** {frame_rate} Hz")
        
        if frame_rate < 24:
            st.warning("Below cinematic threshold (24 fps)")
        elif frame_rate < 30:
            st.info("Cinematic range (24-30 fps)")
        elif frame_rate < 60:
            st.success("Television standard (30-60 fps)")
        else:
            st.success("High refresh rate (60+ fps)")
    
    with viz_col:
        st.markdown("### 🎥 Motion Visualization")
        
        # Generate motion data
        duration = 4.0  # seconds
        t_continuous = np.linspace(0, duration, 1000)
        x_continuous, y_continuous = generate_moving_object_data(
            t_continuous, motion_type, motion_speed
        )
        
        # Sample at specified frame rate
        t_frames, x_frames, y_frames = sample_motion(
            t_continuous, x_continuous, y_continuous, frame_rate
        )
        
        # Apply motion blur if enabled
        if show_motion_blur:
            x_blur = simulate_motion_blur(x_frames, blur_intensity)
            y_blur = simulate_motion_blur(y_frames, blur_intensity)
        else:
            x_blur = x_frames
            y_blur = y_frames
        
        # Create trajectory visualization
        fig_traj = go.Figure()
        
        # Continuous trajectory (what actually happens)
        fig_traj.add_trace(go.Scatter(
            x=x_continuous,
            y=y_continuous,
            mode='lines',
            name='Actual Motion',
            line=dict(color='lightblue', width=3, dash='dash'),
            opacity=0.7
        ))
        
        # Sampled positions (what eye "sees")
        fig_traj.add_trace(go.Scatter(
            x=x_blur,
            y=y_blur,
            mode='markers+lines',
            name=f'Perceived Motion ({frame_rate} fps)',
            line=dict(color='red', width=2),
            marker=dict(size=8, color='red', symbol='circle')
        ))
        
        # Add frame indicators
        if show_persistence:
            # Show trailing effect (persistence of vision)
            for i in range(1, min(len(x_frames), 10)):  # Show last 10 frames
                alpha = 1.0 - (i * 0.1)
                if alpha > 0:
                    idx = len(x_frames) - i - 1
                    if idx >= 0:
                        fig_traj.add_trace(go.Scatter(
                            x=[x_frames[idx]],
                            y=[y_frames[idx]],
                            mode='markers',
                            marker=dict(size=12, color='yellow', 
                                      symbol='circle-open', 
                                      opacity=alpha),
                            name=f'Frame -{i}',
                            showlegend=False
                        ))
        
        fig_traj.update_layout(
            title=f"Motion Perception at {frame_rate} fps",
            xaxis_title="X Position",
            yaxis_title="Y Position",
            height=400,
            showlegend=True,
            hovermode='closest'
        )
        
        # Set equal aspect ratio for circular motion
        if motion_type == 'circular':
            fig_traj.update_layout(
                xaxis=dict(scaleanchor="y", scaleratio=1),
                yaxis=dict(scaleanchor="x", scaleratio=1)
            )
        
        st.plotly_chart(fig_traj, use_container_width=True)
        
        # Time series visualization
        st.markdown("### 📈 Position vs Time")
        
        fig_time = go.Figure()
        
        # Continuous motion
        fig_time.add_trace(go.Scatter(
            x=t_continuous,
            y=x_continuous,
            mode='lines',
            name='Actual X Position',
            line=dict(color='blue', width=2)
        ))
        
        # Sampled motion
        fig_time.add_trace(go.Scatter(
            x=t_frames,
            y=x_blur,
            mode='markers+lines',
            name=f'Perceived X Position ({frame_rate} fps)',
            line=dict(color='red', width=2),
            marker=dict(size=6, color='red')
        ))
        
        # Add frame timing markers
        for t_frame in t_frames[::max(1, len(t_frames)//10)]:  # Show every 10th frame
            fig_time.add_vline(
                x=t_frame,
                line=dict(color='gray', dash='dot', width=1),
                opacity=0.5
            )
        
        fig_time.update_layout(
            title="Temporal Sampling of Motion",
            xaxis_title="Time (s)",
            yaxis_title="Position",
            height=350,
            showlegend=True
        )
        
        st.plotly_chart(fig_time, use_container_width=True)
    
    # Comparative analysis
    st.markdown("---")
    st.markdown("### 🔄 Frame Rate Comparison")
    
    # Show multiple frame rates simultaneously
    frame_rates = [10, 24, 30, 60]
    colors = ['red', 'orange', 'green', 'blue']
    
    fig_compare = go.Figure()
    
    # Add continuous reference
    fig_compare.add_trace(go.Scatter(
        x=t_continuous,
        y=x_continuous,
        mode='lines',
        name='Continuous Motion',
        line=dict(color='black', width=2, dash='dash')
    ))
    
    for i, fr in enumerate(frame_rates):
        t_temp, x_temp, _ = sample_motion(t_continuous, x_continuous, y_continuous, fr)
        fig_compare.add_trace(go.Scatter(
            x=t_temp,
            y=x_temp,
            mode='markers+lines',
            name=f'{fr} fps',
            line=dict(color=colors[i], width=1),
            marker=dict(size=4, color=colors[i])
        ))
    
    fig_compare.update_layout(
        title="Frame Rate Comparison",
        xaxis_title="Time (s)",
        yaxis_title="X Position",
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig_compare, use_container_width=True)
    
    # Frequency analysis
    st.markdown("### 🌊 Frequency Domain Analysis")
    
    # Analyze the "sampling" of visual information
    sampling_freq = frame_rate
    nyquist_visual = sampling_freq / 2
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Visual Sampling Rate", f"{frame_rate} Hz")
    
    with col2:
        st.metric("Visual Nyquist Frequency", f"{nyquist_visual:.1f} Hz")
    
    with col3:
        motion_frequency = motion_speed / (2 * np.pi) if motion_type == 'circular' else motion_speed
        st.metric("Motion Frequency", f"{motion_frequency:.2f} Hz")
    
    # Theory section
    st.markdown("---")
    st.markdown("### 📚 Visual System & Signal Processing Analogy")
    
    theory_col1, theory_col2 = st.columns(2)
    
    with theory_col1:
        st.markdown("""
        **Human Visual System as Sampler:**
        
        - **Photoreceptors**: Act like sensors capturing light intensity
        - **Temporal Resolution**: ~10-60 Hz effective sampling rate
        - **Persistence of Vision**: Natural "reconstruction filter"
        - **Motion Blur**: Equivalent to anti-aliasing filter
        - **Saccadic Eye Movements**: Discrete sampling windows
        
        **Critical Frequencies:**
        - **Flicker Fusion**: 50-60 Hz (varies by individual)
        - **Motion Detection**: 5-10 Hz minimum
        - **Smooth Motion**: 24-30 Hz threshold
        """)
    
    with theory_col2:
        st.markdown("""
        **DSP Parallels:**
        
        | Visual System | DSP Equivalent |
        |---------------|----------------|
        | Frame rate | Sampling frequency |
        | Persistence of vision | Reconstruction filter |
        | Motion blur | Anti-aliasing |
        | Flicker | Aliasing artifact |
        | Smooth motion | Proper reconstruction |
        
        **Applications:**
        - Cinema: 24 fps (minimum acceptable)
        - TV: 30-60 fps (broadcast standards)
        - Gaming: 60-120 fps (competitive gaming)
        - VR: 90+ fps (motion sickness prevention)
        """)
        
        st.latex(r"\\text{Perceived Smoothness} \\propto \\log(\\text{Frame Rate})")

if __name__ == "__main__":
    main()
'''

# Save Human Eye Sampling page
with open('pages/4_👁️_Human_Eye_Sampling.py', 'w') as f:
    f.write(eye_sampling_page_code)

print("✅ Created Human Eye Sampling page")
print("File size:", len(eye_sampling_page_code), "characters")