import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import use
use('Agg')

st.set_page_config(page_title="Human Eye Sampling", page_icon="👁️", layout="wide")

# Enhanced CSS with better contrast and accessibility
st.markdown("""
<style>
    /* Main title */
    .main-title {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #4b5563;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    
    /* Status cards with better contrast */
    .status-card {
        background: white;
        border: 2px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .status-excellent {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 700;
        font-size: 1.15rem;
        box-shadow: 0 4px 12px rgba(16,185,129,0.3);
        margin: 1rem 0;
    }
    
    .status-good {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 700;
        font-size: 1.15rem;
        box-shadow: 0 4px 12px rgba(59,130,246,0.3);
        margin: 1rem 0;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 700;
        font-size: 1.15rem;
        box-shadow: 0 4px 12px rgba(245,158,11,0.3);
        margin: 1rem 0;
    }
    
    .status-poor {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 700;
        font-size: 1.15rem;
        box-shadow: 0 4px 12px rgba(239,68,68,0.3);
        margin: 1rem 0;
        animation: pulse-warning 2s infinite;
    }
    
    @keyframes pulse-warning {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.85; }
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        transition: all 0.3s ease;
        border: 2px solid #e5e7eb;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(99,102,241,0.2);
        border-color: #6366f1;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.3rem 0;
    }
    
    .metric-label {
        color: #1f2937;
        font-size: 0.95rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }
    
    .metric-sublabel {
        color: #6b7280;
        font-size: 0.8rem;
        margin-top: 0.3rem;
    }
    
    /* Info boxes with better contrast */
    .info-box {
        background: #eff6ff;
        border: 2px solid #3b82f6;
        color: #1e40af;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .info-box strong {
        color: #1e3a8a;
    }
    
    .warning-box {
        background: #fef3c7;
        border: 2px solid #f59e0b;
        color: #92400e;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .warning-box strong {
        color: #78350f;
    }
    
    .success-box {
        background: #d1fae5;
        border: 2px solid #10b981;
        color: #065f46;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .success-box strong {
        color: #064e3b;
    }
    
    .danger-box {
        background: #fee2e2;
        border: 2px solid #ef4444;
        color: #991b1b;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .danger-box strong {
        color: #7f1d1d;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1f2937;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #6366f1;
    }
    
    .subsection-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #374151;
        margin: 1rem 0 0.5rem 0;
        padding-left: 0.5rem;
        border-left: 4px solid #8b5cf6;
    }
    
    /* Control panel styling */
    .control-panel {
        background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #e5e7eb;
        margin: 1rem 0;
    }
    
    /* Comparison table */
    .comparison-table {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        border: 2px solid #e5e7eb;
    }
    
    .comparison-table table {
        width: 100%;
        border-collapse: collapse;
    }
    
    .comparison-table th {
        background: #6366f1;
        color: white;
        padding: 0.75rem;
        font-weight: 600;
        text-align: left;
    }
    
    .comparison-table td {
        padding: 0.75rem;
        border-bottom: 1px solid #e5e7eb;
        color: #374151;
    }
    
    /* Preset buttons */
    .preset-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.5rem;
        margin: 1rem 0;
    }
    
    /* Frame rate indicator */
    .fps-indicator {
        display: inline-block;
        padding: 8px 16px;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border-radius: 20px;
        font-weight: 700;
        margin: 5px;
        font-size: 1rem;
        box-shadow: 0 2px 8px rgba(99,102,241,0.3);
    }
    
    /* Motion type badges */
    .motion-badge {
        display: inline-block;
        padding: 6px 14px;
        background: white;
        border: 2px solid #6366f1;
        color: #6366f1;
        border-radius: 20px;
        font-weight: 600;
        margin: 3px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

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
def wagon_wheel_frames(num_spokes=8, wheel_radius=1.0, rpm=60, frame_rate=24, duration=2.0):
    """Generate sampled positions of wheel spokes for visualization of wagon wheel effect."""
    # Calculate angular velocity (rad/s)
    angular_velocity = 2 * np.pi * rpm / 60
    # Number of frames
    num_frames = int(frame_rate * duration)
    t_samples = np.linspace(0, duration, num_frames)
    theta_samples = angular_velocity * t_samples
    # Calculate spoke end positions for each frame
    spokes_angles = np.linspace(0, 2*np.pi, num_spokes, endpoint=False)
    all_spokes = []
    for theta in theta_samples:
        frame = []
        for spoke_angle in spokes_angles:
            angle = theta + spoke_angle
            x = wheel_radius * np.cos(angle)
            y = wheel_radius * np.sin(angle)
            frame.append((x, y))
        all_spokes.append(frame)
    return all_spokes, t_samples

def sample_motion(t, x, y, frame_rate):
    """Sample the continuous motion at given frame rate"""
    dt = 1 / frame_rate
    t_frames = np.arange(0, t[-1], dt)
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

def get_perception_status(frame_rate):
    """Get perception quality with appropriate styling"""
    if frame_rate >= 60:
        return "🌟 Very Smooth - High Refresh Rate", "status-excellent", "Buttery smooth motion, ideal for gaming and VR"
    elif frame_rate >= 30:
        return "✅ Smooth - Standard Quality", "status-good", "Comfortable viewing experience, TV standard"
    elif frame_rate >= 24:
        return "🎬 Acceptable - Cinematic", "status-warning", "Minimum for perceived smooth motion, cinema standard"
    elif frame_rate >= 15:
        return "⚠️ Noticeable Stuttering", "status-warning", "Motion appears choppy, not ideal for viewing"
    else:
        return "❌ Very Choppy - Poor Quality", "status-poor", "Severe motion artifacts, uncomfortable to watch"

def main():
    # Header
    st.markdown('<h1 class="main-title">👁️ Human Vision as a Sampling System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Explore how our eyes process motion through the lens of digital signal processing</p>', unsafe_allow_html=True)
    
    # Info banner
    with st.expander("ℹ️ Understanding Visual Perception & Sampling", expanded=False):
        st.markdown("""
        ### The Eye as a Temporal Sampler
        
        Our visual system doesn't capture the world continuously—it processes information in **discrete snapshots**, 
        much like a digital camera or display screen. This creates fascinating parallels with digital signal processing:
        
        **Key Concepts:**
        - 👁️ **Photoreceptors** respond to light changes with a temporal resolution of ~10-60 Hz
        - 🧠 **Neural processing** integrates discrete visual inputs into perceived smooth motion
        - ⏱️ **Persistence of vision** acts like a natural reconstruction filter (~100ms decay)
        - 🎬 **Critical Flicker Fusion** (~50-60 Hz) is our visual "Nyquist frequency"
        
        **Why This Matters:**
        - Cinema runs at 24 fps (minimum for perceived smooth motion)
        - TV broadcasts at 30-60 fps (comfortable viewing)
        - Gaming targets 60-144 fps (reduced input lag and smoother response)
        - VR requires 90+ fps (prevents motion sickness)
        """)
    
    st.markdown("---")
    
    # Initialize session state
    if 'preset_applied' not in st.session_state:
        st.session_state.preset_applied = None
    
    # Main layout - Better organization
    col1, col2 = st.columns([1, 2.5], gap="large")
    
    with col1:
        st.markdown('<div class="section-header">🎮 Simulation Controls</div>', unsafe_allow_html=True)
        
        # Quick presets
        st.markdown('<div class="subsection-header">🎯 Quick Presets</div>', unsafe_allow_html=True)
        
        preset_cols = st.columns(2)
        with preset_cols[0]:
            if st.button("🎬 Cinema", use_container_width=True, help="24 fps - Traditional film"):
                st.session_state.preset_applied = 'cinema'
                st.rerun()
            if st.button("🎮 Gaming", use_container_width=True, help="60 fps - Smooth gaming"):
                st.session_state.preset_applied = 'gaming'
                st.rerun()
        
        with preset_cols[1]:
            if st.button("📺 TV", use_container_width=True, help="30 fps - Standard broadcast"):
                st.session_state.preset_applied = 'tv'
                st.rerun()
            if st.button("⚠️ Choppy", use_container_width=True, help="10 fps - Visible stuttering"):
                st.session_state.preset_applied = 'choppy'
                st.rerun()
        
        st.markdown("---")
        
        # Apply presets
        if st.session_state.preset_applied == 'cinema':
            frame_rate, motion_type, motion_speed = 24, 'linear', 1.5
        elif st.session_state.preset_applied == 'tv':
            frame_rate, motion_type, motion_speed = 30, 'circular', 1.0
        elif st.session_state.preset_applied == 'gaming':
            frame_rate, motion_type, motion_speed = 60, 'complex', 2.0
        elif st.session_state.preset_applied == 'choppy':
            frame_rate, motion_type, motion_speed = 10, 'oscillating', 1.5
        else:
            frame_rate, motion_type, motion_speed = 30, 'linear', 1.0
        
        if st.session_state.preset_applied:
            st.session_state.preset_applied = None
        
        # Frame rate control
        st.markdown('<div class="subsection-header">👁️ Visual Frame Rate</div>', unsafe_allow_html=True)
        frame_rate = st.slider(
            "Frames Per Second (fps)",
            5, 120, frame_rate, 5,
            help="Simulates display refresh rate or eye temporal resolution"
        )
        
        st.markdown(f'<span class="fps-indicator">{frame_rate} FPS</span>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Motion parameters
        st.markdown('<div class="subsection-header">🏃 Motion Configuration</div>', unsafe_allow_html=True)
        
        motion_type = st.selectbox(
            "Motion Pattern",
            ['linear', 'circular', 'oscillating', 'complex'],
            index=['linear', 'circular', 'oscillating', 'complex'].index(motion_type),
            help="Type of movement to simulate"
        )
        
        motion_descriptions = {
            'linear': '➡️ Straight line motion',
            'circular': '⭕ Rotating in a circle',
            'oscillating': '↔️ Back and forth movement',
            'complex': '🌀 Combined motions'
        }
        st.caption(motion_descriptions[motion_type])
        
        motion_speed = st.slider(
            "Motion Speed",
            0.5, 3.0, motion_speed, 0.1,
            help="Velocity of the moving object"
        )
        
        st.markdown("---")
        
        # Visual effects
        st.markdown('<div class="subsection-header">🎨 Visual Effects</div>', unsafe_allow_html=True)
        
        show_motion_blur = st.checkbox("Enable Motion Blur", value=False,
                                      help="Simulates blur from fast motion")
        
        blur_intensity = 0.1
        if show_motion_blur:
            blur_intensity = st.slider("Blur Intensity", 0.0, 0.5, 0.1, 0.05)
        
        show_persistence = st.checkbox("Show Persistence of Vision", value=False,
                                      help="Displays afterimage effect")
        
        show_frame_markers = st.checkbox("Show Frame Timing", value=True,
                                        help="Display frame capture moments")
    
    with col2:
        # Get perception status
        status_text, status_class, status_desc = get_perception_status(frame_rate)
        
        # Status display
        st.markdown(f"""
        <div class="{status_class}">
            <div>{status_text}</div>
            <div style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.95;">{status_desc}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics dashboard
        st.markdown('<div class="section-header">📊 Visual System Metrics</div>', unsafe_allow_html=True)
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        nyquist_visual = frame_rate / 2
        critical_fusion = 60  # Typical human CFF
        motion_frequency = motion_speed / (2 * np.pi) if motion_type == 'circular' else motion_speed / 2
        
        with metric_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Frame Rate</div>
                <div class="metric-value">{frame_rate}</div>
                <div class="metric-sublabel">Hz (fps)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Nyquist Freq</div>
                <div class="metric-value">{nyquist_visual:.0f}</div>
                <div class="metric-sublabel">Hz (fs/2)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col3:
            ratio = frame_rate / critical_fusion
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">vs CFF</div>
                <div class="metric-value">{ratio:.2f}x</div>
                <div class="metric-sublabel">Critical Fusion</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Motion Freq</div>
                <div class="metric-value">{motion_frequency:.1f}</div>
                <div class="metric-sublabel">Hz</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Generate motion data
        duration = 4.0
        t_continuous = np.linspace(0, duration, 1000)
        x_continuous, y_continuous = generate_moving_object_data(t_continuous, motion_type, motion_speed)
        t_frames, x_frames, y_frames = sample_motion(t_continuous, x_continuous, y_continuous, frame_rate)
        
        if show_motion_blur:
            x_blur = simulate_motion_blur(x_frames, blur_intensity)
            y_blur = simulate_motion_blur(y_frames, blur_intensity)
        else:
            x_blur = x_frames
            y_blur = y_frames
        
        # Tabbed visualization
        tab1, tab2, tab3, tab4,tab5 = st.tabs(["🎥 Motion Trajectory", "📈 Temporal Analysis", "🔄 Frame Rate Comparison", "📚 Theory", "🎡 Wagon Wheel Effect"])
        
        with tab1:
            st.markdown("### Spatial Motion Visualization")
            
            fig_traj = go.Figure()
            
            # Continuous trajectory
            fig_traj.add_trace(go.Scatter(
                x=x_continuous,
                y=y_continuous,
                mode='lines',
                name='Actual Motion Path',
                line=dict(color='#93c5fd', width=4, dash='dash'),
                opacity=0.6,
                hovertemplate='Actual: (%{x:.2f}, %{y:.2f})<extra></extra>'
            ))
            
            # Sampled/perceived positions
            fig_traj.add_trace(go.Scatter(
                x=x_blur,
                y=y_blur,
                mode='markers+lines',
                name=f'Perceived Motion ({frame_rate} fps)',
                line=dict(color='#ef4444', width=3),
                marker=dict(size=10, color='#dc2626', symbol='circle',
                          line=dict(color='white', width=2)),
                hovertemplate='Frame: (%{x:.2f}, %{y:.2f})<extra></extra>'
            ))
            
            # Persistence of vision effect
            if show_persistence and len(x_frames) >= 5:
                trail_length = min(8, len(x_frames) - 1)
                for i in range(1, trail_length + 1):
                    alpha = 1.0 - (i / trail_length)
                    idx = len(x_frames) - i - 1
                    if idx >= 0:
                        fig_traj.add_trace(go.Scatter(
                            x=[x_frames[idx]],
                            y=[y_frames[idx]],
                            mode='markers',
                            marker=dict(
                                size=15 - i,
                                color='#fbbf24',
                                symbol='circle-open',
                                opacity=alpha,
                                line=dict(width=2)
                            ),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
            
            # Start and end markers
            fig_traj.add_trace(go.Scatter(
                x=[x_continuous[0]],
                y=[y_continuous[0]],
                mode='markers',
                name='Start',
                marker=dict(size=15, color='#10b981', symbol='star'),
                hoverinfo='skip'
            ))
            
            fig_traj.add_trace(go.Scatter(
                x=[x_continuous[-1]],
                y=[y_continuous[-1]],
                mode='markers',
                name='End',
                marker=dict(size=15, color='#8b5cf6', symbol='square'),
                hoverinfo='skip'
            ))
            
            fig_traj.update_layout(
                title=dict(
                    text=f"Motion Trajectory at {frame_rate} fps",
                    font=dict(size=16, color='#1f2937')
                ),
                xaxis_title="X Position",
                yaxis_title="Y Position",
                height=500,
                template='plotly_white',
                hovermode='closest',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            if motion_type == 'circular':
                fig_traj.update_layout(
                    xaxis=dict(scaleanchor="y", scaleratio=1),
                    yaxis=dict(scaleanchor="x", scaleratio=1)
                )
            
            st.plotly_chart(fig_traj, use_container_width=True)
            
            # Explanation based on frame rate
            if frame_rate < 24:
                st.markdown("""
                <div class="danger-box">
                    <strong>⚠️ Below Cinematic Threshold:</strong> At this frame rate, individual frames are clearly visible, 
                    creating a "stuttering" or "stop-motion" effect. This is below the threshold for smooth perceived motion.
                </div>
                """, unsafe_allow_html=True)
            elif frame_rate < 30:
                st.markdown("""
                <div class="warning-box">
                    <strong>🎬 Cinematic Range:</strong> This frame rate is used in traditional cinema (24 fps). 
                    While motion appears relatively smooth, fast movements may still show slight judder.
                </div>
                """, unsafe_allow_html=True)
            elif frame_rate < 60:
                st.markdown("""
                <div class="info-box">
                    <strong>📺 Standard Quality:</strong> This is the television standard (30-60 fps). 
                    Motion appears smooth and comfortable for most viewers.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-box">
                    <strong>🌟 High Refresh Rate:</strong> At 60+ fps, motion appears exceptionally smooth. 
                    This is preferred for gaming, sports, and VR applications where responsiveness matters.
                </div>
                """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("### Position Over Time")
            
            fig_time = make_subplots(
                rows=2, cols=1,
                subplot_titles=("X Position vs Time", "Y Position vs Time"),
                vertical_spacing=0.12,
                row_heights=[0.5, 0.5]
            )
            
            # X position
            fig_time.add_trace(
                go.Scatter(
                    x=t_continuous,
                    y=x_continuous,
                    mode='lines',
                    name='Actual X',
                    line=dict(color='#3b82f6', width=3),
                    hovertemplate='Time: %{x:.3f}s<br>X: %{y:.3f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            fig_time.add_trace(
                go.Scatter(
                    x=t_frames,
                    y=x_blur,
                    mode='markers+lines',
                    name=f'Sampled X ({frame_rate} fps)',
                    line=dict(color='#ef4444', width=2),
                    marker=dict(size=8, color='#dc2626'),
                    hovertemplate='Frame: %{x:.3f}s<br>X: %{y:.3f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Y position
            fig_time.add_trace(
                go.Scatter(
                    x=t_continuous,
                    y=y_continuous,
                    mode='lines',
                    name='Actual Y',
                    line=dict(color='#10b981', width=3),
                    showlegend=False,
                    hovertemplate='Time: %{x:.3f}s<br>Y: %{y:.3f}<extra></extra>'
                ),
                row=2, col=1
            )
            
            fig_time.add_trace(
                go.Scatter(
                    x=t_frames,
                    y=y_blur,
                    mode='markers+lines',
                    name=f'Sampled Y ({frame_rate} fps)',
                    line=dict(color='#f59e0b', width=2),
                    marker=dict(size=8, color='#d97706'),
                    showlegend=False,
                    hovertemplate='Frame: %{x:.3f}s<br>Y: %{y:.3f}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Frame timing markers
            if show_frame_markers:
                for t_frame in t_frames[::max(1, len(t_frames)//12)]:
                    fig_time.add_vline(
                        x=t_frame,
                        line=dict(color='#9ca3af', dash='dot', width=1),
                        opacity=0.4,
                        row='all'
                    )
            
            fig_time.update_xaxes(title_text="Time (seconds)", row=2, col=1)
            fig_time.update_yaxes(title_text="X Position", row=1, col=1)
            fig_time.update_yaxes(title_text="Y Position", row=2, col=1)
            
            fig_time.update_layout(
                height=600,
                template='plotly_white',
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig_time, use_container_width=True)
            
            st.markdown("""
            <div class="info-box">
                <strong>💡 Understanding the Plot:</strong><br>
                • <strong>Solid lines:</strong> Actual continuous motion<br>
                • <strong>Red dots & lines:</strong> What the eye "sees" at each frame<br>
                • <strong>Vertical dotted lines:</strong> Frame capture moments<br>
                • Notice how lower frame rates miss details in rapid motion changes
            </div>
            """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("### Frame Rate Comparison")
            
            # Compare multiple frame rates
            frame_rates = [10, 24, 30, 60, 120]
            colors = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#3b82f6']
            
            fig_compare = go.Figure()
            
            # Continuous reference
            fig_compare.add_trace(go.Scatter(
                x=t_continuous,
                y=x_continuous,
                mode='lines',
                name='Continuous (Reference)',
                line=dict(color='#9ca3af', width=4, dash='dash'),
                opacity=0.5
            ))
            
            for i, fr in enumerate(frame_rates):
                t_temp, x_temp, _ = sample_motion(t_continuous, x_continuous, y_continuous, fr)
                
                # Determine line style based on quality
                if fr >= 60:
                    line_width = 2
                elif fr >= 24:
                    line_width = 1.5
                else:
                    line_width = 1
                
                fig_compare.add_trace(go.Scatter(
                    x=t_temp,
                    y=x_temp,
                    mode='markers+lines',
                    name=f'{fr} fps',
                    line=dict(color=colors[i], width=line_width),
                    marker=dict(size=5, color=colors[i]),
                    hovertemplate=f'{fr} fps<br>Time: %{{x:.2f}}s<br>Position: %{{y:.2f}}<extra></extra>'
                ))
            
            fig_compare.update_layout(
                title="Multi-Rate Comparison: How Frame Rate Affects Motion Capture",
                xaxis_title="Time (seconds)",
                yaxis_title="X Position",
                height=500,
                template='plotly_white',
                hovermode='x unified',
                legend=dict(
                    title="Frame Rates",
                    orientation="v",
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99,
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#e5e7eb",
                    borderwidth=2
                )
            )
            
            st.plotly_chart(fig_compare, use_container_width=True)
            
            # Frame rate comparison table
            st.markdown("### 📊 Frame Rate Standards")
            
            comparison_data = {
                "10 fps": ("Very Choppy", "❌", "Early animations, time-lapse"),
                "24 fps": ("Cinematic", "🎬", "Traditional film, movies"),
                "30 fps": ("Standard TV", "📺", "Television broadcasts, streaming"),
                "60 fps": ("Smooth", "✅", "Modern displays, gaming, sports"),
                "120 fps": ("Very Smooth", "🌟", "High-end gaming, VR, slow motion")
            }
            
            st.markdown('<div class="comparison-table">', unsafe_allow_html=True)
            st.markdown("""
            | Frame Rate | Quality | Icon | Common Applications |
            |------------|---------|------|---------------------|
            | **10 fps** | Very Choppy | ❌ | Early animations, time-lapse |
            | **24 fps** | Cinematic | 🎬 | Traditional film, movies |
            | **30 fps** | Standard TV | 📺 | Television broadcasts, streaming |
            | **60 fps** | Smooth | ✅ | Modern displays, gaming, sports |
            | **120 fps** | Very Smooth | 🌟 | High-end gaming, VR, slow motion |
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="success-box">
                <strong>✨ Frame Rate Selection Guide:</strong><br>
                • <strong>24 fps:</strong> Minimum for acceptable motion perception<br>
                • <strong>30 fps:</strong> Standard for comfortable viewing<br>
                • <strong>60 fps:</strong> Recommended for fast action and gaming<br>
                • <strong>120+ fps:</strong> Professional/competitive applications
            </div>
            """, unsafe_allow_html=True)
        
        with tab4:
            st.markdown("### 🧠 Visual Perception & Signal Processing")
            
            theory_col1, theory_col2 = st.columns(2)
            
            with theory_col1:
                st.markdown("#### Human Visual System as Sampler")
                
                st.markdown("""
                <div class="info-box">
                    <strong>🔬 Biological Components:</strong><br><br>
                    
                    <strong>Photoreceptors (Rods & Cones):</strong><br>
                    • Act as light intensity sensors<br>
                    • Temporal response: ~10-60 Hz<br>
                    • Rods: Better for low light, slower<br>
                    • Cones: Color vision, faster response<br><br>
                    
                    <strong>Neural Processing:</strong><br>
                    • Retinal ganglion cells integrate signals<br>
                    • Visual cortex processes motion<br>
                    • Creates perception from discrete inputs<br><br>
                    
                    <strong>Persistence of Vision:</strong><br>
                    • ~100 millisecond decay time<br>
                    • Acts like natural reconstruction filter<br>
                    • Allows discrete frames to blend smoothly
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                st.markdown("#### Critical Frequencies")
                
                st.markdown(f"""
                <div class="status-card">
                    <strong>📏 Current Configuration:</strong><br>
                    • Frame Rate: <strong>{frame_rate} Hz</strong><br>
                    • Nyquist Frequency: <strong>{nyquist_visual:.0f} Hz</strong><br>
                    • Critical Flicker Fusion: <strong>~50-60 Hz</strong><br><br>
                    
                    <strong>Status:</strong> {'✅ Above CFF - No flicker' if frame_rate >= 50 else '⚠️ Below CFF - Potential flicker visible'}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                st.markdown("#### Mathematical Model")
                st.latex(r"\text{Smoothness} \propto \log(\text{Frame Rate})")
                
                st.markdown("""
                The perceived smoothness increases logarithmically with frame rate, 
                meaning doubling from 30 to 60 fps has more impact than 60 to 120 fps.
                """)
            
            with theory_col2:
                st.markdown("#### DSP Parallels")
                
                st.markdown("""
                <div class="comparison-table">
                
                | Visual System | DSP Equivalent | Function |
                |---------------|----------------|----------|
                | **Photoreceptors** | Sensors | Light → Electrical signal |
                | **Frame rate** | Sampling frequency | Temporal resolution |
                | **Persistence** | Reconstruction filter | Smoothing discrete samples |
                | **Motion blur** | Anti-aliasing | Prevents high-freq artifacts |
                | **Flicker** | Aliasing | Below Nyquist artifacts |
                | **CFF (~60 Hz)** | Nyquist frequency | Perceptual limit |
                | **Smooth motion** | Proper reconstruction | Adequate sampling |
                
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                st.markdown("#### Real-World Applications")
                
                st.markdown("""
                <div class="success-box">
                    <strong>🎮 Gaming:</strong><br>
                    • 60 fps minimum for smooth gameplay<br>
                    • 144-240 fps for competitive gaming<br>
                    • Higher rates reduce input lag perception<br><br>
                    
                    <strong>🎬 Cinema:</strong><br>
                    • 24 fps: Traditional film standard<br>
                    • 48 fps: High Frame Rate (HFR) cinema<br>
                    • Motion blur added artificially (180° shutter)<br><br>
                    
                    <strong>📺 Broadcasting:</strong><br>
                    • NTSC: 30 fps (29.97 interlaced)<br>
                    • PAL: 25 fps (50 Hz regions)<br>
                    • Modern: 60-120 fps (sports, HDR)<br><br>
                    
                    <strong>🥽 VR/AR:</strong><br>
                    • 90 fps minimum (prevents nausea)<br>
                    • 120+ fps ideal for immersion<br>
                    • Low latency critical (<20ms)
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                st.markdown("#### Key Formulas")
                
                st.markdown("**Critical Flicker Fusion:**")
                st.latex(r"CFF = \frac{1}{T_{decay}}")
                st.caption("Where T_decay is photoreceptor response time (~16-20ms)")
                
                st.markdown("**Temporal Resolution:**")
                st.latex(r"\Delta t = \frac{1}{f_s}")
                st.caption(f"At {frame_rate} fps, Δt = {1000/frame_rate:.1f} ms between frames")
        with tab5:
            st.markdown("### 🛞 Wagon Wheel Effect (Aliasing)")
            st.markdown("""
            <div class="info-box">
                Spinning wheels can appear to rotate backward or stand still in videos and movies 
                due to temporal sampling aliasing. The illusion is strongest when the frame rate and 
                rotation speed are mismatched. This is known as the <strong>wagon wheel effect</strong>.
            </div>
            """, unsafe_allow_html=True)
            
            wheel_rpm = st.slider("Wheel RPM (rotations per minute)", 10, 180, 60, 10)
            num_spokes = st.slider("Number of Spokes", 4, 12, 8, 1)
            ww_frame_rate = st.slider("Frame Rate (fps)", 8, 60, frame_rate, 2)
            duration = 1.5  # seconds
            
            all_spokes, t_samples = wagon_wheel_frames(num_spokes, 1.0, wheel_rpm, ww_frame_rate, duration)
            fig_wheel = plt.figure(figsize=(8,3))
            for idx, frame in enumerate(all_spokes[::max(1, int(len(all_spokes)/12))]):
                x0, y0 = 0, 0
                for x1, y1 in frame:
                    plt.plot([x0, x1], [y0, y1], 'o-', color='#6366f1', linewidth=3)
                plt.gca().set_aspect('equal')

            plt.title(f"Wheel Sampled at {ww_frame_rate} fps, {wheel_rpm} RPM")
            plt.axis('off')
            st.pyplot(fig_wheel)

            # Explanation box
            st.markdown("""
            <div class="danger-box">
                <strong>Aliasing Illusion:</strong><br>
                If the wheel rotates by close to an integer multiple of the spoke spacing
                (e.g., 360°/num_spokes) per frame, the spokes align frame-to-frame,
                appearing stationary.<br>
                If rotation per frame <i>exceeds</i> spoke spacing, the wheel can appear to rotate <b>backwards</b>.<br>
                Try changing the RPM and frame rate!
            </div>
            """, unsafe_allow_html=True)
            # Interactive demonstrations
            st.markdown("---")
            st.markdown('<div class="section-header">🎮 Interactive Experiments</div>', unsafe_allow_html=True)
            
            exp_col1, exp_col2 = st.columns(2)
            
    with exp_col1:
        st.markdown("""
        <div class="info-box">
            <strong>🧪 Experiment 1: Find Your CFF</strong><br><br>
            
            <strong>Instructions:</strong><br>
            1. Start with a low frame rate (10-15 fps)<br>
            2. Gradually increase the slider<br>
            3. Note when you stop seeing individual frames<br>
            4. This is approximately your Critical Flicker Fusion frequency!<br><br>
            
            <strong>Expected Result:</strong><br>
            Most people stop seeing flicker between 50-60 Hz, though this varies by 
            lighting conditions, age, and fatigue.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
            <strong>🎬 Experiment 2: Cinema vs Gaming</strong><br><br>
            
            <strong>Compare:</strong><br>
            • Set to 24 fps (Cinema preset)<br>
            • Then switch to 60 fps (Gaming preset)<br>
            • Observe the difference in motion smoothness<br><br>
            
            <strong>Why Cinema Uses 24 fps:</strong><br>
            Film standard established in 1920s as economic compromise. Motion blur 
            from mechanical shutter helps smooth the motion. Digital displays at 
            24fps can appear stuttery without blur.
        </div>
        """, unsafe_allow_html=True)
    
    with exp_col2:
        st.markdown("""
        <div class="success-box">
            <strong>🔬 Experiment 3: Motion Blur Effect</strong><br><br>
            
            <strong>Try This:</strong><br>
            1. Set frame rate to 24 fps<br>
            2. Enable "Motion Blur"<br>
            3. Adjust blur intensity<br>
            4. Compare with blur disabled<br><br>
            
            <strong>Observation:</strong><br>
            Motion blur helps mask the discrete nature of low frame rates, 
            making motion appear smoother. This is why cinema looks smooth 
            despite 24 fps.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="status-card">
            <strong>📊 Your Current Setup Analysis:</strong><br><br>
    
            <strong>Configuration:</strong><br>
            • Frame Rate: <strong>{frame_rate} fps</strong><br>
            • Motion Type: <strong>{motion_type.title()}</strong><br>
            • Speed: <strong>{motion_speed:.1f}x</strong><br><br>
    
            <strong>Assessment:</strong><br>
            {'✅ Frame rate exceeds CFF - Flicker-free viewing' if frame_rate >= 50 else '⚠️ Below CFF - Flicker may be visible'}<br>
            {'✅ Suitable for smooth motion perception' if frame_rate >= 30 else '❌ Below smooth motion threshold'}<br>
            {'✅ Gaming-grade refresh rate' if frame_rate >= 60 else '📺 Standard viewing rate' if frame_rate >= 30 else '🎬 Cinematic rate' if frame_rate >= 24 else '❌ Below acceptable standards'}
        </div>
        """.format(frame_rate=frame_rate, motion_type=motion_type, motion_speed=motion_speed), unsafe_allow_html=True)
    
    # Key takeaways section
    st.markdown("---")
    st.markdown('<div class="section-header">💡 Key Takeaways</div>', unsafe_allow_html=True)
    
    takeaway_cols = st.columns(3)
    
    with takeaway_cols[0]:
        st.markdown("""
        <div class="info-box">
            <strong>🎯 The Visual System:</strong><br>
            Our eyes don't capture continuous motion—they process <strong>discrete snapshots</strong> 
            that our brain integrates into smooth perception. This is fundamentally similar to 
            digital sampling in signal processing.
        </div>
        """, unsafe_allow_html=True)
    
    with takeaway_cols[1]:
        st.markdown("""
        <div class="success-box">
            <strong>📏 The Threshold:</strong><br>
            The <strong>Critical Flicker Fusion</strong> frequency (~50-60 Hz) acts as our visual 
            "Nyquist frequency." Above this rate, discrete frames appear as continuous motion. 
            Below it, flicker becomes perceptible.
        </div>
        """, unsafe_allow_html=True)
    
    with takeaway_cols[2]:
        st.markdown("""
        <div class="warning-box">
            <strong>⚡ The Application:</strong><br>
            Understanding visual sampling helps optimize display technology: 24 fps works for 
            cinema (with blur), 60 fps for TV, and 120+ fps for VR and competitive gaming 
            where latency matters.
        </div>
        """, unsafe_allow_html=True)
    
    # Additional resources
    st.markdown("---")
    with st.expander("📖 Further Reading & Technical Details"):
        st.markdown("""
        ### Deep Dive into Visual Perception
        
        **Photoreceptor Response:**
        - Rods: Peak sensitivity ~500nm, temporal response ~100ms
        - Cones: Three types (S, M, L), faster response ~10-50ms
        - Temporal acuity decreases with age and in peripheral vision
        
        **Wagon Wheel Effect:**
        - Classic aliasing demonstration in visual perception
        - Wheels appear to rotate backward at certain speeds
        - Caused by sampling rate (frame rate) vs. motion frequency
        
        **Saccadic Eye Movements:**
        - Rapid jumps (3-4 per second)
        - Brain "fills in" during saccades (saccadic masking)
        - Another form of temporal sampling in vision
        
        **Display Technologies:**
        - CRT: Phosphor persistence acts as reconstruction filter
        - LCD: Sample-and-hold creates different motion artifact
        - OLED: Fast response, requires higher frame rates
        - Black Frame Insertion (BFI): Reduces motion blur artificially
        
        **Future Developments:**
        - Variable refresh rate (VRR/FreeSync/G-Sync)
        - 240-360 Hz gaming displays
        - Micro-LED with per-pixel control
        - Computational display technologies
        """)

if __name__ == "__main__":
    main()