import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

st.set_page_config(page_title="Visual Sampling & Aliasing", page_icon="👁️", layout="wide")

# --- CSS STYLING ---
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    .subtitle { text-align: center; color: #6b7280; margin-bottom: 2rem; }
    
    .game-card {
        background: #1f2937;
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #6366f1;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .status-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-weight: 600;
        text-align: center;
    }
    .status-ok { background: #dcfce7; color: #166534; border: 1px solid #166534; }
    .status-warn { background: #fef9c3; color: #854d0e; border: 1px solid #854d0e; }
    .status-danger { background: #fee2e2; color: #991b1b; border: 1px solid #991b1b; }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

def get_aliasing_status(rpm, fps, spokes):
    """Calculate the perceived effect of the wheel."""
    rps = rpm / 60
    degrees_per_frame = (rps * 360) / fps
    symmetry_angle = 360 / spokes
    
    # The math of perception (modulo arithmetic)
    effective_movement = degrees_per_frame % symmetry_angle
    
    if abs(effective_movement) < 0.1 or abs(effective_movement - symmetry_angle) < 0.1:
        return "Stationary (Sync)", "status-warn", 0
    elif effective_movement > (symmetry_angle / 2):
        # It's closer to the previous spoke, so it looks like it's going backwards
        return "Backward (Aliasing)", "status-danger", effective_movement - symmetry_angle
    else:
        return "Forward (Correct)", "status-ok", effective_movement

def render_wagon_wheel_interactive():
    """The Enhanced Wagon Wheel Experience"""
    st.markdown("### 🎡 The Wagon Wheel Effect: When Cameras Lie")
    st.markdown("This tab demonstrates **Temporal Aliasing**. When a cyclic motion matches the camera's sampling rate, the brain reconstructs the wrong motion.")
    
    # --- GAME MODE vs LAB MODE ---
    mode = st.radio("Select Mode:", ["🔬 Laboratory (Explore)", "🎮 Stroboscope Challenge (Game)"], horizontal=True)
    
    col1, col2 = st.columns([1, 1.8], gap="large")
    
    if mode == "🔬 Laboratory (Explore)":
        with col1:
            st.markdown("#### 🎛️ Lab Controls")
            rpm = st.slider("Wheel Speed (RPM)", 10, 300, 60, 5)
            fps = st.slider("Camera Frame Rate (FPS)", 5, 60, 24, 1)
            spokes = st.slider("Number of Spokes", 1, 12, 8, 1)
            show_tracker = st.checkbox("Show 'Real' Tracker (Red Dot)", value=True)
            
            # Analysis
            status_text, status_class, perceived_move = get_aliasing_status(rpm, fps, spokes)
            
            st.markdown(f"""
            <div class="status-box {status_class}">
                Perception: {status_text}
            </div>
            """, unsafe_allow_html=True)
            
            st.info(f"""
            **The Math:**
            - Spoke Interval: **{360/spokes:.1f}°**
            - Rotation per Frame: **{(rpm/60 * 360)/fps:.1f}°**
            
            The wheel moves **{(rpm/60 * 360)/fps:.1f}°** between photos. 
            {'Since this is more than half the spoke interval, your brain snaps to the **previous** spoke!' if "Backward" in status_text else 'Your brain snaps to the **next** spoke.'}
            """)

    else: # GAME MODE
        with col1:
            st.markdown("#### 🎮 The Challenge")
            st.markdown("""
            <div class="game-card">
                <strong>Goal:</strong> The wheel is spinning at a mystery speed.<br>
                Adjust the <strong>Camera FPS</strong> until the wheel appears <strong>STATIONARY</strong>.
            </div>
            """, unsafe_allow_html=True)
            
            # Initialize random RPM if not set
            if 'target_rpm' not in st.session_state:
                st.session_state.target_rpm = random.randint(100, 250)
            
            if st.button("🎲 Generate New Mystery Speed"):
                st.session_state.target_rpm = random.randint(100, 250)
            
            rpm = st.session_state.target_rpm # Fixed for the user
            spokes = 8 # Fixed for simplicity
            
            fps = st.slider("Adjust Camera FPS to Freeze the Wheel", 10, 100, 20, 1)
            show_tracker = st.checkbox("Show Hint (Red Dot)", value=False)
            
            # Check for win condition (approximate sync)
            rps = rpm / 60
            deg_per_frame = (rps * 360) / fps
            sym = 360/spokes
            modulo = deg_per_frame % sym
            
            if modulo < 1.0 or abs(modulo - sym) < 1.0:
                st.balloons()
                st.success(f"🎉 SUCCESS! You synchronized with the wheel! (Actual RPM: {rpm})")
            else:
                st.caption("Keep tuning... make the spokes stop moving.")

    with col2:
        # --- VISUALIZATION LOGIC ---
        
        # 1. The Wheel Animation
        frames = []
        num_frames = 60 # limit for performance
        
        degrees_per_frame = ((rpm / 60) * 360) / fps
        tracker_color = '#f44336' if show_tracker else 'rgba(0,0,0,0)'
        
        # Generate frames
        for k in range(num_frames):
            angle_deg = -1 * k * degrees_per_frame
            angle_rad = np.radians(angle_deg)
            
            # Spokes
            x_spokes, y_spokes = [], []
            for s in range(spokes):
                s_angle = angle_rad + (2 * np.pi * s / spokes)
                x_spokes.extend([0, np.cos(s_angle), None])
                y_spokes.extend([0, np.sin(s_angle), None])
            
            # Tracker
            tx = 0.8 * np.cos(angle_rad)
            ty = 0.8 * np.sin(angle_rad)
            
            frames.append(go.Frame(data=[
                go.Scatter(x=x_spokes, y=y_spokes),
                go.Scatter(x=[tx], y=[ty], marker=dict(color=tracker_color))
            ]))

        # Base Figure
        fig = go.Figure(
            data=[
                go.Scatter(x=frames[0].data[0].x, y=frames[0].data[0].y, mode='lines', line=dict(color='#e6eef8', width=3), name='Wheel'),
                go.Scatter(x=frames[0].data[1].x, y=frames[0].data[1].y, mode='markers', marker=dict(color=tracker_color, size=15), name='Tracker'),
                go.Scatter(x=np.cos(np.linspace(0, 2*np.pi, 100)), y=np.sin(np.linspace(0, 2*np.pi, 100)), mode='lines', line=dict(color='#6366f1', width=5), hoverinfo='skip')
            ],
            frames=frames
        )
        
        duration_per_frame = 1000 / fps
        
        fig.update_layout(
            title=f"Camera View ({fps} FPS)",
            width=500, height=500,
            xaxis=dict(visible=False, range=[-1.2, 1.2]),
            yaxis=dict(visible=False, range=[-1.2, 1.2], scaleanchor="x"),
            plot_bgcolor='rgba(0,0,0,0)',
            updatemenus=[{
                "type": "buttons",
                "buttons": [
                    {"label": "▶️ Play", "method": "animate", "args": [None, {"frame": {"duration": duration_per_frame, "redraw": True}, "fromcurrent": True}]},
                    {"label": "⏸️ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]}
                ],
                "x": 0.5, "y": -0.1, "xanchor": "center", "direction": "left"
            }]
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 2. The Aliasing Graph (The meaningful addition)
        if mode == "🔬 Laboratory (Explore)":
            st.markdown("#### 📉 Why it happens: The Aliasing Graph")
            
            # Plot Perceived RPM vs Actual RPM
            # We simulate a range of RPMs to show the "Sawtooth" or "Folding" effect
            test_rpms = np.linspace(0, 300, 300)
            perceived_rpms = []
            
            sym_angle = 360/spokes
            
            for r in test_rpms:
                d_per_f = ((r/60)*360)/fps
                eff = d_per_f % sym_angle
                if eff > sym_angle/2:
                    eff -= sym_angle # Negative perception
                
                # Convert back to RPM
                p_rpm = (eff * fps / 360) * 60
                perceived_rpms.append(p_rpm)
                
            fig_alias = go.Figure()
            fig_alias.add_trace(go.Scatter(x=test_rpms, y=perceived_rpms, name="Perceived Speed"))
            fig_alias.add_trace(go.Scatter(x=[rpm], y=[(get_aliasing_status(rpm, fps, spokes)[2] * fps / 360) * 60], 
                                         mode='markers', marker=dict(size=12, color='red'), name="Current Setting"))
            
            fig_alias.update_layout(
                title="The Aliasing Fold (Sawtooth Effect)",
                xaxis_title="Actual RPM",
                yaxis_title="Perceived RPM",
                height=300,
                margin=dict(t=30, b=0, l=0, r=0)
            )
            st.plotly_chart(fig_alias, use_container_width=True)
            st.caption("Notice how the perceived speed rises, hits a limit (Nyquist), and then plunges into negative (backward) values?")

def render_human_eye_tab():
    """The original human eye content, simplified"""
    st.markdown("### 👁️ The Eye: A Biological Sampler")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("Your eye is not a video camera. It doesn't have a shutter. Instead, it uses **Persistence of Vision**.")
        

# [Image of human retina photoreceptors structure]

        st.markdown("""
        **How it works:**
        1.  **Chemical Reaction:** Photons hit the retina (rods/cones).
        2.  **Integration Time:** The chemical reaction takes time (~10-50ms).
        3.  **Decay:** The image "fades out" slowly, not instantly.
        
        This fading creates a natural "Motion Blur" that smooths out real life. Cameras often lack this, making high-speed video look choppy (stroboscopic) unless artificial blur is added.
        """)
        
    with col2:
        # Simple interactive demo for Persistence
        st.write("#### 🧪 Demo: Persistence Trail")
        speed = st.slider("Object Speed", 1, 10, 5)
        trail_length = st.slider("Eye Persistence (Trail Length)", 0, 20, 5)
        
        t = np.linspace(0, 10, 100)
        x = np.sin(t * speed)
        
        fig = go.Figure()
        # Main dot
        fig.add_trace(go.Scatter(x=[x[-1]], y=[0], mode='markers', marker=dict(size=20, color='blue'), name="Object"))
        
        # Trail
        for i in range(trail_length):
            opacity = (trail_length - i) / trail_length
            idx = -1 - (i*2) # spacing
            if abs(idx) < len(x):
                fig.add_trace(go.Scatter(
                    x=[x[idx]], y=[0], 
                    mode='markers', 
                    marker=dict(size=20-i, color='blue', opacity=opacity),
                    showlegend=False, hoverinfo='skip'
                ))
                
        fig.update_layout(
            xaxis=dict(range=[-1.5, 1.5], showgrid=False, zeroline=False, visible=False),
            yaxis=dict(range=[-1, 1], showgrid=False, zeroline=False, visible=False),
            height=200,
            title="Simulated Retinal Persistence"
        )
        st.plotly_chart(fig, use_container_width=True)

def main():
    st.markdown('<h1 class="main-title">👁️ Vision, Sampling & Aliasing</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Why wheels go backward and movies look smooth</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🎡 The Wagon Wheel (Aliasing)", "👁️ The Human Eye", "📚 Theory & Diagrams"])

    with tab1:
        render_wagon_wheel_interactive()
        
    with tab2:
        render_human_eye_tab()
        
    with tab3:
        st.markdown("### 📚 Theoretical Concepts")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 1. Sampling Theorem")
            st.markdown("To capture motion accurately, you must sample at least twice as fast as the motion frequency.")
            st.latex(r"f_{sample} > 2 \cdot f_{motion}")
            
            
        with c2:
            st.markdown("#### 2. Stroboscopic Effect")
            st.markdown("When the sampling rate matches the rotation speed (or a harmonic), the object appears frozen.")
            

# [Image of stroboscope light effect diagram]


if __name__ == "__main__":
    main()