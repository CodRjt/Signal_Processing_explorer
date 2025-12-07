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
    """TAB 2: Biological Compression (Chroma Subsampling)"""
    
    st.markdown("### 🧠 Biological Compression: Rods vs. Cones")
    st.caption("Why your brain prioritizes brightness over color, and how Netflix uses this to save bandwidth.")

    # --- THE CONCEPT ---
    col_bio, col_dsp = st.columns([1, 1], gap="large")
    
    with col_bio:
        st.info("##### 👁️ The Biological Hardware")
        st.markdown("""
        Your retina is an uneven sensor array:
        * **120 Million Rods:** Sensitive to **Brightness (Luma)**. High definition.
        * **6 Million Cones:** Sensitive to **Color (Chroma)**. Low definition.
        
        **Result:** You see "Black & White" in 4K resolution, but "Color" in roughly 480p!
        """)
        
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/3/3c/Human_photoreceptor_distribution.svg",
            caption="The Retina: Huge number of Rods (Purple) vs. few Cones (Blue/Green/Red).",
            use_container_width=True
        )

    with col_dsp:
        st.success("##### 💾 The DSP Application: Chroma Subsampling")
        st.markdown("""
        Since the eye is bad at seeing color detail, engineers realized we can **compress signals** by throwing away color data.
        
        * **YUV 4:4:4:** Full Color Detail (Raw).
        * **YUV 4:2:0:** Quarter Color Detail (JPEG/MP4).
        
        This technique saves **50% of bandwidth** with almost zero perceived quality loss.
        """)

    st.markdown("---")

    # --- THE INTERACTIVE LAB ---
    st.markdown("#### 🧪 The Bandwidth Hack Experiment")
    st.markdown("Use the sliders to lower the resolution of the **Structure (Brightness)** vs. the **Paint (Color)**. See which one breaks the image first.")

    uploaded_file = st.file_uploader("📸 Upload your own photo (optional)", type=['jpg', 'png', 'jpeg'])
    
    img_arr = None
    
    # 1. Load Image (Custom or Default)
    if uploaded_file is not None:
        try:
            from PIL import Image
            original_pil = Image.open(uploaded_file).convert('RGB')
            # Resize giant images to keep app fast
            original_pil.thumbnail((800, 800)) 
            img_arr = np.array(original_pil)
        except Exception as e:
            st.error(f"Error loading uploaded image: {e}")
    
    # If no upload, load default sample from Web
    if img_arr is None:
        try:
            from PIL import Image
            import requests
            from io import BytesIO
            
            # High detail image to show contrast vs color
            url = "https://upload.wikimedia.org/wikipedia/commons/c/c0/Ara_macao_qtl1.jpg" 
            
            # --- FIX: HEADERS ADDED HERE ---
            # Wikipedia requires a User-Agent or it will return 403 Forbidden
            headers = {
                'User-Agent': 'StreamlitDSPExplorer/1.0 (Educational App; +https://streamlit.io)'
            }
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status() # Check for HTTP errors
            # -------------------------------
            
            original_pil = Image.open(BytesIO(response.content)).convert('RGB')
            img_arr = np.array(original_pil)
            
        except Exception as e:
            st.warning(f"⚠️ Could not load sample image (Connection Issue). Using fallback noise pattern.")
            # Fallback noise pattern if internet fails
            img_arr = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

    # 2. Controls
    c1, c2 = st.columns(2)
    with c1:
        # Added unique key='luma_slider'
        luma_res = st.select_slider(
            "💡 Luma (Brightness) Resolution",
            options=[100, 50, 25, 10, 5, 2],
            value=100,
            format_func=lambda x: f"{x}% (Structure)",
            key='luma_slider'
        )
    with c2:
        # Added unique key='chroma_slider'
        chroma_res = st.select_slider(
            "🎨 Chroma (Color) Resolution",
            options=[100, 50, 25, 10, 5, 2],
            value=100,
            format_func=lambda x: f"{x}% (Paint)",
            key='chroma_slider'
        )

    # 3. Processing Logic (Naive YUV Simulation)
    if img_arr is not None:
        img_float = img_arr.astype(float)
        R, G, B = img_float[:,:,0], img_float[:,:,1], img_float[:,:,2]
        
        # Calculate Luma
        Y = 0.299*R + 0.587*G + 0.114*B
        
        # Calculate Chroma (Difference)
        Cb = B - Y
        Cr = R - Y
        
        # Downsample Function
        def resample_channel(channel, pct):
            if pct == 100: return channel
            h, w = channel.shape
            # Downsample
            step = int(100/pct)
            if step < 1: step = 1
            small = channel[::step, ::step]
            
            # Upsample back to original size (Nearest Neighbor to show blocks)
            # Create indexing grid
            h_indices = np.arange(h) // step
            w_indices = np.arange(w) // step
            
            # Clip indices to prevent out-of-bounds due to integer division
            h_indices = np.clip(h_indices, 0, small.shape[0]-1)
            w_indices = np.clip(w_indices, 0, small.shape[1]-1)
            
            # Broadcast to reconstruct
            return small[h_indices[:, None], w_indices]

        # Apply degradation
        Y_mod = resample_channel(Y, luma_res)
        Cb_mod = resample_channel(Cb, chroma_res)
        Cr_mod = resample_channel(Cr, chroma_res)
        
        # Reconstruct RGB
        R_rec = Y_mod + Cr_mod
        B_rec = Y_mod + Cb_mod
        G_rec = (Y_mod - 0.299*R_rec - 0.114*B_rec) / 0.587
        
        # Clip and Stack
        final_img = np.stack([R_rec, G_rec, B_rec], axis=2)
        final_img = np.clip(final_img, 0, 255).astype(np.uint8)

        # 4. Display
        st.image(final_img, caption=f"Simulation: Luma {luma_res}% | Chroma {chroma_res}%", use_container_width=True)
        
        # 5. Dynamic Feedback
        if luma_res < 20:
            st.error("⚠️ **Structure Lost:** When you lower Luma, the image becomes unrecognizable. The eye relies on edges/contrast for object recognition.")
        elif chroma_res < 20:
            st.success("✅ **Bandwidth Hack:** Notice how you can lower Chroma to 10% or 5% and the image still looks 'sharp'? This is because your brain fills in the color gaps!")
        else:
            st.info("Try lowering the **Chroma** slider all the way down. You'll be surprised how much detail remains.")
def main():
    st.markdown('<h1 class="main-title">Vision, Sampling & Aliasing</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Why wheels go backward and movies look smooth</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🎡 The Wagon Wheel (Aliasing)", "👁️ The Human Eye", "📚 Theory & Diagrams"])

    with tab1:
        render_wagon_wheel_interactive()
        
    with tab2:
        render_human_eye_tab()
        
    """"""
    with tab3:
        st.markdown("### 📚 Theoretical Concepts & Diagrams")
        st.markdown("---")

        c1, c2 = st.columns(2, gap="large")
        
        with c1:
            st.markdown("#### 1. The Sampling Theorem (Nyquist)")
            st.markdown("""
            To capture a signal (or motion) accurately, your sampling rate ($f_s$) must be at least **twice** the frequency of the signal ($f_{max}$).
            """)
            st.latex(r"f_{sample} > 2 \cdot f_{signal}")
            
            st.warning("""
            **What happens if you fail?**
            If $f_{sample} < 2 \cdot f_{signal}$, the high-frequency signal is misinterpreted as a lower frequency. This is **Aliasing**.
            """)
            
            # 
            st.image(
                "https://upload.wikimedia.org/wikipedia/commons/f/f6/Aliasing-plot.png",
                caption="Aliasing: The blue dots (samples) make the high-frequency red wave look like the low-frequency blue wave.",
                use_container_width=True
            )
            st.caption("Notice how the blue dots (samples) on the high-frequency wave look like they form a much slower wave.")

        with c2:
            st.markdown("#### 2. The Stroboscopic Effect")
            st.markdown("""
            This is aliasing applied to rotation. When a wheel rotates at frequency $R$, and you capture it at frame rate $F$:
            """)
            
            st.markdown("""
            * **Sync ($R = F$):** The wheel turns exactly 360° between frames. It looks frozen.
            * **Forward ($R < F/2$):** The wheel turns slightly (e.g., 10°). Brain connects it correctly.
            * **Backward ($R > F$):** The wheel turns 350°. The brain thinks it moved **-10°** (backward) because it's the shorter path.
            """)
            
            # 
            st.image(
                "https://upload.wikimedia.org/wikipedia/commons/3/3e/WagonWheelEffect.gif",
                caption="""As the camera accelerates right, the objects first speed up sliding to the left.
                  At the halfway point, they suddenly appear to change direction but continue accelerating left, slowing down. """,
                use_container_width=True
            )
            st.caption("The brain always assumes the shortest path between two frames, causing the illusion of reverse rotation.")
        st.markdown("---")
        st.markdown("### 3. Spatial Aliasing (The Moiré Effect)")
        
        c3, c4 = st.columns([1, 1], gap="large")
        
        with c3:
            st.markdown("""
            **The Problem:**
            Just as time aliasing happens when **motion** is too fast for your frame rate, spatial aliasing happens when a **pattern** is too fine for your sensor's resolution.
            
            **The Result:**
            Ghostly, curved interference patterns appear that don't exist in the real world. This is why news anchors are told not to wear tight-striped shirts!
            """)
            
            st.info("The eye has a finite number of cones. If you look at a screen door from far away, the mesh might turn into weird wavy lines.")

        with c4:
            st.image(
                "https://upload.wikimedia.org/wikipedia/commons/c/cc/Moire.gif",
                caption="Moiré Effect:Moiré pattern created by overlapping two sets of concentric circles.",
                use_container_width=True
            )
        # --- FURTHER READING SECTION ---
        st.markdown("---")
        st.markdown("### 📖 Further Reading (Wikipedia)")
        
        st.markdown("""
        * **[Wagon-wheel effect](https://en.wikipedia.org/wiki/Wagon-wheel_effect)**: Detailed explanation of the illusion seen in movies and machinery.
        * **[Aliasing](https://en.wikipedia.org/wiki/Aliasing)**: The general mathematical principle behind these errors.
        * **[Human Eye](https://en.wikipedia.org/wiki/Photoreceptor_cell)**: The cells that allow us to see.
        * **[Chroma Subsampling]](https://en.wikipedia.org/wiki/Chroma_subsampling)**: The Image and video compression technique inspired by human vision.
        * **[Stroboscopic effect](https://en.wikipedia.org/wiki/Stroboscopic_effect)**: How flashing lights can freeze motion.
        * **[Moiré pattern](https://en.wikipedia.org/wiki/Moir%C3%A9_pattern)**: The spatial interference patterns shown above.
        """)
        
        
if __name__ == "__main__":
    main()