import streamlit as st
import tempfile
import os
import cv2  # Required for image display logic
from video_utils import get_video_frames_generator  # Import our new module
from overlay_utils import PhysicsOverlay  # Import our overlay module
import numpy as np

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="PhysicsLens AI",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
    }
    .main-header {
        font-size: 2.5rem;
        color: #4facfe;
        text-align: center;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-text {
        text-align: center;
        color: #b0b0b0;
        margin-bottom: 2rem;
    }
    /* Highlight the uploader */
    div[data-testid="stFileUploader"] {
        border: 2px dashed #4facfe;
        border-radius: 10px;
        padding: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- UTILITY FUNCTIONS ---
def save_uploaded_file(uploaded_file):
    """
    Saves the uploaded file to a temporary location so OpenCV can read it.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error handling file: {e}")
        return None

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/physics.png", width=80)
    st.title("PhysicsLens Config")
    
    st.markdown("### ⚙️ Analysis Settings")
    analysis_mode = st.selectbox(
        "Complexity Level",
        ["ELI5 (Basic)", "High School Physics", "Undergrad (Advanced)", "PhD (Quantum/Relativity)"]
    )
    
    show_vectors = st.checkbox("Show Force Vectors", value=True)
    show_trajectories = st.checkbox("Show Trajectories", value=True)
    
    st.divider()
    st.info(f"Powered by **Gemini 3** & **OpenCV**")

# --- MAIN LAYOUT ---
st.markdown('<div class="main-header">🔭 PhysicsLens AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Upload a video to analyze motion, forces, and physics principles in real-time.</div>', unsafe_allow_html=True)

# --- FILE UPLOAD SECTION ---
uploaded_file = st.file_uploader("Drop your physics video here", type=['mp4', 'mov', 'avi'])

if uploaded_file is not None:
    # 1. Save file locally so OpenCV can read it
    tfile_path = save_uploaded_file(uploaded_file)
    
    if tfile_path:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Footage")
            st.video(uploaded_file)
            st.caption(f"Filename: {uploaded_file.name}")

        with col2:
            st.subheader("AI Analysis")
            
            # State Management: Only run analysis when clicked
            analyze_btn = st.button("✨ Analyze Physics", type="primary", use_container_width=True)
            
            if analyze_btn:
                # --- CHECK API KEY ---
                # We check secrets first, or allow user to input manually (good for demos)
                api_key = st.secrets.get("GOOGLE_API_KEY")
                if not api_key:
                    st.error("❌ Google API Key is missing. Please set it in .streamlit/secrets.toml")
                    st.stop()
                
                # Configure AI
                from ai_utils import configure_gemini, analyze_physics_with_gemini
                if not configure_gemini(api_key):
                    st.stop()

                # --- STEP 1: FRAME EXTRACTION ---
                progress_bar = st.progress(0)
                status_text = st.empty()
                frame_placeholder = st.empty()
                
                try:
                    # Initialize Generator
                    # Note: We resize to 600 for better visibility of vectors
                    generator = get_video_frames_generator(tfile_path, stride=5, resize_width=600)
                    
                    processed_frames = []
                    status_text.markdown("**1/2 🔄 Processing Video Frames & Trajectories...**")
                    
                    # Simulation variables for the demo effect
                    sim_x = 100
                    sim_y = 100
                    
                    for meta in generator:
                        # 1. Keep the clean frame for AI analysis
                        clean_frame = meta['frame']
                        processed_frames.append(meta)
                        
                        # 2. Create a copy for visualization (So we don't confuse the AI later)
                        vis_frame = clean_frame.copy()
                        
                        # 3. APPLY OVERLAYS (Demo Mode)
                        if show_vectors:
                            # Calculate center of frame
                            h, w = vis_frame.shape[:2]
                            center_x, center_y = w // 2, h // 2
                            
                            # SIMULATION: Draw a Gravity Vector (Always down)
                            PhysicsOverlay.draw_vector(
                                vis_frame,
                                start_point=(center_x, center_y),
                                vector=(0, 50), # 50px down
                                label="Mg (Gravity)",
                                color=PhysicsOverlay.COLOR_FORCE,
                                scale=1.0
                            )
                            
                            # SIMULATION: Draw a Dynamic Velocity Vector
                            # Uses frame_id to rotate the vector to show it's "alive"
                            t = meta['frame_id'] * 0.1
                            vel_x = 40 * np.cos(t)
                            vel_y = 40 * np.sin(t)
                            
                            PhysicsOverlay.draw_vector(
                                vis_frame,
                                start_point=(center_x, center_y),
                                vector=(vel_x, vel_y),
                                label="Velocity",
                                color=PhysicsOverlay.COLOR_VELOCITY,
                                scale=1.5
                            )
                            
                            # Add HUD
                            PhysicsOverlay.draw_hud(vis_frame, {
                                "Frame": meta['frame_id'],
                                "Time": f"{meta['timestamp']:.2f}s",
                                "Est. Speed": f"{abs(vel_x*2):.1f} m/s"
                            })

                        # 4. Update UI with the VISUALIZED frame
                        current = meta['frame_id']
                        total = meta['total_frames']
                        pct = min(int((current / total) * 50), 100)
                        progress_bar.progress(pct)
                        
                        # Streamlit displays the frame with vectors
                        frame_placeholder.image(vis_frame, channels="RGB", caption=f"Processing Frame {current}")
                        
                    
                 # --- STEP 2: AI ANALYSIS ---
                    status_text.markdown("**2/2 🧠 Gemini is analyzing physics...**")
                    progress_bar.progress(75)
                    
                    # Call the AI module
                    # Ensure ai_utils.py exists in your folder!
                    ai_result = analyze_physics_with_gemini(processed_frames, analysis_level=analysis_mode)
                    
                    progress_bar.progress(100)
                    status_text.success("✅ Analysis Complete!")
                    
                    # --- DISPLAY RESULTS ---
                    st.divider()
                    
                    # Handle Errors gracefully
                    if "error" in ai_result:
                        st.error("AI Analysis Failed")
                        st.json(ai_result)
                    else:
                        # Pretty Display of Physics Data
                        r_col1, r_col2 = st.columns([1, 2])
                        
                        with r_col1:
                            st.metric("Detected Object", ai_result.get("main_object", "Unknown"))
                            st.metric("Est. Velocity", ai_result.get("velocity_estimation", "N/A"))
                            st.metric("Principle", ai_result.get("physics_principle", "N/A"))
                        
                        with r_col2:
                            st.info(f"**AI Explanation ({analysis_mode}):**\n\n{ai_result.get('explanation', 'No explanation provided.')}")
                        
                        # Show raw JSON for debugging (Collapsed)
                        with st.expander("🛠️ View Raw Gemini Response (JSON)"):
                            st.json(ai_result)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    # Optional: Print traceback for debugging
                    # import traceback
                    # st.text(traceback.format_exc())
                
                finally:
                    # Clean up the temp file
                    if os.path.exists(tfile_path):
                        os.remove(tfile_path)