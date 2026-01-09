import streamlit as st
import tempfile
import os
import cv2
import numpy as np

# --- LOCAL MODULE IMPORTS ---
# Ensure you have these files in the same directory:
# 1. video_utils.py
# 2. overlay_utils.py
# 3. track_utils.py
# 4. ai_utils.py
from video_utils import get_video_frames_generator
from overlay_utils import PhysicsOverlay
from track_utils import MotionTracker 

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
                api_key = st.secrets.get("GOOGLE_API_KEY")
                if not api_key:
                    st.error("❌ Google API Key is missing. Please set it in .streamlit/secrets.toml")
                    st.stop()
                
                # Configure AI (Import locally to avoid startup errors if keys are missing)
                try:
                    from ai_utils import configure_gemini, analyze_physics_with_gemini
                    if not configure_gemini(api_key):
                        st.stop()
                except ImportError:
                    st.error("❌ ai_utils.py is missing!")
                    st.stop()

                # --- STEP 1: FRAME EXTRACTION & TRACKING ---
                progress_bar = st.progress(0)
                status_text = st.empty()
                frame_placeholder = st.empty()
                
                processed_frames = []
                
                try:
                    # 1. Initialize Generator (Resize for performance)
                    generator = get_video_frames_generator(tfile_path, stride=2, resize_width=600)
                    
                    # 2. Initialize Tracker (ONCE, before loop)
                    tracker = MotionTracker()
                    
                    status_text.markdown("**1/2 🔄 Tracking Physics Object...**")
                    
                    # --- MAIN PROCESSING LOOP ---
                    for meta in generator:
                        clean_frame = meta['frame']
                        processed_frames.append(meta)
                        
                        # Create a copy for visualization so we don't draw on the AI's data
                        vis_frame = clean_frame.copy()
                        
                        # --- REAL CV TRACKING ---
                        center, velocity = tracker.process_frame(vis_frame)
                        
                        if show_vectors:
                            # Center Calculation
                            h, w = vis_frame.shape[:2]
                            display_center = center if center else (w//2, h//2)
                            
                            # A. Draw Gravity (Static Downward Force)
                            PhysicsOverlay.draw_vector(
                                vis_frame,
                                start_point=display_center,
                                vector=(0, 40), 
                                label="Mg", 
                                color=PhysicsOverlay.COLOR_FORCE
                            )
                            
                            # B. Draw Real-Time Velocity
                            vx, vy = velocity
                            # Threshold to remove noise (only draw if moving)
                            if abs(vx) > 1 or abs(vy) > 1:
                                PhysicsOverlay.draw_vector(
                                    vis_frame,
                                    start_point=display_center,
                                    vector=(vx, vy),
                                    label=f"v={np.sqrt(vx**2+vy**2):.1f}",
                                    color=PhysicsOverlay.COLOR_VELOCITY,
                                    scale=5.0 # Scale up for visibility
                                )

                            # C. Draw HUD
                            PhysicsOverlay.draw_hud(vis_frame, {
                                "Frame": meta['frame_id'],
                                "Status": "Tracking" if center else "Scanning...",
                                "Velocity X": f"{vx:.2f}",
                                "Velocity Y": f"{vy:.2f}"
                            })

                        # Update UI
                        current = meta['frame_id']
                        total = meta['total_frames']
                        pct = min(int((current / total) * 50), 100)
                        progress_bar.progress(pct)
                        frame_placeholder.image(vis_frame, channels="RGB", caption=f"Analyzing Frame {current}")
                    
                    # --- STEP 2: AI ANALYSIS ---
                    status_text.markdown("**2/2 🧠 Gemini is analyzing physics...**")
                    progress_bar.progress(75)

                    # --- STEP 3: VISUAL FREE BODY DIAGRAM (The "Wow" Factor) ---
                    st.divider()
                    st.subheader("🤖 AI-Generated Free Body Diagram")
                    st.markdown("Gemini is now calculating exact vector coordinates for a keyframe...")
                    
                    # 1. Pick a "Key Frame" (e.g., middle of the video)
                    # We saved processed_frames list earlier
                    if len(processed_frames) > 0:
                        mid_index = len(processed_frames) // 2
                        key_frame_meta = processed_frames[mid_index]
                        key_frame_bgr = key_frame_meta['original_frame_bgr'] # Use original for best AI clarity
                        
                        # 2. Call the new AI function
                        # Make sure to import it at top: from ai_utils import get_physics_overlay_coordinates
                        from ai_utils import get_physics_overlay_coordinates
                        
                        ai_coords = get_physics_overlay_coordinates(key_frame_bgr)
                        
                        if ai_coords:
                            # 3. Draw on the frame using the AI's coordinates
                            fbd_frame = key_frame_bgr.copy()
                            fbd_frame = PhysicsOverlay.draw_ai_overlay(fbd_frame, ai_coords)
                            
                            # 4. Display Side-by-Side
                            c1, c2 = st.columns(2)
                            with c1:
                                st.image(key_frame_meta['frame'], caption="Raw Frame", use_container_width=True)
                            with c2:
                                # Convert BGR to RGB for Streamlit
                                fbd_rgb = cv2.cvtColor(fbd_frame, cv2.COLOR_BGR2RGB)
                                st.image(fbd_rgb, caption="Gemini-Guided Vector Overlay", use_container_width=True)
                                
                            with st.expander("See Coordinate Data"):
                                st.json(ai_coords)
                        else:
                            st.warning("Could not generate coordinate overlay.")
                    
                    # Call the AI module
                    ai_result = analyze_physics_with_gemini(processed_frames, analysis_level=analysis_mode)
                    
                    progress_bar.progress(100)
                    status_text.success("✅ Analysis Complete!")
                    
                    # --- DISPLAY RESULTS ---
                    st.divider()
                    
                    # --- BUG FIX START ---
                    # Check if result is None BEFORE checking for "error" key
                    if ai_result is None:
                        st.error("🚨 AI Error: The model returned no response.")
                        st.warning(" Troubleshooting:")
                        st.markdown("""
                        1. Check your **Internet Connection**.
                        2. Verify `GOOGLE_API_KEY` in `.streamlit/secrets.toml`.
                        3. You might have hit the **Free Tier Quota** (Wait 60s and try again).
                        """)
                    
                    elif "error" in ai_result:
                        st.error("AI Analysis Failed with message:")
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
                    # --- BUG FIX END ---

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    
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
                    # Uncomment for deep debugging:
                    # import traceback
                    # st.text(traceback.format_exc())
                
                finally:
                    # Clean up the temp file
                    if os.path.exists(tfile_path):
                        os.remove(tfile_path)