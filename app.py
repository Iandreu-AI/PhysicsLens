import streamlit as st
import tempfile
import os
import cv2  # Required for image display logic
from video_utils import get_video_frames_generator  # Import our new module

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
                    generator = get_video_frames_generator(tfile_path, stride=5, resize_width=400)
                    
                    processed_frames = []
                    status_text.markdown("**1/2 🔄 Processing Video Frames...**")
                    
                    for meta in generator:
                        processed_frames.append(meta)
                        
                        # Update UI
                        current = meta['frame_id']
                        total = meta['total_frames']
                        pct = int((current / total) * 50) # First 50% of progress bar
                        progress_bar.progress(pct)
                        frame_placeholder.image(meta['frame'], channels="RGB", use_column_width=True, caption=f"Frame {current}")

                    # --- STEP 2: AI ANALYSIS ---
                    status_text.markdown("**2/2 🧠 Gemini is analyzing physics...**")
                    progress_bar.progress(75)
                    
                    # Call our new AI module
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
                
                finally:
                    if os.path.exists(tfile_path):
                        os.remove(tfile_path)