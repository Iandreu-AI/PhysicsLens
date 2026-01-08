import streamlit as st
import tempfile
import os
import time

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="PhysicsLens AI",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (PRO POLISH) ---
# A little CSS to make the drag-and-drop area pop and clean up the UI
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
    Expert Tip: OpenCV needs a file path, not a BytesIO object. 
    We save the upload to a temp file to bridge Streamlit and OpenCV.
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
    # 1. Save file locally so OpenCV can read it later
    tfile_path = save_uploaded_file(uploaded_file)
    
    if tfile_path:
        # Layout: Split screen (Input vs Output placeholder)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Footage")
            st.video(uploaded_file)
            st.caption(f"Filename: {uploaded_file.name}")

        with col2:
            st.subheader("AI Analysis")
            
            # This is where your state management shines.
            # Only run analysis when clicked, not on every re-render.
            analyze_btn = st.button("✨ Analyze Physics", type="primary", use_container_width=True)
            
            if analyze_btn:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Mocking the processing pipeline for the demo
                status_text.text("Extracting keyframes with OpenCV...")
                time.sleep(1) 
                progress_bar.progress(33)
                
                status_text.text("Sending telemetry to Gemini 3...")
                time.sleep(1)
                progress_bar.progress(66)
                
                status_text.text("Rendering overlays...")
                time.sleep(1)
                progress_bar.progress(100)
                
                # Success State
                st.success("Analysis Complete!")
                
                # Placeholder for the Output Video
                # In the next step, we will replace this with the actual OpenCV output
                st.info("Overlay generation logic will be injected here.")
                
                # Mock result data
                with st.expander("📝 Generated Physics Report (Gemini 3)", expanded=True):
                    st.markdown(f"""
                    **detected_object**: Projectile
                    **estimated_velocity**: 14.5 m/s
                    **principle**: Newtonian Mechanics (Parabolic Trajectory)
                    
                    **Explanation ({analysis_mode}):**
                    The object follows a curved path due to gravity acting downwards while it maintains constant horizontal velocity.
                    """)
                    
    # Cleanup: Good hygiene to remove temp file after session (optional, depending on persistence needs)
    # os.remove(tfile_path)