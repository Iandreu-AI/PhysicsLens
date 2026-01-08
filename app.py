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
                # Setup UI Containers for real-time feedback
                progress_bar = st.progress(0)
                status_text = st.empty()
                frame_placeholder = st.empty() # Placeholder for the video preview
                
                try:
                    # Initialize Generator
                    # Stride=5: Process every 5th frame for speed
                    # Resize=400: Smaller images for faster UI rendering
                    generator = get_video_frames_generator(tfile_path, stride=5, resize_width=400)
                    
                    processed_frames = []
                    status_text.markdown("**🔄 Processing Video Frames...**")
                    
                    # Real-time Processing Loop
                    for meta in generator:
                        frame_rgb = meta['frame']
                        current_frame = meta['frame_id']
                        total = meta['total_frames']
                        
                        # Update Progress Bar
                        percentage = int((current_frame / total) * 100) if total > 0 else 0
                        progress_bar.progress(min(percentage, 100))
                        
                        # Display "Live" Preview
                        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True, caption=f"Frame {current_frame}")
                        
                        # Store metadata (we will send this to Gemini later)
                        processed_frames.append(meta)
                        
                    # Finalize UI
                    progress_bar.progress(100)
                    status_text.success(f"✅ Extracted {len(processed_frames)} Keyframes for Analysis")
                    
                    # Show Result Summary
                    with st.expander("📊 Analysis Data", expanded=True):
                        st.write(f"**Total Frames Scanned:** {len(processed_frames)}")
                        if processed_frames:
                            st.write(f"**Video Duration:** {processed_frames[-1]['timestamp']:.2f}s")
                        st.info("Ready to send frames to Gemini 3 API in Phase 4.")

                except Exception as e:
                    st.error(f"Analysis Failed: {e}")
                
                finally:
                    # Clean up the temp file to save disk space
                    if os.path.exists(tfile_path):
                        os.remove(tfile_path)