import streamlit as st
import tempfile
import os
import cv2
import numpy as np

# --- LOCAL MODULE IMPORTS ---
# We use the existing utils, but orchestrate them differently
from overlay_utils import PhysicsOverlay
import ai_utils 

# --- CONFIGURATION & CSS ---
st.set_page_config(
    page_title="PhysicsLens: Snapshot Analysis",
    page_icon="📸",
    layout="wide"
)

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    h1, h2, h3 { color: #f0f2f6; font-family: 'Inter', sans-serif; }
    .main-header {
        text-align: center; 
        font-weight: 800; 
        font-size: 2.5rem;
        background: -webkit-linear-gradient(0deg, #4facfe, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .step-card {
        background-color: #1a1c24;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #30333d;
        text-align: center;
    }
    div[data-testid="stImage"] { margin: auto; }
    </style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None
if "processed_data" not in st.session_state:
    st.session_state.processed_data = None
if "analysis_text" not in st.session_state:
    st.session_state.analysis_text = None

# --- HELPER FUNCTIONS ---

def extract_three_keyframes(video_path):
    """
    Efficiently seeks and extracts exactly 3 frames (10%, 50%, 90%).
    Returns a list of dicts: {'frame': rgb_array, 'timestamp': float, 'label': str}
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define indices for Start, Middle, End
    indices = [
        int(total_frames * 0.1),
        int(total_frames * 0.5),
        int(total_frames * 0.9)
    ]
    labels = ["Initial State", "Mid-Motion", "Final State"]
    
    keyframes = []
    
    for idx, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB immediately
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp = frame_idx / fps if fps > 0 else 0
            
            keyframes.append({
                'frame': frame_rgb, # RGB for AI/Display
                'frame_bgr': frame, # BGR for OpenCV drawing if needed
                'timestamp': timestamp,
                'label': labels[idx],
                'original_idx': idx
            })
            
    cap.release()
    return keyframes

def reset_analysis_state():
    """Clears previous analysis data."""
    st.session_state.processed_data = None
    st.session_state.analysis_text = None

# --- MAIN UI WORKFLOW ---

def render_sidebar():
    with st.sidebar:
        st.title("⚙️ Configuration")
        mode = st.selectbox(
            "Education Level",
            ["ELI5 (Basic)", "High School Physics", "Undergrad (Advanced)", "PhD (Quantum/Relativity)"],
            index=1
        )
        api_key = st.text_input("Google API Key", type="password")
        
        st.divider()
        st.info("📸 **Snapshot Mode Active**\n\nOptimized for low latency. Analyzes 3 distinct moments instead of full video.")
        return mode, api_key

def main():
    # 1. Header
    st.markdown('<div class="main-header">PhysicsLens: Snapshot Engine</div>', unsafe_allow_html=True)
    
    # 2. Config
    analysis_level, user_api_key = render_sidebar()
    
    # 3. Secure API Key
    # Priority: User Input -> Secrets
    api_key = user_api_key or st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.warning("⚠️ Please enter your Google API Key in the sidebar to proceed.")
        st.stop()
        
    ai_utils.configure_gemini(api_key)

    # 4. File Ingestion
    uploaded_file = st.file_uploader("Upload Experiment Video", type=['mp4', 'mov', 'avi'])
    
    if uploaded_file:
        # State Reset Logic
        if st.session_state.last_uploaded_file != uploaded_file.name:
            reset_analysis_state()
            st.session_state.last_uploaded_file = uploaded_file.name

        # Save to Temp
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.flush()
        
        # --- PHASE 1: IMMEDIATE FEEDBACK ---
        # Show raw video immediately so user knows it loaded
        st.caption("📽️ Raw Footage")
        st.video(tfile.name)
        
        # --- PHASE 2 & 3: PROCESSING (Only if not done) ---
        if st.session_state.processed_data is None:
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step A: Extraction
                status_text.markdown("**Step 1/3**: Extracting Keyframes (Start, Mid, End)...")
                keyframes = extract_three_keyframes(tfile.name)
                
                if not keyframes:
                    st.error("Could not extract frames. Video might be corrupt.")
                    st.stop()
                
                progress_bar.progress(33)
                
                # Step B: AI Analysis (Dual Mode)
                status_text.markdown("**Step 2/3**: Querying Gemini Vision Model (Vector + Concept)...")
                
                # B1. Vector Data (Batch)
                # We need BGR frames for the AI function we wrote earlier, or adapt it.
                # ai_utils.get_batch_physics_overlays expects BGR list.
                bgr_frames = [kf['frame_bgr'] for kf in keyframes]
                vector_data = ai_utils.get_batch_physics_overlays(bgr_frames)
                
                # B2. Text Explanation (Concept)
                # ai_utils.analyze_physics_with_gemini expects list of dicts with 'frame' key
                text_data = ai_utils.analyze_physics_with_gemini(keyframes, analysis_level=analysis_level)
                
                progress_bar.progress(66)
                
                # Step C: Rendering
                status_text.markdown("**Step 3/3**: Rendering Physics Overlays...")
                
                final_images = []
                for idx, kf in enumerate(keyframes):
                    # Copy the RGB frame for drawing
                    # Note: OpenCV drawing functions usually work on BGR, but since we are displaying
                    # in Streamlit (RGB), let's convert, draw, then display.
                    # Wait, overlay_utils uses OpenCV functions which expect BGR usually? 
                    # Let's check overlay_utils.draw_vector. It uses cv2.arrowedLine. 
                    # If we pass RGB, it draws RGB colors. That works fine as long as consistent.
                    
                    vis_frame = kf['frame'].copy() 
                    
                    # Find matching AI data
                    # ai_utils returns list with 'frame_index'
                    frame_vectors = next((item for item in vector_data if item.get("frame_index") == idx), None)
                    
                    # Fallback if list order matches
                    if not frame_vectors and idx < len(vector_data):
                        frame_vectors = vector_data[idx]
                        
                    if frame_vectors:
                        vis_frame = PhysicsOverlay.draw_ai_overlay(vis_frame, frame_vectors)
                    
                    final_images.append(vis_frame)
                
                # Save to State
                st.session_state.processed_data = final_images
                st.session_state.analysis_text = text_data
                st.session_state.keyframe_labels = [kf['label'] for kf in keyframes]
                
                progress_bar.progress(100)
                status_text.empty() # Clear status
                progress_bar.empty() # Clear bar
                
            except Exception as e:
                st.error(f"Processing Error: {e}")
                st.stop()
            finally:
                # Cleanup temp file
                # os.remove(tfile.name) # Keep it for the video player above
                pass

        # --- PHASE 4: VISUALIZATION & PRESENTATION ---
        
        if st.session_state.processed_data:
            st.divider()
            st.markdown("### 🔬 Snapshot Analysis")
            
            # Gallery View
            cols = st.columns(3)
            images = st.session_state.processed_data
            labels = st.session_state.keyframe_labels
            
            for i, col in enumerate(cols):
                with col:
                    st.image(images[i], use_container_width=True, caption=labels[i])

            # Educational Context
            txt_data = st.session_state.analysis_text
            
            if txt_data and not txt_data.get("error"):
                st.divider()
                c1, c2 = st.columns([1, 2])
                
                with c1:
                    st.markdown("#### 📊 Key Metrics")
                    st.success(f"**Subject:** {txt_data.get('main_object', 'Object')}")
                    st.info(f"**Principle:** {txt_data.get('physics_principle', 'Physics')}")
                    st.warning(f"**Motion:** {txt_data.get('velocity_estimation', 'Analyzed')}")
                    
                with c2:
                    st.markdown(f"#### 🎓 Expert Explanation ({analysis_level})")
                    st.markdown(f"> {txt_data.get('explanation', 'No explanation available.')}")
            else:
                st.error("Could not generate text explanation.")

if __name__ == "__main__":
    main()