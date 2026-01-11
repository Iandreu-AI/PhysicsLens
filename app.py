import streamlit as st
import tempfile
import os
import cv2
import numpy as np

# --- LOCAL MODULE IMPORTS ---
from overlay_utils import PhysicsOverlay
import ai_utils 
from track_utils import MotionTracker
from video_utils import generate_annotated_video

# --- CONFIGURATION & CSS ---
st.set_page_config(
    page_title="PhysicsLens Analysis",
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
    .metric-card {
        background-color: #1a1c24;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #4facfe;
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
if "output_video_path" not in st.session_state:
    st.session_state.output_video_path = None

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
            # Convert BGR to RGB immediately for display/AI
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp = frame_idx / fps if fps > 0 else 0
            
            keyframes.append({
                'frame': frame_rgb,       # RGB for AI/Display
                'frame_bgr': frame,       # BGR for OpenCV drawing
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
    st.session_state.output_video_path = None

# --- MAIN UI WORKFLOW ---

def render_sidebar():
    with st.sidebar:
        st.title("⚙️ Configuration")
        mode = st.selectbox(
            "Education Level",
            ["ELI5 (Basic)", "High School Physics", "Undergrad (Advanced)", "PhD (Quantum/Relativity)"],
            index=1
        )        
        st.divider()
        st.info("💡 Tip: Ensure the video has a clear moving object against a stable background.")
        return mode

def main():
    # 1. Header
    st.markdown('<div class="main-header">PhysicsLens: Snapshot Engine</div>', unsafe_allow_html=True)
    
    # 2. Config
    analysis_level = render_sidebar()
    
    # 3. Secure API Key (Server-Side Only)
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except FileNotFoundError:
        st.error("⚠️ Server Error: API Key not configured. Please contact the administrator.")
        st.stop()
    except KeyError:
        st.error("⚠️ Server Error: 'GOOGLE_API_KEY' not found in secrets.")
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
        st.caption("📽️ Raw Footage")
        c_left, c_center, c_right = st.columns([1, 2, 1])
        with c_center:
            st.video(tfile.name)
        
        # --- PHASE 2 & 3: PROCESSING ---
        if st.session_state.processed_data is None:
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step A: Extraction
                status_text.markdown("**Step 1/4**: Extracting Keyframes...")
                keyframes = extract_three_keyframes(tfile.name)
                
                if not keyframes:
                    st.error("Could not extract frames. Video might be corrupt.")
                    st.stop()
                
                progress_bar.progress(25)
                
                # Step B: AI Analysis (Snapshot)
                status_text.markdown("**Step 2/4**: Querying Gemini Vision Model...")
                bgr_frames = [kf['frame_bgr'] for kf in keyframes]
                
                # Get Vector Data (Gravity, Normal Force positions)
                vector_data = ai_utils.get_batch_physics_overlays(bgr_frames)
                
                # Get Text Explanation
                text_data = ai_utils.analyze_physics_with_gemini(keyframes, analysis_level=analysis_level)
                
                progress_bar.progress(50)
                
                # Step C: Motion Tracking & Rendering (Full Video)
                status_text.markdown("**Step 3/4**: Rendering High-Performance Overlay...")
                
                # We use a distinct file name to ensure Streamlit doesn't cache an old version
                output_filename = f"tracked_{uploaded_file.name}"
                output_path = os.path.join(tempfile.gettempdir(), output_filename)
                
                # Initialize a fresh tracker
                tracker = MotionTracker()
                
                # Import the new processor
                from video_utils import generate_annotated_video
                
                # EXECUTE PIPELINE
                # We pass the AI data merely for context (gravity extraction), 
                # but we do NOT call the API again.
                success, final_path = generate_annotated_video(tfile.name, output_path, tracker, vector_data)
        

                if success:
                    st.session_state.output_video_path = final_path
                else:
                    st.error(f"Video rendering failed: {final_path}")

                progress_bar.progress(75)

                # Step D: Rendering Snapshot Gallery
                status_text.markdown("**Step 4/4**: Finalizing Gallery...")
                
                final_images = []
                for idx, kf in enumerate(keyframes):
                    vis_frame = kf['frame'].copy() 
                    
                    # Robust AI Data Matching
                    frame_vectors = None
                    # Try to match by index if available
                    if vector_data and isinstance(vector_data, list):
                        frame_vectors = next((item for item in vector_data if item.get("frame_index") == idx), None)
                        # Fallback: just take the i-th element
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
                status_text.empty() 
                progress_bar.empty()
                
            except Exception as e:
                st.error(f"Processing Error: {e}")
                # Print stack trace for debugging
                import traceback
                st.code(traceback.format_exc())
                st.stop()

        # --- PHASE 4: VISUALIZATION & PRESENTATION ---
        
        # 1. Snapshot Gallery
        if st.session_state.processed_data:
            st.divider()
            st.markdown("### 🔬 Snapshot Analysis")
            
            cols = st.columns(3)
            images = st.session_state.processed_data
            labels = st.session_state.keyframe_labels
            
            for i, col in enumerate(cols):
                with col:
                    st.image(images[i], use_container_width=True, caption=labels[i])

            # 2. Educational Context
            txt_data = st.session_state.analysis_text
            
            if txt_data and not txt_data.get("error"):
                st.divider()
                c1, c2 = st.columns([1, 2])
                
                with c1:
                    st.markdown("#### 📊 Key Metrics")
                    st.markdown(f"""
                    <div class="metric-card">
                        <b>Subject:</b> {txt_data.get('main_object', 'Object')}<br>
                        <b>Principle:</b> {txt_data.get('physics_principle', 'Physics')}<br>
                        <b>Motion:</b> {txt_data.get('velocity_estimation', 'Analyzed')}
                    </div>
                    """, unsafe_allow_html=True)
                    
                with c2:
                    st.markdown(f"#### 🎓 Expert Explanation ({analysis_level})")
                    st.info(f"{txt_data.get('explanation', 'No explanation available.')}")
            else:
                error_details = txt_data.get("error") if txt_data else "Unknown Data Error"
                st.error(f"⚠️ Analysis Text Failed: {error_details}")

            # 3. Full Motion Video
            st.divider()
            st.markdown("### 🎥 Full Motion Analysis")
            st.caption("Real-time tracking.")
            
            if st.session_state.output_video_path and os.path.exists(st.session_state.output_video_path):
                try:
                    # Fix: Open as binary and read bytes into memory
                    # This ensures Streamlit receives the actual data, not just a fragile path
                    with open(st.session_state.output_video_path, 'rb') as video_file:
                        video_bytes = video_file.read()
                    
                    if len(video_bytes) > 0:
                        st.video(video_bytes, format="video/mp4")
                    else:
                        st.error("⚠️ Generated video file is empty.")
                        
                except Exception as e:
                    st.error(f"Error reading video file: {e}")
            else:
                st.warning("Video tracking data unavailable.")

if __name__ == "__main__":
    main()