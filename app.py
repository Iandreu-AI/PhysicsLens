import streamlit as st
import tempfile
import os
import cv2
import numpy as np
import time

# --- LOCAL MODULE IMPORTS ---
from video_utils import get_video_frames_generator
from overlay_utils import PhysicsOverlay
from track_utils import MotionTracker 

# --- CONFIGURATION & CUSTOM CSS ---
st.set_page_config(
    page_title="PhysicsLens AI",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Global Theme */
    .stApp { background-color: #0e1117; }
    
    /* Headers */
    h1, h2, h3 { color: #f0f2f6; font-family: 'Inter', sans-serif; }
    .main-header {
        font-size: 3rem;
        background: -webkit-linear-gradient(45deg, #4facfe, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-text { text-align: center; color: #a0a0a0; margin-bottom: 2rem; }
    
    /* Metrics Styling */
    div[data-testid="stMetricValue"] { font-size: 1.5rem; color: #4facfe; }
    
    /* Gallery Card Styling */
    .caption-text { color: #b0b0b0; font-size: 0.9rem; margin-top: 10px; font-weight: 500; text-align: center; }
    
    /* Hide Default Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "keyframes" not in st.session_state:
    st.session_state.keyframes = []
if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None

# --- HELPER FUNCTIONS ---

def reset_state():
    """Resets the session state when a new file is uploaded."""
    st.session_state.processing_complete = False
    st.session_state.analysis_results = None
    st.session_state.keyframes = []

def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error handling file: {e}")
        return None

def extract_and_sample_frames(video_path, num_samples=3):
    """
    Extracts frames and picks exactly 3 distinct moments: Start, Middle, End.
    """
    # Optimized stride for speed
    generator = get_video_frames_generator(video_path, stride=5, resize_width=400)
    tracker = MotionTracker()
    
    all_frames = []
    
    # 1. Process video for tracking data
    for meta in generator:
        clean_frame = meta['frame'].copy()
        center, velocity = tracker.process_frame(clean_frame)
        
        meta['center'] = center
        meta['velocity'] = velocity
        all_frames.append(meta)
        
    # 2. Sample 3 Keyframes (Start, Middle, End)
    total = len(all_frames)
    if total == 0: return [], []
    
    # Explicitly pick 0% (Start), 50% (Peak Action), 90% (Result)
    # We avoid 100% to avoid fade-outs/black screens
    idx_start = 0
    idx_mid = total // 2
    idx_end = int(total * 0.90) 
    
    # Ensure indices are unique if video is very short
    indices = sorted(list(set([idx_start, idx_mid, idx_end])))
    
    sampled_frames = [all_frames[i] for i in indices]
    
    return sampled_frames, all_frames

# --- UI COMPONENT FUNCTIONS ---

def render_sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/physics.png", width=64)
        st.markdown("### Analysis Config")
        
        mode = st.selectbox(
            "Complexity Level",
            ["ELI5 (Basic)", "High School Physics", "Undergrad (Advanced)", "PhD (Quantum/Relativity)"]
        )
        
        st.markdown("---")
        show_raw = st.toggle("Show Raw Video Player", value=False)
        
        st.info("System Ready: High-Speed Mode")
        return mode, show_raw

def render_header():
    st.markdown('<div class="main-header">PhysicsLens AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-text">Automated Physics Extraction & Visual Reasoning Engine</div>', unsafe_allow_html=True)

def process_pipeline(tfile_path, api_key, analysis_mode):
    try:
        from ai_utils import configure_gemini, analyze_physics_with_gemini, get_physics_overlay_coordinates
        if not configure_gemini(api_key):
            st.error("Invalid API Key")
            st.stop()
    except ImportError:
        st.error("Missing ai_utils.py")
        st.stop()

    with st.status("🚀 Initializing Physics Engine...", expanded=True) as status:
        
        # 1. Computer Vision Pass
        status.write("📷 Scanning video structure...")
        sampled_frames, all_frames_context = extract_and_sample_frames(tfile_path, num_samples=3)
        
        if not sampled_frames:
            status.update(label="❌ Error reading video", state="error")
            return

        # 2. Semantic Analysis
        status.write("🧠 Generating Conceptual Physics Model...")
        step = max(1, len(all_frames_context) // 10)
        context_subset = all_frames_context[::step][:10]
        ai_text_result = analyze_physics_with_gemini(context_subset, analysis_level=analysis_mode)
        
        # 3. Visual Reasoning (The Heavy Lifting)
        status.write(f"📐 Calculating Force Vectors for {len(sampled_frames)} keyframes...")
        
        final_keyframes = []
        progress_bar = st.progress(0)
        total_kfs = len(sampled_frames)
        
        for idx, kf in enumerate(sampled_frames):
            vis_frame = kf['frame'].copy()
            
            # AI Call for High Fidelity Overlay
            coords = get_physics_overlay_coordinates(kf['original_frame_bgr'])
            
            if coords:
                # Apply high-fidelity AI overlay
                vis_frame = PhysicsOverlay.draw_ai_overlay(vis_frame, coords)
                
                # Create a smart caption from the vectors found
                vectors_found = [v.get('name', 'Force') for v in coords.get('vectors', [])]
                vector_str = ", ".join(vectors_found[:2]) # Top 2 forces
                kf['caption'] = f"Forces: {vector_str}"
            else:
                # Fallback to CV if AI times out
                center = kf.get('center')
                vel = kf.get('velocity', (0,0))
                if center:
                    vis_frame = PhysicsOverlay.draw_vector(vis_frame, center, vel, "Motion", PhysicsOverlay.COLOR_VELOCITY, scale=3.0)
                kf['caption'] = "Motion Tracking"
            
            kf['processed_image'] = vis_frame
            final_keyframes.append(kf)
            
            # Update progress bar
            progress_bar.progress((idx + 1) / total_kfs)

        status.update(label="✅ Analysis Complete", state="complete", expanded=False)
        
        st.session_state.analysis_results = ai_text_result
        st.session_state.keyframes = final_keyframes
        st.session_state.processing_complete = True

def render_results(analysis_mode):
    result = st.session_state.analysis_results
    keyframes = st.session_state.keyframes
    
    if not result:
        st.error("No results found. Please re-analyze.")
        return

    # --- 1. Top Level Metrics ---
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Object Detected", result.get("main_object", "Unknown"))
    m2.metric("Primary Principle", result.get("physics_principle", "Analysis Pending"))
    m3.metric("Est. Velocity", result.get("velocity_estimation", "Calculating..."))
    
    # --- 2. Keyframe Gallery (3 distinct frames) ---
    st.subheader("🎞️ Physics Timeline")
    
    # Single row of 3 images
    cols = st.columns(3)
    
    labels = ["Initial State", "Mid-Motion", "Final State"]
    
    for idx, col in enumerate(cols):
        if idx < len(keyframes):
            kf = keyframes[idx]
            label = labels[idx] if idx < 3 else "Keyframe"
            
            with col:
                st.image(kf['processed_image'], use_container_width=True, channels="RGB")
                st.markdown(f"<div class='caption-text'><strong>{label}</strong> (t={kf['timestamp']:.2f}s)<br>{kf.get('caption', '')}</div>", unsafe_allow_html=True)

    # --- 3. Deep Dive Summary ---
    st.divider()
    st.subheader(f"📝 Expert Analysis ({analysis_mode})")
    
    if result.get("error"):
        st.error(f"Analysis Error: {result.get('explanation')}")
    else:
        st.info(result.get("explanation"))

# --- MAIN EXECUTION ---
def main():
    analysis_mode, show_raw_video = render_sidebar()
    render_header()
    
    uploaded_file = st.file_uploader("Upload footage to begin analysis...", type=['mp4', 'mov', 'avi'])
    
    if uploaded_file:
        if st.session_state.current_file_name != uploaded_file.name:
            reset_state()
            st.session_state.current_file_name = uploaded_file.name
            
        tfile_path = save_uploaded_file(uploaded_file)
        
        if show_raw_video:
            st.video(uploaded_file)
        
        api_key = st.secrets.get("GOOGLE_API_KEY")
        if not api_key:
            st.warning("⚠️ Google API Key missing. Add it to .streamlit/secrets.toml to proceed.")
            st.stop()
            
        if not st.session_state.processing_complete:
            process_pipeline(tfile_path, api_key, analysis_mode)
            
        if st.session_state.processing_complete:
            render_results(analysis_mode)
            
        if os.path.exists(tfile_path):
            os.remove(tfile_path)

if __name__ == "__main__":
    main()