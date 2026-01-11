import streamlit as st
import tempfile
import os
import cv2
import numpy as np
import time

# --- LOCAL MODULE IMPORTS ---
from ai_utils import get_batch_physics_overlays
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

def play_live_inference(video_path):
    """
    Plays the video with real-time CV overlays instead of raw footage.
    """
    cap = cv2.VideoCapture(video_path)
    tracker = MotionTracker()
    
    st_frame = st.empty()
    st_info = st.empty()
    
    # Processing stride (play every Nth frame to keep up with UI)
    stride = 2 
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % stride == 0:
            # 1. Resize for UI performance
            frame = cv2.resize(frame, (640, 360))
            
            # 2. Track Motion
            center, velocity = tracker.process_frame(frame)
            
            # 3. Apply Overlays (Using your overlay_utils.py)
            h, w = frame.shape[:2]
            display_center = center if center else (w//2, h//2)
            
            # Gravity (Static)
            PhysicsOverlay.draw_vector(frame, display_center, (0, 30), "Mg", PhysicsOverlay.COLOR_FORCE)
            
            # Velocity (Dynamic)
            vx, vy = velocity
            if abs(vx) > 1 or abs(vy) > 1:
                PhysicsOverlay.draw_vector(frame, display_center, (vx, vy), "Vel", PhysicsOverlay.COLOR_VELOCITY, scale=4.0)
            
            # HUD
            PhysicsOverlay.draw_hud(frame, {"Status": "Tracking", "Vx": f"{vx:.1f}", "Vy": f"{vy:.1f}"})

            # 4. Display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st_frame.image(frame_rgb, channels="RGB", use_container_width=True)
            
        frame_count += 1

    cap.release()
    st_info.caption("Preview complete. Analysis below.")

def extract_and_sample_frames(video_path, num_samples=3):
    """
    Extracts frames and picks exactly 3 distinct moments: Start, Middle, End.
    """
    generator = get_video_frames_generator(video_path, stride=5, resize_width=400)
    tracker = MotionTracker()
    all_frames = []
    
    for meta in generator:
        clean_frame = meta['frame'].copy()
        center, velocity = tracker.process_frame(clean_frame)
        meta['center'] = center
        meta['velocity'] = velocity
        all_frames.append(meta)
        
    total = len(all_frames)
    if total == 0: return [], []
    
    idx_start = 0
    idx_mid = total // 2
    idx_end = int(total * 0.90) 
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
        st.info("System Ready: Live Inference Mode")
        return mode

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

    with st.status(" Processing Video ...", expanded=True) as status:
        
        status.write(" Sampling Keyframes...")
        sampled_frames, all_frames_context = extract_and_sample_frames(tfile_path, num_samples=3)
        
        if not sampled_frames:
            status.update(label="❌ Error reading video", state="error")
            return

        status.write("🧠 Generating Conceptual Model...")
        step = max(1, len(all_frames_context) // 10)
        context_subset = all_frames_context[::step][:10]
        ai_text_result = analyze_physics_with_gemini(context_subset, analysis_level=analysis_mode)
        
        status.write(f"Calculating Vectors for {len(sampled_frames)} keyframes...")
        # Prepare list of raw BGR frames for the batch function
        raw_frames_for_ai = [kf['original_frame_bgr'] for kf in sampled_frames]
        
        # Single API Call
        batch_coords = get_batch_physics_overlays(raw_frames_for_ai)
        
        final_keyframes = []
        progress_bar = st.progress(0)
        total_kfs = len(sampled_frames)
        
        # 3. Merge Results
        for idx, kf in enumerate(sampled_frames):
            vis_frame = kf['frame'].copy()
            
            # Find matching data in batch response (safely)
            ai_data = next((item for item in batch_coords if item.get("frame_index") == idx), None)
            
            # Fallback if AI missed an index (rare but possible)
            if not ai_data and idx < len(batch_coords):
                ai_data = batch_coords[idx]

            if ai_data:
                vis_frame = PhysicsOverlay.draw_ai_overlay(vis_frame, ai_data)
                vectors_found = [v.get('name', 'Force') for v in ai_data.get('vectors', [])]
                vector_str = ", ".join(vectors_found[:2])
                kf['caption'] = f"Forces: {vector_str}"
            else:
                # CV Fallback if AI fails specific frame
                center = kf.get('center')
                vel = kf.get('velocity', (0,0))
                if center:
                    vis_frame = PhysicsOverlay.draw_vector(vis_frame, center, vel, "Motion", PhysicsOverlay.COLOR_VELOCITY, scale=3.0)
                kf['caption'] = "Motion Tracking (CV Fallback)"
            
            kf['processed_image'] = vis_frame
            final_keyframes.append(kf)
            progress_bar.progress((idx + 1) / total_kfs)

        status.update(label="✅ Analysis Complete", state="complete", expanded=False)
        st.session_state.analysis_results = ai_text_result
        st.session_state.keyframes = final_keyframes
        st.session_state.processing_complete = True

def render_results(analysis_mode, tfile_path):
    """
    Updated render function that handles full video generation and playback.
    """
    result = st.session_state.analysis_results
    
    # 1. Metrics Row
    st.divider()
    m1, m2, m3 = st.columns(3)
    if result:
        m1.metric("Object", result.get("main_object", "Detected Object"))
        m2.metric("Principle", result.get("physics_principle", "Dynamics"))
        m3.metric("State", "Analysis Complete")
    
    # 2. The Video Player (The Hero Component)
    st.subheader("▶️ Analysis Replay")
    
    # Define output path
    output_video_path = tfile_path.replace(".mp4", "_processed.mp4")
    
    # Check if we need to generate (or if it's already cached)
    if not os.path.exists(output_video_path):
        with st.spinner("Rendering final physics engine output..."):
            from track_utils import MotionTracker
            from video_utils import generate_annotated_video
            
            # New tracker instance for the clean render pass
            render_tracker = MotionTracker()
            
            # Use the AI data we stored in session state to inform the renderer
            # (In a full app, we pass the specific batch_coords here)
            ai_context = st.session_state.get('batch_coords', [])
            
            success = generate_annotated_video(tfile_path, output_video_path, render_tracker, ai_context)
            
            if not success:
                st.error("Failed to render video.")
                return

    # 3. Display Video & Download Button
    if os.path.exists(output_video_path):
        # Display the video
        st.video(output_video_path)
        
        # Read file for download button
        with open(output_video_path, "rb") as file:
            btn = st.download_button(
                label="📥 Download Analysis (.mp4)",
                data=file,
                file_name="PhysicsLens_Analysis.mp4",
                mime="video/mp4"
            )
    
    # 4. The Explanation (Below the video)
    st.divider()
    st.subheader(f"📝 Expert Commentary ({analysis_mode})")
    if result:
        st.info(result.get("explanation", "Analysis available."))

# --- MAIN EXECUTION ---
def main():
    analysis_mode = render_sidebar()
    render_header()
    
    uploaded_file = st.file_uploader("Upload footage to begin analysis...", type=['mp4', 'mov', 'avi'])
    
    if uploaded_file:
        if st.session_state.current_file_name != uploaded_file.name:
            reset_state()
            st.session_state.current_file_name = uploaded_file.name
            
        tfile_path = save_uploaded_file(uploaded_file)
        
        # --- NEW: Live Inference Player instead of Raw Video ---
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.caption("Live Physics Preview (CV Tracking)")
            # This plays the video WITH overlays immediately
            play_live_inference(tfile_path)
        
        api_key = st.secrets.get("GOOGLE_API_KEY")
        if not api_key:
            st.warning("⚠️ Google API Key missing.")
            st.stop()
            
        if not st.session_state.processing_complete:
            process_pipeline(tfile_path, api_key, analysis_mode)
            
        if st.session_state.processing_complete:
            render_results(analysis_mode)
            
        if os.path.exists(tfile_path):
            os.remove(tfile_path)

if __name__ == "__main__":
    main()