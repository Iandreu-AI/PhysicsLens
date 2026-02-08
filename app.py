import streamlit as st
import tempfile
import os
import cv2
import numpy as np
import time
import base64

# --- LOCAL MODULE IMPORTS ---
from overlay_utils import PhysicsOverlay
from frame_optimizer import optimize_frames
import ai_utils
from ai_utils import (
    get_physics_vectors,
    draw_vectors_with_debug,
    CoordinateTransformer
)
from track_utils import MotionTracker

# --- CONFIGURATION & CSS ---
st.set_page_config(
    page_title="PhysicsLens",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* PRESET: Minimalist Light Mode with Serif */
    @import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Inter:wght@400;500;600&display=swap');

    /* BASE BACKGROUND */
    .stApp {
        background-color: #FAF9F6; /* Baby Powder / Off-white */
        color: #1A1A1A;
        font-family: 'Inter', sans-serif;
    }

    /* TYPOGRAPHY */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Lora', serif !important;
        color: #1A1A1A !important;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    p, div, span, label, button, .stButton, .stSelectbox, .stFileUploader {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* OVERRIDES FOR MOCKUP ACCURACY */
    
    /* SIDEBAR SELECTBOX (Dark Theme) */
    div[data-testid="stSidebar"] div[data-testid="stSelectbox"] > div > div {
        background-color: #1E293B;
        color: white;
        border: 1px solid #334155;
    }
    div[data-testid="stSidebar"] div[data-testid="stSelectbox"] label {
        color: #64748B !important;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    div[data-testid="stSidebar"] div[data-testid="stSelectbox"] svg {
        fill: white !important;
    }

    .main-header {
        text-align: center;
        margin-bottom: 3rem;
        padding-top: 2rem;
    }
    
    .hero-title {
        font-family: 'Lora', serif !important;
        font-size: 5rem !important;
        font-weight: 500;
        color: #1A1A1A;
        margin-bottom: 0.5rem;
    }

    .hero-subtitle {
        font-family: 'Inter', sans-serif !important;
        font-size: 1.2rem;
        color: #475569;
        font-weight: 400;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #F1F5F9; /* Very light cool grey */
        border-right: 1px solid #E2E8F0;
    }
    
    /* CUSTOM CARDS */
    .feature-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); /* Soft shadow */
    }

    /* BUTTONS */
    /* BUTTONS as ACTION CARDS */
    .stButton > button {
        background-color: #1E293B; /* Dark Slate */
        color: white !important;
        border: none;
        padding: 2rem 1.5rem; /* Large padding for card feeling */
        font-size: 1.1rem !important;
        height: auto;
        width: 100%;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
    }
    .stButton > button:hover {
        background-color: #0F172A;
        transform: translateY(-2px);
    }

    /* WEBCAM BUTTON SPECIFIC */
    /* We will use a hack to identify the webcam button if possible, but for now the global button style fits the reference's "Record" button */
    
    /* OR DIVIDER */
    .or-divider {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        font-family: 'Lora', serif;
        font-size: 1.5rem;
        color: #1E293B;
        padding-top: 3rem; /* Visual alignment with cards */
    }

    /* FILE UPLOADER - Clean White Card */
    [data-testid="stFileUploader"] {
        background: #FFFFFF;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    /* The internal dropzone */
    [data-testid="stFileUploader"] section {
        background-color: #F8FAFC;
        border: 2px dashed #CBD5E1;
        border-radius: 8px;
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }
    
    /* Browse Files Button Styling */
    [data-testid="stFileUploader"] button {
        background-color: #0F172A !important; /* Extremely Dark Blue/Black */
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.5rem 1.2rem !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    [data-testid="stFileUploader"] button:hover {
        background-color: #1E293B !important;
        transform: scale(1.02);
    }
    
    /* Customize the drag-and-drop text */
    [data-testid="stFileUploader"] section > div > div > span {
        font-size: 0.9rem !important;
        color: #1E293B !important;
        white-space: nowrap !important;
    }
    
    /* Hide default "Drag and drop" text and show custom */
    [data-testid="stFileUploader"] section label span {
        visibility: hidden;
        position: relative;
    }
    
    [data-testid="stFileUploader"] section label span::before {
        content: "Upload video files, types allowed (mp4, mov, avi)";
        visibility: visible;
        position: absolute;
        left: 0;
        font-size: 0.9rem;
        color: #334155;
        white-space: nowrap;
    }
    
    /* Hide the small instructions if possible or style them */
    [data-testid="stFileUploader"] small {
        display: none;
    }
    
    /* Sidebar "Settings" fake card styling helper */
    .sidebar-card {
        border: 2px solid #334155;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 2rem;
        background: transparent;
    }
    
    /* TABS STYLING - Black text with Orange active */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #1A1A1A !important;
        font-weight: 500;
        border-bottom: 2px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        color: #F97316 !important; /* Orange */
        border-bottom-color: #F97316 !important;
    }
    
    /* CLEAN BUTTON - Orange styling */
    .clean-button button {
        background-color: #F97316 !important; /* Orange */
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 6px -1px rgba(249, 115, 22, 0.3) !important;
    }
    .clean-button button:hover {
        background-color: #EA580C !important;
        transform: translateY(-1px) !important;
    }
    
    /* SIDEBAR HISTORY BUTTONS */
    div[data-testid="stSidebar"] .stButton > button[key^="history_"] {
        background-color: #F8FAFC !important;
        color: #1E293B !important;
        border: 1px solid #E2E8F0 !important;
        padding: 0.5rem !important;
        font-size: 0.9rem !important;
        text-align: left !important;
    }
    div[data-testid="stSidebar"] .stButton > button[key^="history_"]:hover {
        background-color: #E2E8F0 !important;
    }
    
    /* CHAT MESSAGE STYLING - Black text */
    [data-testid="stChatMessageContent"] {
        color: #000000 !important;
    }
    [data-testid="stChatMessageContent"] p {
        color: #000000 !important;
    }
    
    /* CHAT AVATARS - Make them bigger */
    [data-testid="stChatMessage"] img {
        width: 60px !important;
        height: 60px !important;
        border-radius: 50%;
    }
    
    </style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
# New history-based structure
if "history" not in st.session_state:
    st.session_state.history = []
if "current_analysis" not in st.session_state:
    st.session_state.current_analysis = None

# Legacy state variables (for backward compatibility during transition)
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None
if "active_video_path" not in st.session_state:
    st.session_state.active_video_path = None
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False

# Helper function to get current analysis data
def get_current_data(key, default=None):
    """Get data from current_analysis or return default."""
    if st.session_state.current_analysis:
        return st.session_state.current_analysis.get(key, default)
    return default


def reset_analysis_state():
    """Reset current analysis without losing history."""
    st.session_state.current_analysis = None
    st.session_state.active_video_path = None

def save_to_history(analysis_data):
    """Save completed analysis to history."""
    import datetime
    
    # Generate unique ID
    analysis_id = f"analysis_{int(datetime.datetime.now().timestamp())}"
    
    # Extract phenomena name from analysis
    phenomena_name = analysis_data.get('analysis_text', {}).get('main_object', 'Unknown Phenomenon')
    
    history_item = {
        'id': analysis_id,
        'timestamp': datetime.datetime.now(),
        'phenomena_name': phenomena_name,
        'video_path': st.session_state.active_video_path,
        'processed_data': analysis_data.get('processed_data'),
        'analysis_text': analysis_data.get('analysis_text'),
        'output_video_path': analysis_data.get('output_video_path'),
        'vector_data': analysis_data.get('vector_data'),
        'chat_history': analysis_data.get('chat_history', [])
    }
    
    # Add to history (newest first)
    st.session_state.history.insert(0, history_item)
    
    # Limit history to 10 items
    if len(st.session_state.history) > 10:
        st.session_state.history = st.session_state.history[:10]

# --- MAIN UI WORKFLOW ---

def render_sidebar():
    with st.sidebar:
        # Logo - Top Left
        # Logo - Top Left
        logo_file = "logo.png"
        if os.path.exists(logo_file):
            with open(logo_file, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            
            st.markdown(
                f"""
                <div style="margin-bottom: 2rem; text-align: center;">
                    <img src="data:image/png;base64,{data}" style="width: 100%; max-width: 300px; transform: scale(1.5); transform-origin: center;">
                </div>
                """,
                unsafe_allow_html=True
            )
            
            
        # Settings (Just Label + Selectbox)
        st.markdown('<p style="font-weight: 500; margin-bottom: -10px;">Select your level</p>', unsafe_allow_html=True)
        mode = st.selectbox(
            "Target Audience",
            ["ELI5 (Basic)", "High School Physics", "Undergrad (Advanced)", "PhD (Quantum/Relativity)"],
            index=1,
            label_visibility="collapsed"
        )
        
        # History Section
        if st.session_state.history:
            st.markdown("---")
            st.markdown('<p style="font-weight: 600; margin-bottom: 0.5rem; color: #1E293B;">Analysis History</p>', unsafe_allow_html=True)
            
            for idx, item in enumerate(st.session_state.history):
                # Format timestamp
                time_str = item['timestamp'].strftime("%H:%M")
                
                # Create button for each history item
                button_label = f"{item['phenomena_name']}"
                
                # Use a unique key for each button
                if st.button(button_label, key=f"history_{item['id']}", use_container_width=True):
                    # Load this analysis
                    st.session_state.current_analysis = item
                    st.rerun()
                
                # Show timestamp below button
                st.markdown(f"<div style='font-size:0.75rem; color:#64748B; margin-top:-0.5rem; margin-bottom:0.5rem;'>{time_str}</div>", unsafe_allow_html=True)
        
        return mode

def render_physics_tutor(chat_history, vector_data, analysis_text):
    """
    Renders the AI Physics Tutor chat interface with enlarged, custom avatars.
    """
    # --- 1. CSS OVERRIDES FOR AVATAR SCALING ---
    st.markdown("""
        <style>
        /* Force the avatar container to be larger */
        [data-testid="stChatMessageAvatar"] {
            width: 3.5rem !important;
            height: 3.5rem !important;
            background-color: transparent !important;
            border: none !important;
        }

        /* Target the actual image tag to fill the container and shape it */
        [data-testid="stChatMessageAvatar"] img {
            width: 3.5rem !important;
            height: 3.5rem !important;
            border-radius: 50% !important;
            object-fit: cover !important;
        }
        
        /* Optional: Add spacing adjustments for the larger avatars */
        [data-testid="stChatMessageContent"] {
            padding-left: 1rem !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.subheader("💬 AI Physics Tutor")

    # --- 2. ASSET LOADING ---
    # Define paths for custom avatars
    user_img_path = "user.png"
    assistant_img_path = "logo.png"

    # Load User Avatar (Fallback to None if file missing)
    user_avatar = user_img_path if os.path.exists(user_img_path) else None
    
    # Load Assistant Avatar
    assistant_avatar = assistant_img_path if os.path.exists(assistant_img_path) else None

    # --- 3. CHAT HISTORY RENDER LOOP ---
    chat_container = st.container()
    
    with chat_container:
        for message in chat_history:
            role = message["role"]
            content = message["content"]
            
            # Apply specific avatar based on role
            if role == "user":
                avatar_icon = user_avatar
            else:
                avatar_icon = assistant_avatar
            
            with st.chat_message(role, avatar=avatar_icon):
                st.markdown(content)

    # --- 4. INPUT HANDLING & RESPONSE GENERATION ---
    if prompt := st.chat_input("Ask a follow-up question..."):
        
        # Immediately render the user's new message
        with st.chat_message("user", avatar=user_avatar):
            st.markdown(prompt)

        # Render Assistant Response with Streaming
        with st.chat_message("assistant", avatar=assistant_avatar):
            response_container = st.empty()
            full_response = ""
            
            try:
                # Retrieve streaming response from AI utility
                stream = ai_utils.get_chat_response_stream(
                    chat_history, 
                    prompt, 
                    vector_data,
                    analysis_text
                )
                
                for chunk in stream:
                    full_response += chunk
                    response_container.markdown(full_response + "▌")
                
                # Final render without cursor
                response_container.markdown(full_response)
                
                return prompt, full_response

            except Exception as e:
                error_msg = "I'm having trouble connecting to the physics engine right now."
                response_container.error(f"{error_msg} ({str(e)})")
                return prompt, error_msg

    return None, None

def main():
    # 1. Header
    st.markdown("""
        <div class="main-header">
            <div class="hero-title">PhysicsLens</div>
            <div class="hero-subtitle">Automated Video Analysis & Kinematics Engine</div>
        </div>
    """, unsafe_allow_html=True)
    
    # 2. Config
    difficulty = render_sidebar()
    
    # 3. Secure API Key
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        ai_utils.configure_gemini(api_key)

    # 4. Source Selection (Layout: Upload | OR | Webcam)
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_up, col_or, col_cam = st.columns([5, 1, 5])
    
    with col_up:
        uploaded_file = st.file_uploader("Upload video files", type=['mp4', 'mov', 'avi'])
        st.markdown("<div style='color:#94a3b8; font-size:0.85rem; margin-top:0.2rem; margin-left: 0.5rem;'></div>", unsafe_allow_html=True)
    
    with col_or:
        st.markdown("<div class='or-divider'>OR</div>", unsafe_allow_html=True)
        
    with col_cam:
        # Spacer to align with upload box visual center roughly if needed, or just let flex handle it
        st.write("") # slight spacing
        if st.button("📹  Record from your webcam", use_container_width=True):
            st.session_state.is_recording = True
    
    # --- RECORDING OVERLAY ---
    if st.session_state.is_recording:
        st.info("Webcam Active")
        recorder = WebcamRecorder()
        frames = recorder.run() # Handles its own UI loop
        
        if frames:
            # Save recorded frames to temp video
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            height, width, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            out = cv2.VideoWriter(tfile.name, fourcc, 30.0, (width, height))
            for f in frames:
                out.write(f)
            out.release()
            
            reset_analysis_state() # 1. Clear old data first
            st.session_state.active_video_path = tfile.name # 2. Then set the new video path
            st.session_state.is_recording = False
            st.rerun()
            
    # Handle File Upload
    elif uploaded_file:
        if st.session_state.last_uploaded_file != uploaded_file.name:
            reset_analysis_state()
            st.session_state.last_uploaded_file = uploaded_file.name
            
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            tfile.flush()
            st.session_state.active_video_path = tfile.name
    
    # --- ANALYSIS FLOW ---
    if st.session_state.active_video_path:
        video_path = st.session_state.active_video_path
        
        st.markdown("___")
        # Centered Layout: Spacer, Content(Video+Card), Spacer
        # Actually, let's put Video on top (smaller) and Card below or side.
        # User requested: "Make video smaller. And below it add the progress bar"
        
        # We will use 3 columns to center the video
        c_left, c_center, c_right = st.columns([1, 2, 1])
        with c_center:
             st.video(video_path)
        
        # Progress Bar & Controls BELOW the video
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Control Row (Button + Status)
        c_ctl_left, c_ctl_mid, c_ctl_right = st.columns([1, 2, 1])
        with c_ctl_mid:
             if not st.session_state.current_analysis:
                # Initial State: Show Button
                if st.button("🚀 Start Analysis", use_container_width=True):
                    st.session_state.processing_active = True
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Stage 1: Extract keyframes
                        status_text.markdown("<span style='color:#64748B'>*Scanning for best physics frames...*</span>", unsafe_allow_html=True)
                        keyframes = optimize_frames(video_path)
                        progress_bar.progress(25)
                        
                        if not keyframes:
                            st.error("Extraction failed.")
                            st.stop()
                        
                        # Stage 2: AI Analysis
                        status_text.markdown("*Querying Gemini Vision...*")
                        text_data = ai_utils.analyze_physics_with_gemini(keyframes,difficulty=difficulty)
                        progress_bar.progress(40)
                        
                        # Stage 3: Vector Generation (NEW HYBRID SYSTEM)
                        status_text.markdown("*Generating physics vectors with hybrid system...*")
                        vector_data_raw = []
                        bgr_frames = [kf['frame_bgr'] for kf in keyframes]
                        
                        prev_frame = None
                        for idx, frame_bgr in enumerate(bgr_frames):
                            # Use the new hybrid system
                            vector_data = get_physics_vectors(frame_bgr, prev_frame, idx)
                            vector_data_raw.append(vector_data)
                            prev_frame = frame_bgr
                            
                            # Update progress incrementally
                            progress = 40 + int((idx / len(bgr_frames)) * 20)  # 40-60%
                            progress_bar.progress(progress)
                        
                        st.session_state.vector_data = vector_data_raw
                        progress_bar.progress(60)
                        
                        # Stage 4: Create snapshot images with fixed vectors
                        status_text.markdown("*Finalizing visualization with fixed vectors...*")
                        final_images = []
                        for idx, kf in enumerate(keyframes):
                            bgr_frame = kf['frame_bgr'].copy()
                            
                            if vector_data_raw and idx < len(vector_data_raw):
                                # Use the new drawing function
                                show_debug = st.session_state.get('show_debug_markers', True)
                                vis_frame = draw_vectors_with_debug(
                                    bgr_frame, 
                                    vector_data_raw[idx], 
                                    show_debug=show_debug
                                )
                                vis_frame_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
                            else:
                                vis_frame_rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                            
                            final_images.append(vis_frame_rgb)
                            
                            progress = 85 + int((idx / len(keyframes)) * 10)
                            progress_bar.progress(progress)
                        
                        # Create analysis data structure
                        analysis_data = {
                            'processed_data': final_images,
                            'analysis_text': text_data,
                            'output_video_path': st.session_state.get('output_video_path'),
                            'vector_data': vector_data_raw,
                            'chat_history': []
                        }
                        
                        # Save to history and set as current
                        save_to_history(analysis_data)
                        st.session_state.current_analysis = st.session_state.history[0]
                        
                        progress_bar.progress(100)
                        status_text.markdown("✅ *Analysis complete!*")
                        time.sleep(0.5)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Runtime Error: {e}")
                        import traceback
                        st.error(traceback.format_exc())
                        st.stop()
        
        # Clean Button - Show if analysis exists
        if st.session_state.current_analysis:
            st.markdown("<br>", unsafe_allow_html=True)
            c_clean_left, c_clean_mid, c_clean_right = st.columns([1, 2, 1])
            with c_clean_mid:
                st.markdown('<div class="clean-button">', unsafe_allow_html=True)
                if st.button("🧹 Clean - Start New Analysis", use_container_width=True, key="clean_btn"):
                    reset_analysis_state()
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

        # --- DASHBOARD (Reordered) ---
        if st.session_state.current_analysis:
            st.markdown("___")
            st.markdown("## 📊 Analysis Dashboard")
            
            # REORDERED TABS: 1. Snapshot, 2. Metrics + Chat, 3. Motion
            tab1, tab2, tab3 = st.tabs(["Snapshot Gallery", "Metrics & Report", "Motion Tracking"])
            
            txt_data = get_current_data('analysis_text', {})

            # TAB 1: SNAPSHOT GALLERY
            with tab1:
                cols = st.columns(3)
                images = get_current_data('processed_data', [])
                labels = ["Initial State", "Mid-Motion", "Final State"]
                for i, col in enumerate(cols):
                    with col:
                        if i < len(images):
                            st.image(images[i], caption=labels[i])
                            
            # TAB 2: METRICS & REPORT (+ CHAT)
            with tab2:
                # Top Row: Metrics
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f"""
                    <div class="feature-card">
                        <div class="metric-label" style="color:#64748B; font-size:0.9rem; text-transform:uppercase; letter-spacing:0.05em;">Subject</div>
                        <div class="metric-value" style="font-size:1.5rem; font-weight:600; color:#0F172A;">{txt_data.get('main_object', 'N/A')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div class="feature-card">
                        <div class="metric-label" style="color:#64748B; font-size:0.9rem; text-transform:uppercase; letter-spacing:0.05em;">Motion Type</div>
                        <div class="metric-value" style="font-size:1.5rem; font-weight:600; color:#0F172A;">{txt_data.get('motion_type', 'N/A')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""
                    <div class="feature-card">
                        <div class="metric-label" style="color:#64748B; font-size:0.9rem; text-transform:uppercase; letter-spacing:0.05em;">Physics Principle</div>
                        <div class="metric-value" style="font-size:1.5rem; font-weight:600; color:#0F172A;">{txt_data.get('physics_principle', 'N/A')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                c_text, c_eqn = st.columns([2, 1])
                with c_text:
                    st.markdown("""<div class="feature-card"><h3>🎓 Expert Explanation</h3>""", unsafe_allow_html=True)
                    st.write(txt_data.get('explanation', 'No details available.'))
                    st.markdown("</div>", unsafe_allow_html=True)
                with c_eqn:
                    st.markdown("""<div class="feature-card"><h3>🧮 Governing Equation</h3>""", unsafe_allow_html=True)
                    if txt_data.get('key_formula'):
                        st.latex(txt_data['key_formula'])
                    else:
                        st.caption("No equation detected.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                # CHAT SECTION (At Bottom of Tab 2)
                st.markdown("___")
                # 1. Call the specialized renderer
                # It handles the UI, CSS injection, and Streaming internally
                new_user_msg, new_ai_msg = render_physics_tutor(
                    chat_history=get_current_data('chat_history', []),
                    vector_data=get_current_data('vector_data'),
                    analysis_text=get_current_data('analysis_text')
                )

                # 2. Update Session State if a new exchange occurred
                if new_user_msg and new_ai_msg:
                    # Append new messages to the current analysis history
                    current_history = st.session_state.current_analysis['chat_history']
                    current_history.append({"role": "user", "content": new_user_msg})
                    current_history.append({"role": "assistant", "content": new_ai_msg})
                    
                    # Force a rerun to ensure the history is saved and the input box clears properly
                    st.rerun()

            # TAB 3: MOTION TRACKING
            with tab3:
                output_video = get_current_data('output_video_path')
                if output_video and os.path.exists(output_video):
                    try:
                        with open(output_video, 'rb') as vf:
                             st.video(vf.read())
                    except:
                        st.info("Video unavailable.")
                else:
                    st.info("Tracking video processing skipped or failed.")

if __name__ == "__main__":
    main()