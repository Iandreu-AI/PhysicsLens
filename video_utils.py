import cv2
import time
import numpy as np
import streamlit as st
import shutil
import subprocess
import os

# ==========================================
# 1. FFMPEG & COMPRESSION UTILITIES
# ==========================================

def is_ffmpeg_installed():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False

def convert_to_h264(input_path, output_path):
    """
    Converts a video to H.264 format using FFmpeg for browser compatibility.
    """
    try:
        # -y: overwrite
        # -preset ultrafast: quick encoding for UX
        # -pix_fmt yuv420p: ensures browser support
        command = [
            "ffmpeg", "-y", "-i", input_path, "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "23", "-preset", "ultrafast", "-movflags", "+faststart", output_path
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception as e:
        # Fallback: just copy if FFmpeg fails, though browser might not play it
        if input_path != output_path: 
            shutil.copy2(input_path, output_path)
        return False

def save_frames_to_video(frames, output_path, fps=30.0):
    """
    Helper: Writes a list of OpenCV frames to a temporary MP4 file, 
    then compresses it for the web.
    """
    if not frames:
        return False

    height, width, layers = frames[0].shape
    
    # Write raw video
    temp_path = output_path.replace(".mp4", "_raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    out.release()

    # Convert/Compress
    if is_ffmpeg_installed():
        convert_to_h264(temp_path, output_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)
    else:
        shutil.move(temp_path, output_path)
        
    return True

# ==========================================
# 2. WEBCAM UI ENGINE (RECORDER ONLY)
# ==========================================

class WebcamRecorder:
    def __init__(self):
        # Initialize session state for the recorder state machine
        if 'recorder_state' not in st.session_state:
            st.session_state.recorder_state = 'preview' 
        if 'recorded_frames' not in st.session_state:
            st.session_state.recorded_frames = []

    def _get_fresh_camera(self):
        """
        Attempts to open a NEW camera connection.
        Includes wait-and-retry logic for hardware locking issues.
        """
        max_retries = 3
        for i in range(max_retries):
            # Try Default Backend (Index 0)
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                # Test read to ensure it's not a zombie handle
                ret, _ = cap.read()
                if ret:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    return cap
                else:
                    cap.release()
            
            # If failed, wait before retry (Hardware Cooldown)
            time.sleep(0.5)
            
        return None

    def _draw_overlay(self, frame, text, color=(0, 255, 0), scale=1.0):
        """Draws text with a black outline for visibility."""
        h, w, _ = frame.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, scale, 2)[0]
        cx, cy = (w - text_size[0]) // 2, (h + text_size[1]) // 2
        
        # Outline
        cv2.putText(frame, text, (cx + 2, cy + 2), font, scale, (0, 0, 0), 3)
        # Text
        cv2.putText(frame, text, (cx, cy), font, scale, color, 2)
        return frame

    def run(self):
        """
        Main execution loop for the Streamlit UI.
        Returns: list of frames if recording is accepted, else None.
        """
        state = st.session_state.recorder_state

        # --- VIEW 1: PREVIEW ---
        if state == 'preview':
            st.info("Step 1: Frame your shot.")
            
            # Callback to switch state
            def go_to_record():
                st.session_state.recorder_state = 'recording'
            
            st.button("Start Recording", type="primary", use_container_width=True, on_click=go_to_record)

            frame_slot = st.empty()
            
            # Open Camera Local Scope
            cap = self._get_fresh_camera()
            
            if not cap:
                st.error("Camera is busy or not found. Please check permissions or refresh.")
                return None

            try:
                # Run loop until state changes (button click triggers rerun)
                while st.session_state.recorder_state == 'preview':
                    ret, frame = cap.read()
                    if not ret: break
                    
                    frame = self._draw_overlay(frame, "PREVIEW", (255, 255, 0))
                    # Convert BGR to RGB for Streamlit
                    frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    time.sleep(0.03) # Cap at ~30 FPS
            finally:
                # CRITICAL: Release camera before the script reruns!
                cap.release()
                # EXTRA SAFETY: Give Windows DirectShow time to unlock the resource
                time.sleep(0.5)

        # --- VIEW 2: RECORDING ---
        elif state == 'recording':
            frame_slot = st.empty()
            st.warning("Initializing Camera...")
            
            # Wait a moment for the hardware to free up from Preview mode
            time.sleep(0.5)
            
            cap = self._get_fresh_camera()
            
            if not cap:
                st.error("Camera failed to restart. Hardware locked.")
                if st.button("Try Again"):
                    st.session_state.recorder_state = 'preview'
                    st.rerun()
                return None

            frames = []
            try:
                # 1. Countdown (3 seconds)
                start = time.time()
                while time.time() - start < 3.0:
                    ret, frame = cap.read()
                    if not ret: break
                    rem = 3 - int(time.time() - start)
                    frame = self._draw_overlay(frame, str(rem), (0, 255, 255), 4.0)
                    frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    time.sleep(0.01)

                # 2. Record (5 seconds)
                start = time.time()
                while time.time() - start < 5.0:
                    ret, frame = cap.read()
                    if not ret: break
                    frames.append(frame)
                    elapsed = time.time() - start
                    
                    vis = frame.copy()
                    # Blinking red dot
                    if int(elapsed * 2) % 2 == 0:
                        cv2.circle(vis, (30, 30), 10, (0, 0, 255), -1)
                    vis = self._draw_overlay(vis, f"REC {elapsed:.1f}", (0, 0, 255))
                    frame_slot.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

                # Success
                st.session_state.recorded_frames = frames
                st.session_state.recorder_state = 'done'
                
            except Exception as e:
                st.error(f"Recording Error: {e}")
            finally:
                cap.release()
                time.sleep(0.2)
                st.rerun()

        # --- VIEW 3: DONE / REVIEW ---
        elif state == 'done':
            st.success(f"Recorded {len(st.session_state.recorded_frames)} frames.")
            
            # Show the last frame as a static preview
            if st.session_state.recorded_frames:
                last_frame = st.session_state.recorded_frames[-1]
                st.image(cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB), caption="Last Frame", use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                # Return frames to app for processing
                if st.button("✅ Use Video", type="primary", use_container_width=True):
                    frames = st.session_state.recorded_frames
                    # Reset state for next time
                    st.session_state.recorder_state = 'preview'
                    st.session_state.recorded_frames = []
                    return frames
            with c2:
                # Discard and try again
                if st.button("🔄 Retake", use_container_width=True):
                    st.session_state.recorder_state = 'preview'
                    st.session_state.recorded_frames = []
                    st.rerun()

        return None