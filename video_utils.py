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
    try:
        command = [
            "ffmpeg", "-y", "-i", input_path, "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "23", "-preset", "ultrafast", "-movflags", "+faststart", output_path
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception as e:
        if input_path != output_path: shutil.copy2(input_path, output_path)
        return False

# ==========================================
# 2. STANDARD TRACKING
# ==========================================

from overlay_utils import PhysicsOverlay

def generate_annotated_video(input_path, output_path, tracker, ai_metadata=None):
    if not os.path.exists(input_path): return False, None
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened(): return False, None

    raw_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = raw_width - (raw_width % 2)
    height = raw_height - (raw_height % 2)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    temp_output_path = output_path.replace(".mp4", "_temp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
    
    tracker.mode = "DETECT"
    draw_gravity = True if (ai_metadata and isinstance(ai_metadata, list)) else False

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if frame.shape[:2] != (height, width): frame = cv2.resize(frame, (width, height))
            
            center, velocity = tracker.process_frame(frame)
            if center:
                vx, vy = velocity
                if abs(vx) > 0.5 or abs(vy) > 0.5:
                    PhysicsOverlay.draw_vector(frame, center, (vx, vy), label="v", color=PhysicsOverlay.COLOR_VELOCITY, scale=4.0)
                if draw_gravity:
                    PhysicsOverlay.draw_vector(frame, center, (0, 50), label="Fg", color=PhysicsOverlay.COLOR_GRAVITY, scale=1.0)
                cv2.circle(frame, center, 5, (0, 255, 255), -1)
                speed = np.sqrt(vx**2 + vy**2)
                PhysicsOverlay.draw_hud(frame, {"Speed": f"{speed:.1f}", "State": tracker.mode})
            else:
                PhysicsOverlay.draw_smart_label(frame, "Scanning...", (30, height - 40))
            out.write(frame)
    finally:
        cap.release()
        out.release()

    if is_ffmpeg_installed(): convert_to_h264(temp_output_path, output_path)
    else: shutil.move(temp_output_path, output_path)
    if os.path.exists(temp_output_path): os.remove(temp_output_path)
    return True, output_path

# ==========================================
# 3. WEBCAM UI ENGINE (C++ CRASH FIX)
# ==========================================

class WebcamRecorder:
    def __init__(self):
        if 'recorder_state' not in st.session_state:
            st.session_state.recorder_state = 'preview' 
        if 'recorded_frames' not in st.session_state:
            st.session_state.recorded_frames = []

    def _get_fresh_camera(self):
        """
        Attempts to open a NEW camera connection.
        Includes wait-and-retry logic for hardware locking.
        """
        max_retries = 3
        for i in range(max_retries):
            # Try Default Backend
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
        h, w, _ = frame.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, scale, 2)[0]
        cx, cy = (w - text_size[0]) // 2, (h + text_size[1]) // 2
        cv2.putText(frame, text, (cx + 2, cy + 2), font, scale, (0, 0, 0), 3)
        cv2.putText(frame, text, (cx, cy), font, scale, color, 2)
        return frame

    def run(self):
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
                st.error("Camera is busy. Please refresh the page.")
                return None

            try:
                # Run loop until state changes
                while st.session_state.recorder_state == 'preview':
                    ret, frame = cap.read()
                    if not ret: break
                    
                    frame = self._draw_overlay(frame, "PREVIEW", (255, 255, 0))
                    frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    time.sleep(0.03)
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
                # 1. Countdown
                start = time.time()
                while time.time() - start < 3.0:
                    ret, frame = cap.read()
                    if not ret: break
                    rem = 3 - int(time.time() - start)
                    frame = self._draw_overlay(frame, str(rem), (0, 255, 255), 4.0)
                    frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    time.sleep(0.01)

                # 2. Record
                start = time.time()
                while time.time() - start < 5.0:
                    ret, frame = cap.read()
                    if not ret: break
                    frames.append(frame)
                    elapsed = time.time() - start
                    
                    vis = frame.copy()
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

        # --- VIEW 3: DONE ---
        elif state == 'done':
            st.success(f"Recorded {len(st.session_state.recorded_frames)} frames.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Process", type="primary", use_container_width=True):
                    frames = st.session_state.recorded_frames
                    st.session_state.recorder_state = 'preview'
                    st.session_state.recorded_frames = []
                    return frames
            with c2:
                if st.button("🔄 Retake", use_container_width=True):
                    st.session_state.recorder_state = 'preview'
                    st.session_state.recorded_frames = []
                    st.rerun()

        return None

# ==========================================
# 4. SMART PIPELINE (Unchanged)
# ==========================================

class SmartPhysicsPipeline:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def run(self, output_path, ai_function):
        # 1. Smart Filter
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        step = max(1, self.total_frames // 4)
        indices = [step, step*2, step*3]
        
        # 2. AI Analysis
        frames = []
        for idx in indices:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, f = self.cap.read()
            if ret: frames.append(f)
            
        vector_data = ai_function(frames) if frames else []
        
        mapped = {}
        if isinstance(vector_data, list):
            for i, d in enumerate(vector_data):
                if i < len(indices): mapped[indices[i]] = d

        # 3. Reconstruction
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (self.width, self.height))
        
        idx = 0
        sorted_keys = sorted(mapped.keys())
        
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            
            current_data = None
            if sorted_keys:
                closest_k = min(sorted_keys, key=lambda x: abs(x - idx))
                if abs(closest_k - idx) < 30:
                    current_data = mapped[closest_k]

            if current_data and 'vectors' in current_data:
                for v in current_data['vectors']:
                    try:
                        sx, sy = v['start']
                        ex, ey = v['end']
                        if isinstance(sx, float) and sx <= 1.0: sx, sy = int(sx*self.width), int(sy*self.height)
                        if isinstance(ex, float) and ex <= 1.0: ex, ey = int(ex*self.width), int(ey*self.height)
                        cv2.arrowedLine(frame, (int(sx), int(sy)), (int(ex), int(ey)), (0, 255, 0), 4)
                    except: pass
            
            out.write(frame)
            idx += 1
            
        out.release()
        return output_path, indices