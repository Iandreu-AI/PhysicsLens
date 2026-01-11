import cv2
import os
import numpy as np
import subprocess
import shutil
from overlay_utils import PhysicsOverlay

def is_ffmpeg_installed():
    """Checks if ffmpeg is available in the system path."""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False

def convert_to_h264(input_path, output_path):
    """
    Converts video to H.264/AAC for web compatibility using FFmpeg.
    """
    try:
        # -y: Overwrite
        # -vcodec libx264: Force H.264
        # -pix_fmt yuv420p: Mandatory color space for browser playback
        # -movflags +faststart: Moves metadata to front for instant streaming
        command = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "23",
            "-preset", "ultrafast",  # Speed up processing for Hackathons
            "-movflags", "+faststart",
            output_path
        ]
        
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception as e:
        print(f"FFmpeg conversion failed: {e}")
        # Fallback: Copy original if conversion dies, though it might not play
        if input_path != output_path:
            shutil.copy2(input_path, output_path)
        return False

def generate_annotated_video(input_path, output_path, tracker, ai_metadata=None):
    """
    Process video with tracking, sanitizing resolution for H.264 compatibility.
    """
    if not os.path.exists(input_path):
        return False, None

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return False, None

    # --- 1. RESOLUTION SANITIZATION (The Black Screen Fix) ---
    # H.264 macroblocks require dimensions to be divisible by 2.
    raw_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    width = raw_width - (raw_width % 2)
    height = raw_height - (raw_height % 2)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120: fps = 30.0

    # --- 2. TEMP FILE WRITING ---
    # We write to a temp file first. We use 'mp4v' for the intermediate step
    # because it is the most stable writer inside OpenCV.
    temp_output_path = output_path.replace(".mp4", "_temp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
    
    tracker.mode = "DETECT"
    draw_gravity = True if (ai_metadata and isinstance(ai_metadata, list)) else False

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # RESIZE IF NEEDED (To fix odd-dimension bug)
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            
            # --- TRACKING & DRAWING ---
            center, velocity = tracker.process_frame(frame)
            
            if center:
                vx, vy = velocity
                # Velocity Vector
                if abs(vx) > 0.5 or abs(vy) > 0.5:
                    PhysicsOverlay.draw_vector(frame, center, (vx, vy), label="v", 
                                             color=PhysicsOverlay.COLOR_VELOCITY, scale=4.0)
                # Gravity Vector
                if draw_gravity:
                    PhysicsOverlay.draw_vector(frame, center, (0, 50), label="Fg", 
                                             color=PhysicsOverlay.COLOR_GRAVITY, scale=1.0)
                
                # Center point
                cv2.circle(frame, center, 5, (0, 255, 255), -1)

                # HUD
                speed = np.sqrt(vx**2 + vy**2)
                PhysicsOverlay.draw_hud(frame, {"Speed": f"{speed:.1f}", "State": tracker.mode})
            else:
                PhysicsOverlay.draw_smart_label(frame, "Scanning...", (30, height - 40))

            out.write(frame)
            
    except Exception as e:
        print(f"Error processing frames: {e}")
        return False, None
    finally:
        cap.release()
        out.release()

    # --- 3. TRANSCODING (The Browser Fix) ---
    print("Finalizing video codec...")
    
    if is_ffmpeg_installed():
        # Best case: Use FFmpeg to make a perfect H.264 file
        success = convert_to_h264(temp_output_path, output_path)
    else:
        # Worst case: FFmpeg missing. 
        # Attempt to rename the intermediate file, but warn the user.
        print("WARNING: FFmpeg not found. Video may be black in Chrome/Safari.")
        shutil.move(temp_output_path, output_path)
        success = True

    # Clean up temp
    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)

    return success, output_path