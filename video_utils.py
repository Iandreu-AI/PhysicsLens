import cv2
import os

def get_video_frames_generator(video_path, stride=1, resize_width=None):
    """
    Expert-level video frame extractor.
    
    Args:
        video_path (str): Path to the temp video file.
        stride (int): Process every Nth frame. (1 = every frame, 5 = every 5th).
        resize_width (int): Optional. Resize frame width (maintaining aspect ratio).
                            
    Yields:
        dict: metadata containing 'frame' (RGB numpy array), 'timestamp', 'frame_id'
    """
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # 1. Initialize Capture
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video: {video_path}. Check codecs.")

    # 2. Get Video Properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_count = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break  # End of video
            
            # 3. Stride Optimization
            if frame_count % stride != 0:
                frame_count += 1
                continue

            # 4. Color Space Conversion (BGR -> RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 5. Optional Resizing
            if resize_width:
                h, w, _ = frame_rgb.shape
                scale = resize_width / w
                new_dim = (resize_width, int(h * scale))
                frame_rgb = cv2.resize(frame_rgb, new_dim, interpolation=cv2.INTER_AREA)

            # 6. Timestamp Calculation
            timestamp = frame_count / fps if fps > 0 else 0
            
            yield {
                "frame": frame_rgb,
                "original_frame_bgr": frame, 
                "timestamp": timestamp,
                "frame_id": frame_count,
                "total_frames": total_frames,
                "fps": fps
            }
            
            frame_count += 1
            
    finally:
        cap.release()

# --- NEW FUNCTION ADDED HERE ---

def generate_annotated_video(input_path, output_path, tracker_instance, ai_metadata=None):
    """
    Reads input video, applies tracking and overlays frame-by-frame, 
    and writes to a browser-compatible MP4.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return False
        
    # Video Properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize Writer
    # 'avc1' is H.264 (High compatibility for Chrome/Streamlit). 
    # If this fails on your machine, try 'mp4v' instead.
    fourcc = cv2.VideoWriter_fourcc(*'avc1') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Reset tracker for the full pass
    tracker_instance.mode = "DETECT" 
    tracker_instance.velocity_buffer = [(0,0)] * 5
    
    # Extract AI Force Rules
    has_gravity = False
    has_friction = False
    if ai_metadata and isinstance(ai_metadata, list):
        # Scan first valid frame data for force types
        first_data = next((x for x in ai_metadata if 'vectors' in x), None)
        if first_data:
            names = [v['name'].lower() for v in first_data.get('vectors', [])]
            has_gravity = any('gravity' in n for n in names)
            has_friction = any('friction' in n for n in names)

    # Local import to avoid circular dependencies (video_utils imports overlay_utils)
    from overlay_utils import PhysicsOverlay 
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Track Object
        center, velocity = tracker_instance.process_frame(frame)
        
        # 2. Draw Overlays
        if center:
            # A. Velocity Vector (Dynamic from CV)
            vx, vy = velocity
            if abs(vx) > 0.5 or abs(vy) > 0.5:
                PhysicsOverlay.draw_vector(frame, center, (vx, vy), 
                                         "v", PhysicsOverlay.COLOR_VELOCITY, scale=5.0)
            
            # B. Gravity Vector (Static from AI Context)
            if has_gravity or True: # Default to True for physics demo
                # Gravity always points down (0, 30)
                PhysicsOverlay.draw_vector(frame, center, (0, 40), 
                                         "mg", PhysicsOverlay.COLOR_GRAVITY, scale=1.0)
                
            # C. Friction (Opposite to velocity)
            if has_friction and abs(vx) > 1:
                fx = -vx * 0.3
                fy = -vy * 0.3
                PhysicsOverlay.draw_vector(frame, center, (fx, fy), 
                                         "fk", PhysicsOverlay.COLOR_FRICTION, scale=5.0)

            # D. HUD
            speed = (vx**2 + vy**2)**0.5
            PhysicsOverlay.draw_smart_label(frame, f"Speed: {speed:.1f} px/f", (30, height - 30))

        # 3. Write Frame
        out.write(frame)

    cap.release()
    out.release()
    return True