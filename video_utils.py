import cv2
import os

def get_video_frames_generator(video_path, stride=1, resize_width=None):
    """
    Expert-level video frame extractor.
    
    Args:
        video_path (str): Path to the temp video file.
        stride (int): Process every Nth frame. (1 = every frame, 5 = every 5th).
                      Crucial for performance when analyzing long videos.
        resize_width (int): Optional. Resize frame width (maintaining aspect ratio).
                            Reduces latency for heavy AI analysis.
                            
    Yields:
        dict: metadata containing 'frame' (RGB numpy array), 'timestamp', 'frame_id'
    """
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # 1. Initialize Capture
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video: {video_path}. Check codecs.")

    # 2. Get Video Properties (Essential for physics calculations)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_count = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break  # End of video
            
            # 3. Stride Optimization (Skip frames we don't need)
            if frame_count % stride != 0:
                frame_count += 1
                continue

            # 4. Color Space Conversion (BGR -> RGB)
            # This is the most common bug in CV apps. We fix it here.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 5. Optional Resizing (Optimization)
            if resize_width:
                h, w, _ = frame_rgb.shape
                scale = resize_width / w
                new_dim = (resize_width, int(h * scale))
                frame_rgb = cv2.resize(frame_rgb, new_dim, interpolation=cv2.INTER_AREA)

            # 6. Timestamp Calculation (Physics needs time, not just frame numbers)
            timestamp = frame_count / fps if fps > 0 else 0
            
            yield {
                "frame": frame_rgb,
                "original_frame_bgr": frame, # Keep original if we need to save back to video
                "timestamp": timestamp,
                "frame_id": frame_count,
                "total_frames": total_frames,
                "fps": fps
            }
            
            frame_count += 1
            
    finally:
        # 7. Resource Management
        cap.release()

# Example Usage Test
if __name__ == "__main__":
    # Mock test
    print("This module is designed to be imported.")