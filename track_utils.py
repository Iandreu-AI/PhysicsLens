# --- REWRITE FILE: track_utils.py ---

import cv2
import numpy as np

class MotionTracker:
    def __init__(self, verification_frames=5):
        """
        A Hybrid Tracker that uses Background Subtraction to FIND the object,
        and then switches to a Correlation Filter Tracker (CSRT) to LOCK onto it.
        
        Args:
            verification_frames (int): Number of frames the object must be detected
                                     consistently before locking on.
        """
        # --- State Management ---
        self.mode = "DETECT" # Options: "DETECT", "TRACK"
        
        # --- Detection Tools (For finding the object initially) ---
        # History=500 means it learns the background quickly. 
        # VarThreshold=25 is sensitive enough for initial movement.
        self.detector = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)
        self.consecutive_detections = 0
        self.potential_bbox = None
        self.verification_limit = verification_frames
        
        # --- Tracking Tools (For handling camera shake/handheld) ---
        self.tracker = None
        self.tracker_bbox = None
        
        # --- Physics Data ---
        self.prev_center = None
        self.velocity = (0, 0)
        # Circular buffer for smooth velocity (last 5 frames)
        self.velocity_buffer = [(0,0)] * 5 

    def _create_tracker(self):
        """
        Factory to create a robust tracker. 
        Tries CSRT (Best accuracy), falls back to KCF (Fastest), then MIL.
        """
        # OpenCV 4.x+ API handling
        if hasattr(cv2, 'TrackerCSRT_create'):
            return cv2.TrackerCSRT_create()
        elif hasattr(cv2, 'legacy'):
            return cv2.legacy.TrackerCSRT_create()
        else:
            # Fallback for older OpenCV versions or specific builds
            print("Warning: CSRT Tracker not found, falling back to KCF.")
            return cv2.TrackerKCF_create()

    def _smooth_velocity(self, current_v):
        """Apply Moving Average to reduce 'jitter' in the arrow."""
        self.velocity_buffer.pop(0)
        self.velocity_buffer.append(current_v)
        
        avg_x = sum(v[0] for v in self.velocity_buffer) / len(self.velocity_buffer)
        avg_y = sum(v[1] for v in self.velocity_buffer) / len(self.velocity_buffer)
        return (avg_x, avg_y)

    def process_frame(self, frame):
        """
        Main pipeline processing.
        Returns: center (x, y), velocity (vx, vy)
        """
        h, w = frame.shape[:2]
        center = None
        
        # ==========================================================
        # MODE A: DETECTION (Look for movement)
        # ==========================================================
        if self.mode == "DETECT":
            # 1. Preprocess
            blurred = cv2.GaussianBlur(frame, (11, 11), 0)
            mask = self.detector.apply(blurred)
            
            # 2. Denoise mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # 3. Find Contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest moving object
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                
                # Filter: Must be significant size, but not THE WHOLE SCREEN
                # (Whole screen movement = Camera Panning, we want to ignore that during detection)
                if 500 < area < (w * h * 0.6):
                    x, y, bw, bh = cv2.boundingRect(largest)
                    self.potential_bbox = (x, y, bw, bh)
                    self.consecutive_detections += 1
                else:
                    self.consecutive_detections = 0
            else:
                self.consecutive_detections = 0
            
            # 4. Check if we are ready to LOCK ON
            if self.consecutive_detections >= self.verification_limit:
                print(">>> Object Locked. Switching to Tracker.")
                self.mode = "TRACK"
                self.tracker = self._create_tracker()
                self.tracker.init(frame, self.potential_bbox)
                
                # Set initial center
                bx, by, bw, bh = self.potential_bbox
                center = (int(bx + bw/2), int(by + bh/2))

        # ==========================================================
        # MODE B: TRACKING (Handle Handheld/Shake)
        # ==========================================================
        elif self.mode == "TRACK":
            success, bbox = self.tracker.update(frame)
            
            if success:
                self.tracker_bbox = bbox
                bx, by, bw, bh = [int(v) for v in bbox]
                center = (int(bx + bw/2), int(by + bh/2))
                
                # Draw Box for Debugging (Optional, can be removed for production)
                # cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)
            else:
                # Tracking lost (object left frame or moved too fast)
                print(">>> Tracking Lost. Resetting to Detection.")
                self.mode = "DETECT"
                self.consecutive_detections = 0
                self.detector = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)

        # ==========================================================
        # PHYSICS CALCULATION
        # ==========================================================
        if center and self.prev_center:
            # Calculate raw velocity (pixels per frame)
            raw_vx = center[0] - self.prev_center[0]
            raw_vy = center[1] - self.prev_center[1]
            
            # Apply smoothing
            self.velocity = self._smooth_velocity((raw_vx, raw_vy))
        else:
            # Decay velocity if object stops/lost
            self.velocity = (self.velocity[0] * 0.8, self.velocity[1] * 0.8)

        if center:
            self.prev_center = center

        return center, self.velocity