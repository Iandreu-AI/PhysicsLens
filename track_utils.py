import cv2
import numpy as np

class MotionTracker:
    def __init__(self, verification_frames=3):
        """
        A Hybrid Tracker optimized for handheld video.
        
        Args:
            verification_frames (int): lowered to 3 for faster lock-on.
        """
        # --- State Management ---
        self.mode = "DETECT" # Options: "DETECT", "TRACK"
        
        # --- Detection Tools ---
        # History=100 adapts faster to changing backgrounds (handheld camera)
        self.detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=False)
        self.consecutive_detections = 0
        self.potential_bbox = None
        self.verification_limit = verification_frames
        
        # --- Tracking Tools ---
        self.tracker = None
        
        # --- Physics Data ---
        self.prev_center = None
        self.velocity = (0, 0)
        self.velocity_buffer = [(0,0)] * 5 

    def _create_tracker(self):
        """Factory to create a robust tracker."""
        if hasattr(cv2, 'TrackerCSRT_create'):
            return cv2.TrackerCSRT_create()
        if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
            return cv2.legacy.TrackerCSRT_create()
        if hasattr(cv2, 'TrackerKCF_create'):
            return cv2.TrackerKCF_create()
        return cv2.TrackerMIL_create()

    def _smooth_velocity(self, current_v):
        """Apply Moving Average to reduce 'jitter'."""
        self.velocity_buffer.pop(0)
        self.velocity_buffer.append(current_v)
        avg_x = sum(v[0] for v in self.velocity_buffer) / len(self.velocity_buffer)
        avg_y = sum(v[1] for v in self.velocity_buffer) / len(self.velocity_buffer)
        return (avg_x, avg_y)

    def process_frame(self, frame):
        """
        Returns: center (x, y), velocity (vx, vy)
        """
        h, w = frame.shape[:2]
        center = None
        
        # ==========================================================
        # MODE A: DETECTION
        # ==========================================================
        if self.mode == "DETECT":
            # 1. Aggressive Preprocess (Blur more to ignore camera grain)
            blurred = cv2.GaussianBlur(frame, (21, 21), 0)
            mask = self.detector.apply(blurred)
            
            # 2. Clean Mask
            _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
            
            # 3. Find Contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                
                # Broad size filter
                if 200 < area < (w * h * 0.8):
                    x, y, bw, bh = cv2.boundingRect(largest)
                    self.potential_bbox = (x, y, bw, bh)
                    self.consecutive_detections += 1
                    
                    # Return center IMMEDIATELY so user sees blue box
                    center = (int(x + bw/2), int(y + bh/2))
                else:
                    self.consecutive_detections = max(0, self.consecutive_detections - 1)
            
            # 4. Lock On Logic
            if self.consecutive_detections >= self.verification_limit:
                print(">>> Object Locked. Switching to Tracker.")
                self.tracker = self._create_tracker()
                if self.tracker:
                    self.tracker.init(frame, self.potential_bbox)
                    self.mode = "TRACK"

        # ==========================================================
        # MODE B: TRACKING
        # ==========================================================
        elif self.mode == "TRACK":
            success, bbox = self.tracker.update(frame)
            if success:
                bx, by, bw, bh = [int(v) for v in bbox]
                center = (int(bx + bw/2), int(by + bh/2))
            else:
                print(">>> Tracking Lost. Resetting.")
                self.mode = "DETECT"
                self.consecutive_detections = 0
                # Don't reset detector immediately, just mode
        
        # ==========================================================
        # PHYSICS
        # ==========================================================
        if center and self.prev_center:
            raw_vx = center[0] - self.prev_center[0]
            raw_vy = center[1] - self.prev_center[1]
            self.velocity = self._smooth_velocity((raw_vx, raw_vy))
        
        # Decay if lost
        if not center:
             self.velocity = (self.velocity[0] * 0.9, self.velocity[1] * 0.9)
             # Prevent flickering by holding last position briefly?
             if self.prev_center:
                 center = self.prev_center

        if center:
            self.prev_center = center

        return center, self.velocity