import cv2
import numpy as np

class MotionTracker:
    def __init__(self):
        # MOG2 is a Gaussian Mixture-based Background/Foreground Segmentation Algorithm
        # detectShadows=False is faster and often cleaner for simple physics
        self.back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
        self.prev_center = None
        self.velocity = (0, 0)
        
        # Smooth the velocity to avoid jittery arrows
        self.velocity_buffer = [] 
        self.max_buffer_size = 5

    def process_frame(self, frame):
        """
        Detects the main moving object and calculates its velocity.
        Returns: (center_x, center_y), (velocity_x, velocity_y)
        """
        # 1. Pre-processing (Blur to reduce noise)
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # 2. Apply Background Subtraction to get the "Mask"
        # White pixels = moving stuff, Black = static background
        fg_mask = self.back_sub.apply(blurred)
        
        # 3. Clean up the mask (remove small white dots/noise)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # 4. Find Contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        center = None
        
        # 5. Find the largest moving object
        if contours:
            # Get largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Filter out tiny movements (noise)
            if cv2.contourArea(largest_contour) > 500:
                # Get bounding box and center
                x, y, w, h = cv2.boundingRect(largest_contour)
                cx = x + w // 2
                cy = y + h // 2
                center = (cx, cy)
                
                # Draw the bounding box on the mask for debugging (optional)
                # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 6. Calculate Velocity (Change in position)
        if center and self.prev_center:
            dx = center[0] - self.prev_center[0]
            dy = center[1] - self.prev_center[1]
            
            # Smooth the velocity
            self.velocity_buffer.append((dx, dy))
            if len(self.velocity_buffer) > self.max_buffer_size:
                self.velocity_buffer.pop(0)
            
            # Average velocity
            avg_dx = sum(v[0] for v in self.velocity_buffer) / len(self.velocity_buffer)
            avg_dy = sum(v[1] for v in self.velocity_buffer) / len(self.velocity_buffer)
            self.velocity = (avg_dx, avg_dy)
        else:
            # If object stops or is lost, decay velocity
            self.velocity = (self.velocity[0]*0.8, self.velocity[1]*0.8)

        # Update previous center
        if center:
            self.prev_center = center
            
        return center, self.velocity