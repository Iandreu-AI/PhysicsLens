import cv2
import numpy as np

class PhysicsOverlay:
    """
    Expert CV module for drawing physics overlays on frames.
    Designed to work with RGB frames (Streamlit standard).
    """
    
    # Colors in RGB format (since video_utils converts to RGB)
    COLOR_VELOCITY = (0, 255, 0)      # Green
    COLOR_FORCE = (255, 0, 0)         # Red
    COLOR_TRAJECTORY = (255, 255, 0)  # Yellow
    COLOR_TEXT = (255, 255, 255)      # White
    
    @staticmethod
    def draw_vector(frame, start_point, vector, label=None, color=(0,255,0), scale=1.0):
        """
        Draws a physics vector (arrow) on the frame.
        
        Args:
            frame: Numpy array (RGB).
            start_point: Tuple (x, y) for the origin.
            vector: Tuple (dx, dy) representing direction and magnitude.
            label: String to display next to the arrow.
            color: RGB tuple.
            scale: Visual scaling factor (pixels per unit).
        """
        if start_point is None or vector is None:
            return frame

        img_h, img_w = frame.shape[:2]
        
        # 1. Calculate End Point
        x_start, y_start = int(start_point[0]), int(start_point[1])
        dx, dy = vector
        
        # Apply scaling to make small physical vectors visible on screen
        x_end = int(x_start + (dx * scale))
        y_end = int(y_start + (dy * scale))
        
        # 2. Safety Check (don't draw if 0 length or way off screen)
        if x_start == x_end and y_start == y_end:
            return frame

        # 3. Draw Arrow
        # thickness=2, tipLength=0.2 (20% of line length)
        cv2.arrowedLine(
            frame, 
            (x_start, y_start), 
            (x_end, y_end), 
            color, 
            thickness=3, 
            line_type=cv2.LINE_AA, 
            tipLength=0.2
        )
        
        # 4. Draw Anchor Point (Origin dot)
        cv2.circle(frame, (x_start, y_start), 4, color, -1, cv2.LINE_AA)

        # 5. Draw Label
        if label:
            PhysicsOverlay._draw_label(frame, (x_end, y_end), label, color)
            
        return frame

    @staticmethod
    def _draw_label(frame, position, text, bg_color):
        """Helper to draw text with a background box for contrast."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        pad = 4

        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
        x, y = position
        
        # Ensure text doesn't go off-screen
        x = min(x, frame.shape[1] - text_w - 10)
        y = max(y, text_h + 10)

        # Draw background rectangle (semi-transparent look via solid draw)
        cv2.rectangle(frame, 
                      (x - pad, y - text_h - pad), 
                      (x + text_w + pad, y + pad), 
                      (0, 0, 0), # Black background
                      -1)
        
        # Draw Text
        cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    @staticmethod
    def draw_ai_overlay(frame, ai_data):
        """
        Draws vectors based on normalized Gemini JSON data.
        """
        if not ai_data:
            return frame
            
        h, w = frame.shape[:2]
        
        # Helper to convert normalized (0.0-1.0) to pixel (0-W)
        def to_pix(coord):
            return (int(coord[0] * w), int(coord[1] * h))
        
        # Draw Center of Mass
        if "object_center" in ai_data:
            center = to_pix(ai_data["object_center"])
            cv2.circle(frame, center, 6, (255, 255, 255), -1)
            cv2.circle(frame, center, 8, (0, 0, 0), 2)
            
        # Draw Vectors
        if "vectors" in ai_data:
            for vec in ai_data["vectors"]:
                start = to_pix(vec["start"])
                end = to_pix(vec["end"])
                
                # Parse Color name to BGR
                c_map = {
                    "red": (0, 0, 255),
                    "green": (0, 255, 0),
                    "blue": (255, 0, 0),
                    "yellow": (0, 255, 255)
                }
                color = c_map.get(vec.get("color", "green"), (0, 255, 0))
                
                # Draw using our existing robust method, but calculating delta manually
                cv2.arrowedLine(frame, start, end, color, 4, cv2.LINE_AA, tipLength=0.2)
                
                # Label
                PhysicsOverlay._draw_label(frame, end, vec["name"], color)
                
        return frame