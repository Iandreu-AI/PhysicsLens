import cv2
import numpy as np

class PhysicsOverlay:
    # Colors
    COLOR_VELOCITY = (0, 255, 0)      # Green
    COLOR_FORCE = (255, 0, 0)         # Red
    COLOR_TEXT = (255, 255, 255)      # White
    
    @staticmethod
    def draw_vector(frame, start_point, vector, label=None, color=(0,255,0), scale=1.0):
        """Draws standard CV vectors (Velocity/Gravity)"""
        if start_point is None or vector is None: return frame
        
        x_start, y_start = int(start_point[0]), int(start_point[1])
        dx, dy = vector
        x_end = int(x_start + (dx * scale))
        y_end = int(y_start + (dy * scale))
        
        if x_start == x_end and y_start == y_end: return frame

        cv2.arrowedLine(frame, (x_start, y_start), (x_end, y_end), color, 3, cv2.LINE_AA, tipLength=0.2)
        cv2.circle(frame, (x_start, y_start), 4, color, -1, cv2.LINE_AA)

        if label:
            PhysicsOverlay._draw_label(frame, (x_end, y_end), label, color)
        return frame

    @staticmethod
    def draw_ai_overlay(frame, ai_data):
        """Draws vectors based on normalized Gemini JSON data."""
        if not ai_data: return frame
            
        h, w = frame.shape[:2]
        
        def to_pix(coord):
            return (int(coord[0] * w), int(coord[1] * h))
        
        if "object_center" in ai_data:
            center = to_pix(ai_data["object_center"])
            cv2.circle(frame, center, 6, (255, 255, 255), -1)
            cv2.circle(frame, center, 8, (0, 0, 0), 2)
            
        if "vectors" in ai_data:
            for vec in ai_data["vectors"]:
                start = to_pix(vec["start"])
                end = to_pix(vec["end"])
                
                c_map = {"red": (0,0,255), "green": (0,255,0), "blue": (255,0,0)}
                color = c_map.get(vec.get("color", "green"), (0,255,0))
                
                cv2.arrowedLine(frame, start, end, color, 4, cv2.LINE_AA, tipLength=0.2)
                PhysicsOverlay._draw_label(frame, end, vec["name"], color)
                
        return frame

    @staticmethod
    def _draw_label(frame, position, text, bg_color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        (text_w, text_h), _ = cv2.getTextSize(text, font, scale, 1)
        x, y = position
        x = min(x, frame.shape[1] - text_w - 10)
        y = max(y, text_h + 10)
        cv2.rectangle(frame, (x-2, y-text_h-2), (x+text_w+2, y+2), (0,0,0), -1)
        cv2.putText(frame, text, (x, y), font, scale, (255,255,255), 1, cv2.LINE_AA)

    @staticmethod
    def draw_hud(frame, data_dict):
        y = 30
        for k, v in data_dict.items():
            cv2.putText(frame, f"{k}: {v}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            y += 25
        return frame