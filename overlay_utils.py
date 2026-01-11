import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Union, List

# --- PART 1: CONFIGURATION & STYLING ---
@dataclass
class VisualConfig:
    """
    Centralized configuration for physics visualization styling.
    Colors are defined in BGR format for OpenCV.
    """
    # Semantic Colors
    COLOR_GRAVITY: Tuple[int, int, int] = (0, 0, 255)       # Red
    COLOR_VELOCITY: Tuple[int, int, int] = (255, 0, 0)      # Blue
    COLOR_NORMAL: Tuple[int, int, int] = (0, 255, 0)        # Green
    COLOR_FRICTION: Tuple[int, int, int] = (0, 165, 255)    # Orange
    COLOR_PATH: Tuple[int, int, int] = (0, 255, 255)        # Yellow
    COLOR_DEFAULT: Tuple[int, int, int] = (0, 255, 0)       # Fallback Green
    
    # Text Styling
    COLOR_TEXT: Tuple[int, int, int] = (255, 255, 255)      # White
    COLOR_TEXT_OUTLINE: Tuple[int, int, int] = (0, 0, 0)    # Black
    
    # Rendering Constants
    FONT: int = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE: float = 0.5
    LINE_THICKNESS: int = 2
    AA_MODE: int = cv2.LINE_AA  # Anti-aliasing for smooth edges

# --- PART 2: RENDERER CLASS ---
class PhysicsOverlay:
    """
    The main renderer class used by app.py. 
    Combines high-fidelity drawing with the app's specific logic.
    """
    
    # --- EXPOSED CONSTANTS FOR VIDEO_UTILS ---
    COLOR_VELOCITY = VisualConfig.COLOR_VELOCITY
    COLOR_GRAVITY = VisualConfig.COLOR_GRAVITY    # <--- Fixed: Added this
    COLOR_FRICTION = VisualConfig.COLOR_FRICTION  # <--- Fixed: Added this
    COLOR_FORCE = VisualConfig.COLOR_GRAVITY      # Alias

    @staticmethod
    def _to_point(pt) -> Tuple[int, int]:
        """Safely converts numpy/float coordinates to integer pixels."""
        if isinstance(pt, np.ndarray):
            pt = pt.flatten()
        return (int(round(pt[0])), int(round(pt[1])))

    @staticmethod
    def _hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
        """Converts '#RRGGBB' string to (B, G, R) tuple for OpenCV."""
        try:
            if not isinstance(hex_color, str):
                return VisualConfig.COLOR_DEFAULT
            
            hex_color = hex_color.lstrip('#')
            if len(hex_color) != 6:
                return VisualConfig.COLOR_DEFAULT
                
            # Parse RGB
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            
            # Return BGR
            return (b, g, r)
        except Exception:
            return VisualConfig.COLOR_DEFAULT

    @staticmethod
    def draw_smart_label(frame, text, position, bg_color=None):
        """Draws high-contrast text with an outline."""
        x, y = position
        # Outline
        cv2.putText(frame, text, (x, y), VisualConfig.FONT, VisualConfig.FONT_SCALE, 
                    VisualConfig.COLOR_TEXT_OUTLINE, 3, VisualConfig.AA_MODE)
        # Inner Text
        text_c = bg_color if bg_color else VisualConfig.COLOR_TEXT
        cv2.putText(frame, text, (x, y), VisualConfig.FONT, VisualConfig.FONT_SCALE, 
                    text_c, 1, VisualConfig.AA_MODE)

    @staticmethod
    def draw_vector(frame, start_point, vector, label=None, color=(0,255,0), scale=1.0):
        """
        Draws a physics vector (arrow). Used by the CV Tracker in app.py.
        """
        if start_point is None or vector is None: return frame
        
        p1 = PhysicsOverlay._to_point(start_point)
        dx = vector[0] * scale
        dy = vector[1] * scale
        p2 = (int(p1[0] + dx), int(p1[1] + dy))
        
        # Don't draw tiny noise vectors
        magnitude = np.sqrt(dx**2 + dy**2)
        if magnitude < 5.0: return frame

        # Dynamic Tip Length
        tip_len = 0.3 if magnitude < 50 else 0.15

        # Arrow
        cv2.arrowedLine(frame, p1, p2, color, VisualConfig.LINE_THICKNESS + 1, 
                        VisualConfig.AA_MODE, tipLength=tip_len)
        
        # Anchor Dot
        cv2.circle(frame, p1, 4, color, -1, VisualConfig.AA_MODE)

        # Label
        if label:
            PhysicsOverlay.draw_smart_label(frame, label, (p2[0]+10, p2[1]), color)
            
        return frame
    
    @staticmethod
    def draw_hud(frame, data: dict):
        """
        Draws a Heads-Up Display (HUD) with statistics in the top-left corner.
        """
        x, y = 20, 30  # Starting position
        for key, value in data.items():
            text = f"{key}: {value}"
            # Reuse our smart label function for outline/contrast
            PhysicsOverlay.draw_smart_label(frame, text, (x, y))
            y += 25  # Move down for next line
        return frame

    @staticmethod
    def draw_ai_overlay(frame, ai_data):
        """
        Draws vectors based on normalized Gemini JSON data.
        Used by the Keyframe Gallery in app.py.
        """
        if not ai_data: return frame
            
        h, w = frame.shape[:2]
        
        # Helper to convert normalized (0.0-1.0) to pixel coordinates
        def to_pix(coord):
            if not coord or len(coord) < 2: return (0,0)
            return (int(coord[0] * w), int(coord[1] * h))
        
        # Draw Center of Mass
        if "object_center" in ai_data:
            center = to_pix(ai_data["object_center"])
            # Outer Glow
            cv2.circle(frame, center, 8, (0,0,0), 2, VisualConfig.AA_MODE)
            # Inner Dot
            cv2.circle(frame, center, 6, (255,255,255), -1, VisualConfig.AA_MODE)
            
        # Draw Vectors from JSON
        if "vectors" in ai_data:
            for vec in ai_data["vectors"]:
                start = to_pix(vec.get("start", [0,0]))
                end = to_pix(vec.get("end", [0,0]))
                
                # Handle Color: AI sends Hex, or Name. We prefer Hex now.
                color_raw = vec.get("color", "green")
                if isinstance(color_raw, str) and color_raw.startswith("#"):
                    color = PhysicsOverlay._hex_to_bgr(color_raw)
                else:
                    # Fallback for old "name" based colors
                    c_map = {
                        "red": VisualConfig.COLOR_GRAVITY,
                        "green": VisualConfig.COLOR_VELOCITY,
                        "blue": VisualConfig.COLOR_NORMAL,
                        "orange": VisualConfig.COLOR_FRICTION
                    }
                    color = c_map.get(color_raw, VisualConfig.COLOR_VELOCITY)
                
                # Draw the vector
                # Note: We use arrowedLine directly here because we have absolute start/end points
                # rather than start + velocity_delta
                cv2.arrowedLine(frame, start, end, color, 4, VisualConfig.AA_MODE, tipLength=0.2)
                PhysicsOverlay.draw_smart_label(frame, vec.get("name", ""), end, color)
                
        return frame