import google.generativeai as genai
import PIL.Image
import json
import cv2
import numpy as np

def configure_gemini(api_key):
    """Configures the Gemini API."""
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        print(f"Configuration Error: {e}")
        return False

def get_physics_overlay_coordinates(frame_bgr):
    """
    Generates Free Body Diagram coordinates.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Convert BGR to RGB
        color_converted = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = PIL.Image.fromarray(color_converted)

        prompt = """
        You are a Physics Engine CV module.
        Analyze the image. Identify the MAIN moving physical object.
        
        Return a valid JSON object ONLY. No markdown.
        Format:
        {
            "object_center": [x, y],
            "vectors": [
                { "name": "Gravity", "start": [x, y], "end": [x, y], "color": "red" },
                { "name": "Velocity", "start": [x, y], "end": [x, y], "color": "green" },
                { "name": "Normal Force", "start": [x, y], "end": [x, y], "color": "blue" }
            ]
        }
        Rules: Coordinates must be normalized floats (0.0 to 1.0).
        """

        response = model.generate_content([prompt, pil_image])
        
        # CLEANUP JSON
        text = response.text
        text = text.replace("```json", "").replace("```", "").strip()
        
        # Return parsed JSON
        return json.loads(text)

    except Exception as e:
        print(f"Overlay Error: {e}")
        return None

def analyze_physics_with_gemini(frames_data, analysis_level="High School Physics"):
    """
    Analyzes the video frames to produce the text explanation.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash') # Flash is faster for text
        
        # We take the middle frame as the reference image
        if not frames_data:
            return {"error": "No frames to analyze"}
            
        mid_idx = len(frames_data) // 2
        ref_frame = frames_data[mid_idx]['frame'] # This is already RGB from video_utils
        pil_image = PIL.Image.fromarray(ref_frame)

        prompt = f"""
        You are a Physics Professor analyzing a video of a moving object.
        Level: {analysis_level}.
        
        Analyze the image and provide a physics breakdown.
        
        Return JSON ONLY:
        {{
            "main_object": "e.g., Basketball",
            "velocity_estimation": "e.g., Fast moving downward",
            "physics_principle": "e.g., Projectile Motion / Conservation of Energy",
            "explanation": "A concise {analysis_level} explanation of the forces at play."
        }}
        """

        response = model.generate_content([prompt, pil_image])
        
        # CLEANUP JSON
        text = response.text
        text = text.replace("```json", "").replace("```", "").strip()
        
        # --- THE FIX IS HERE ---
        # We assign to 'parsed_data' and return 'parsed_data'
        parsed_data = json.loads(text)
        return parsed_data

    except Exception as e:
        print(f"Analysis Error: {e}")
        # Return a robust error dict so app.py doesn't crash
        return {
            "error": str(e),
            "explanation": "Could not process video due to an AI error."
        }