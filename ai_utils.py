import google.generativeai as genai
import PIL.Image
import json
import cv2
import numpy as np

# 1. Config
def configure_gemini(api_key):
    genai.configure(api_key=api_key)
    return True

# 2. The Text Analysis Function (Keep this from before if you have it)
def analyze_physics_with_gemini(frames_list, analysis_level="High School Physics"):
    # ... your existing logic for text explanation ...
    pass 

# 3. The New Coordinate Function (Add this)
def get_physics_overlay_coordinates(frame_bgr):
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    # Convert BGR to RGB
    color_converted = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = PIL.Image.fromarray(color_converted)

    prompt = """
    You are a Physics Engine Computer Vision module.
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
    Rules: Coordinates must be normalized (0.0 to 1.0).
    """

    try:
        response = model.generate_content([prompt, pil_image])
        text_response = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text_response)
        return data
    except Exception as e:
        print(f"Gemini Coordinate Error: {e}")
        return None