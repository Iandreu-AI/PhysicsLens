import google.generativeai as genai
import PIL.Image
import json
import cv2
import numpy as np

# Configure your API key here or in main app
def configure_gemini(api_key):
    genai.configure(api_key=api_key)
    return True

def get_physics_overlay_coordinates(frame_bgr):
    """
    Sends a frame to Gemini and asks for vector coordinates.
    Returns: JSON dict with normalized coordinates.
    """
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    # 1. Convert CV2 BGR to PIL Image (RGB)
    color_converted = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = PIL.Image.fromarray(color_converted)

    # 2. The Expert Prompt
    # We enforce specific JSON structure for robust parsing
    prompt = """
    You are a Physics Engine Computer Vision module.
    Analyze the image. Identify the MAIN moving physical object (ball, car, person, etc).
    
    Return a valid JSON object ONLY. No markdown, no text.
    The JSON must contain normalized coordinates (0.0 to 1.0) for a Free Body Diagram.
    
    Format:
    {
        "object_center": [x, y],
        "vectors": [
            { "name": "Gravity", "start": [x, y], "end": [x, y], "color": "red" },
            { "name": "Velocity", "start": [x, y], "end": [x, y], "color": "green" },
            { "name": "Normal Force", "start": [x, y], "end": [x, y], "color": "blue" }
        ]
    }
    
    Rules:
    - x=0, y=0 is top-left. x=1, y=1 is bottom-right.
    - If the object is in freefall, Gravity should point straight down.
    - If on a slope, Normal Force should be perpendicular to the surface.
    - Velocity should follow the implied motion.
    """

    try:
        response = model.generate_content([prompt, pil_image])
        
        # 3. Clean and Parse JSON
        text_response = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text_response)
        return data
        
    except Exception as e:
        print(f"Gemini Coordinate Error: {e}")
        return None