import google.generativeai as genai
from PIL import Image
import json
import streamlit as st

def configure_gemini(api_key):
    """Configures the Gemini API with the provided key."""
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"API Configuration Error: {e}")
        return False

def analyze_physics_with_gemini(frames_data, analysis_level="High School Physics"):
    """
    Sends a sequence of frames to Gemini 1.5 Flash (Fast & Multimodal)
    to analyze the physics taking place.
    
    Args:
        frames_data (list): List of dicts from our video_utils (contains 'frame' as numpy array).
        analysis_level (str): The complexity of the explanation.
        
    Returns:
        dict: Parsed JSON response from Gemini.
    """
    
    # 1. Model Selection
    # 'gemini-1.5-flash' is optimized for high-volume, low-latency multimodal tasks.
    # It is significantly faster/cheaper than Pro for video frame analysis.
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # 2. Data Preparation: Convert Numpy (OpenCV) -> PIL Images
    # We take a sample to avoid token limits (e.g., max 10 frames for this demo)
    # We ensure we select frames evenly distributed across the clip.
    sample_size = 10
    step = max(1, len(frames_data) // sample_size)
    selected_frames = frames_data[::step][:sample_size]
    
    pil_images = []
    for meta in selected_frames:
        # Convert RGB numpy array to PIL Image
        img = Image.fromarray(meta['frame'])
        pil_images.append(img)
        
    # 3. The Expert Physics Prompt
    # We demand JSON format to enable programmatic overlays later.
    prompt = f"""
    You are a physics expert. Analyze this sequence of {len(pil_images)} video frames.
    Level of explanation: {analysis_level}.

    Identify the primary moving object and the physics principles at play.
    
    RETURN ONLY RAW JSON. Do not use Markdown code blocks.
    Structure:
    {{
        "main_object": "name of object",
        "physics_principle": "Name of principle (e.g., Projectile Motion, Conservation of Momentum)",
        "velocity_estimation": "Approximate speed estimate with units (e.g., '15 m/s')",
        "explanation": "A concise explanation of the motion tailored to a {analysis_level} audience.",
        "vectors": [
            {{ "frame_index": 0, "description": "initial release", "force_direction": "up-right" }},
            {{ "frame_index": {len(pil_images)-1}, "description": "impact/end", "force_direction": "down" }}
        ]
    }}
    """
    
    # 4. Multimodal Request
    # We pass the prompt + the list of images
    request_content = [prompt] + pil_images
    
    try:
        response = model.generate_content(request_content)
        
        # 5. Robust JSON Cleaning (The "Senior Dev" Touch)
        # LLMs often wrap JSON in ```json ... ```. We clean that.
        raw_text = response.text
        clean_text = raw_text.replace("```json", "").replace("```", "").strip()
        
        return json.loads(clean_text)
        
    except Exception as e:
        return {
            "error": str(e),
            "raw_response": response.text if 'response' in locals() else "No response"
        }