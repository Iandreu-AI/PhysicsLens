import google.generativeai as genai
import PIL.Image
import json
import cv2
import numpy as np
import time

def configure_gemini(api_key):
    """Configures the Gemini API."""
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        print(f"Configuration Error: {e}")
        return False

def get_batch_physics_overlays(frames_bgr_list):
    """
    Analyzes multiple frames in a single API call to ensure 
    temporal consistency and reduce latency.
    """
    try:
        # Using Flash for high-speed vision processing
        model = genai.GenerativeModel('gemini-2.0-flash-exp') 
        
        # 1. Prepare the Prompt
        prompt_parts = [
            """
            You are a Physics Analysis Engine. I am providing a sequence of keyframes from a video.
            Your task is to analyze the motion consistency across these frames and generate vector data for EACH frame.

            ### Requirement
            Identify the MAIN moving object. It must be the SAME object across all frames.
            For each frame, return the coordinates for:
            1. Object Center (normalized 0.0-1.0)
            2. Active Forces (Gravity, Velocity, Friction, Normal, etc.)

            ### Output Format
            Return ONLY a valid JSON List of objects. No markdown.
            Structure:
            [
                {
                    "frame_index": 0,
                    "object_center": [x, y],
                    "vectors": [ 
                        {"name": "Gravity", "start": [x,y], "end": [x,y], "color": "red"},
                        {"name": "Velocity", "start": [x,y], "end": [x,y], "color": "green"}
                    ]
                },
                ...
            ]
            
            Important: 
            - Coordinates must be floats between 0.0 and 1.0.
            - Ensure vectors are physically logical (e.g., Gravity always points down).
            """
        ]

        # 2. Append Images to Prompt
        for idx, frame in enumerate(frames_bgr_list):
            # Convert BGR (OpenCV) to RGB (PIL)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = PIL.Image.fromarray(rgb)
            
            prompt_parts.append(f"--- Frame Index {idx} ---")
            prompt_parts.append(pil_img)

        # 3. Fire Single API Call
        response = model.generate_content(prompt_parts)
        
        # 4. Parse Response
        text = response.text
        text = text.replace("```json", "").replace("```", "").strip()
        
        data = json.loads(text)
        
        # Handle edge case where AI wraps list in a dict
        if isinstance(data, dict): 
            data = data.get("frames", data.get("data", [data]))
            
        return data

    except Exception as e:
        print(f"Batch Overlay Error: {e}")
        return []

def analyze_physics_with_gemini(frames_data, analysis_level="High School Physics"):
    """
    Analyzes the video frames to produce the educational text explanation.
    """
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        if not frames_data:
            return {"error": "No frames to analyze"}
            
        # Select the middle frame for main context
        mid_idx = len(frames_data) // 2
        ref_frame = frames_data[mid_idx]['frame'] # Already RGB
        pil_image = PIL.Image.fromarray(ref_frame)

        prompt = f"""
        You are an expert Physics Professor. Analyze this video frame.
        
        Target Audience Level: {analysis_level}
        
        Return a JSON object with this structure (No Markdown):
        {{
            "main_object": "Specific object name",
            "velocity_estimation": "Qualitative + quantitative motion description (e.g. 'Fast downward, ~10m/s')",
            "physics_principle": "Primary physics principle (e.g. 'Projectile Motion')",
            "explanation": "A 3-5 sentence explanation of the physics observed, tailored to the audience level."
        }}
        """

        response = model.generate_content([prompt, pil_image])
        
        text = response.text
        text = text.replace("```json", "").replace("```", "").strip()
        
        return json.loads(text)

    except Exception as e:
        print(f"Analysis Error: {e}")
        return {
            "error": str(e),
            "main_object": "Unknown",
            "physics_principle": "Analysis Failed",
            "velocity_estimation": "N/A",
            "explanation": "Could not process video due to an AI error. Please try again."
        }