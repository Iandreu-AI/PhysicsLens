import google.generativeai as genai
from PIL import Image
import json
import cv2
import numpy as np
import re
import time
import ast

# --- CONFIGURATION ---

def configure_gemini(api_key):
    """Configures the Gemini API."""
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        print(f"Configuration Error: {e}")
        return False

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# --- ROBUST JSON PARSING ---

def _clean_json_response(text):
    """
    Sanitizes Gemini response to ensure valid JSON.
    Strips markdown code blocks.
    """
    text = text.strip()
    # Strip Markdown code blocks (```json ... ```)
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()

def _escape_latex_in_json(text):
    """
    The 'Nuclear' Option for fixing LaTeX in JSON.
    1. Escapes ALL backslashes (\\ -> \\\\).
    2. Reverts specific valid JSON escapes (\\\\n -> \\n, \\\\" -> \", etc).
    This ensures \\sigma becomes \\\\sigma (valid string) instead of \\s (invalid escape).
    """
    # 1. Escape everything
    fixed = text.replace('\\', '\\\\')
    
    # 2. Un-escape valid JSON control characters
    # We look for double backslashes followed by a valid JSON escape char
    # and replace them with single backslash + char
    # Valid JSON escapes: " \ / b f n r t u
    fixed = re.sub(r'\\\\(["\\/bfnrtu])', r'\\\1', fixed)
    
    return fixed

def _robust_json_load(text):
    """
    Tries multiple strategies to parse JSON from AI.
    """
    original_text = text
    
    # STRATEGY 1: Direct Parse (Best Case)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass # Fall through
    
    # STRATEGY 2: The Nuclear LaTeX Fix
    try:
        fixed_text = _escape_latex_in_json(text)
        return json.loads(fixed_text)
    except json.JSONDecodeError:
        pass # Fall through

    # STRATEGY 3: Python AST Eval (For single quotes or loose syntax)
    try:
        # ast.literal_eval is safer than eval, handles python-dict style
        # We need to ensure booleans are capitalized for Python
        py_style = text.replace("true", "True").replace("false", "False").replace("null", "None")
        return ast.literal_eval(py_style)
    except (ValueError, SyntaxError):
        pass

    # STRATEGY 4: Extract substring { ... }
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and start < end:
            json_candidate = text[start:end+1]
            return json.loads(_escape_latex_in_json(json_candidate))
    except json.JSONDecodeError:
        pass

    # FAILURE
    print(f"JSON Parsing Failed. Raw: {original_text[:200]}...")
    return {
        "error": "JSON Parsing Failed",
        "explanation": "Could not parse AI response. Try a simpler video.",
        "main_object": "Error",
        "raw_response": original_text
    }

# --- CORE FUNCTIONS ---

def get_batch_physics_overlays(frames_bgr_list):
    """
    Sends frames to Gemini to get vector coordinates.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Shortened prompt for brevity in this fix - keep your original if preferred
        prompt_text = """
        Analyze these frames. Identify the moving object.
        Return strictly valid JSON.
        Format:
        [
          {
            "frame_index": 0,
            "vectors": [ {"name": "Gravity", "start": [0.5, 0.5], "end": [0.5, 0.8], "color": "#FF0000"} ]
          }
        ]
        """
        
        content_payload = [prompt_text]
        for idx, frame in enumerate(frames_bgr_list):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            content_payload.append(f"--- Frame {idx} ---")
            content_payload.append(pil_img)

        response = model.generate_content(
            content_payload,
            safety_settings=safety_settings
        )
        
        clean_text = _clean_json_response(response.text)
        data = _robust_json_load(clean_text)
        
        if isinstance(data, dict): 
            data = data.get("frames", data.get("data", [data]))
            
        return data

    except Exception as e:
        print(f"Batch Overlay Error: {e}")
        return []

def analyze_physics_with_gemini(keyframes, analysis_level="High School Physics"):
    """
    Analyzes video frames for educational text.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        if not keyframes:
            return {"error": "No frames to analyze"}
            
        mid_idx = len(keyframes) // 2
        ref_frame_rgb = keyframes[mid_idx]['frame']
        pil_image = Image.fromarray(ref_frame_rgb)

        # Your original prompt logic here...
        prompt = f"""
        You are a Physics AI. Analyze this image for a {analysis_level} student.
        Identify forces, motion, and Provide LaTeX equations.
        
        IMPORTANT: Return RAW JSON.
        {{
            "main_object": "Object Name",
            "motion_type": "Type",
            "visual_cues": "Cues",
            "active_forces": ["Gravity"],
            "physics_principle": "Principle",
            "key_formula": "LaTeX here",
            "latex_equations": ["LaTeX 1", "LaTeX 2"],
            "explanation": "Explanation here."
        }}
        """

        response = model.generate_content(
            [prompt, pil_image],
            safety_settings=safety_settings 
        )
        
        clean_text = _clean_json_response(response.text)
        
        # USE THE ROBUST LOADER
        return _robust_json_load(clean_text)

    except Exception as e:
        return {
            "error": str(e),
            "explanation": "AI Analysis failed."
        }