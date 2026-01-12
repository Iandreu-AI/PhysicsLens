import google.generativeai as genai
from PIL import Image
import json
import cv2
import numpy as np
import re
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
    """Sanitizes Gemini response to ensure valid JSON."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()

def _fix_invalid_escapes(text):
    """
    Scans for backslashes that are NOT followed by valid JSON escape characters
    and double-escapes them.
    Valid JSON escapes: ", \, /, b, f, n, r, t, u
    Invalid (LaTeX) examples that crash JSON: \s, \p, \c, \d
    """
    # Regex explanation:
    # (?<!\\) -> Lookbehind: Ensure current char is not preceded by a backslash (avoids \\)
    # \\      -> Match a literal backslash
    # (?![\\"/bfnrtu]) -> Lookahead: Ensure next char is NOT a valid escape char
    # We replace match with \\\\ (double backslash)
    return re.sub(r'(?<!\\)\\(?![\\"/bfnrtu])', r'\\\\', text)

def _robust_json_load(text):
    """
    Tries multiple strategies to parse JSON from AI.
    """
    original_text = text
    
    # STRATEGY 1: Direct Parse (Best Case)
    try:
        return json.loads(text)
    except Exception:
        pass 
    
    # STRATEGY 2: Fix Invalid Escapes (Targeted Regex)
    try:
        fixed_text = _fix_invalid_escapes(text)
        return json.loads(fixed_text)
    except Exception:
        pass

    # STRATEGY 3: Python AST Eval (Last Resort for single quotes)
    try:
        # Convert null/true/false to Python None/True/False just in case
        py_text = text.replace("null", "None").replace("true", "True").replace("false", "False")
        # Fix escapes for Python eval as well
        py_text = _fix_invalid_escapes(py_text)
        return ast.literal_eval(py_text)
    except Exception:
        pass

    # FALLBACK: Return a dummy object so the app doesn't crash
    print(f"CRITICAL: JSON Parsing Failed. Raw text sample: {original_text[:100]}...")
    return {
        "error": "Data Parsing Error",
        "explanation": "The AI analysis generated complex mathematical symbols that could not be processed. However, the video tracking is still available.",
        "main_object": "Unknown Object",
        "motion_type": "Unknown",
        "visual_cues": "N/A",
        "active_forces": [],
        "physics_principle": "Analysis Unavailable",
        "latex_equations": [],
        "key_formula": ""
    }

# --- CORE FUNCTIONS ---

def get_batch_physics_overlays(frames_bgr_list):
    """Sends frames to Gemini to get vector coordinates."""
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Simplified prompt to reduce risk of weird text
        prompt_text = """
        Analyze frames. Return JSON for physics vectors.
        Structure:
        [
          {
            "frame_index": 0,
            "vectors": [ {"name": "Gravity", "start": [0.5, 0.5], "end": [0.5, 0.8], "color": "#FF0000"} ]
          }
        ]
        RETURN RAW JSON ONLY.
        """
        
        content_payload = [prompt_text]
        for idx, frame in enumerate(frames_bgr_list):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            content_payload.append(f"--- Frame {idx} ---")
            content_payload.append(pil_img)

        response = model.generate_content(content_payload, safety_settings=safety_settings)
        clean_text = _clean_json_response(response.text)
        data = _robust_json_load(clean_text)
        
        if isinstance(data, dict): 
            # Handle potential error dict from fallback
            if "error" in data and "frames" not in data:
                return []
            data = data.get("frames", data.get("data", [data]))
            
        return data

    except Exception as e:
        print(f"Batch Overlay Error: {e}")
        return []

def analyze_physics_with_gemini(keyframes, analysis_level="High School Physics"):
    """Analyzes video frames for educational text."""
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        if not keyframes:
            return {"error": "No frames"}
            
        mid_idx = len(keyframes) // 2
        ref_frame_rgb = keyframes[mid_idx]['frame']
        pil_image = Image.fromarray(ref_frame_rgb)

        # UPDATED PROMPT: Explicit instructions on escaping
        prompt = f"""
        You are a Physics AI. Analyze this image for a {analysis_level} audience.
        
        CRITICAL JSON RULES:
        1. Output strictly valid JSON.
        2. IF YOU WRITE LATEX, YOU MUST DOUBLE-ESCAPE BACKSLASHES.
           Wrong: "\\frac" or "\\sigma"
           Right: "\\\\frac" or "\\\\sigma"
        
        JSON Structure:
        {{
            "main_object": "Object Name",
            "motion_type": "Type",
            "visual_cues": "Observation",
            "active_forces": ["Gravity", "Friction"],
            "physics_principle": "Principle",
            "key_formula": "LaTeX Equation (e.g. F = ma)",
            "latex_equations": ["Eq1", "Eq2"],
            "explanation": "Brief explanation."
        }}
        """

        response = model.generate_content(
            [prompt, pil_image],
            safety_settings=safety_settings 
        )
        
        clean_text = _clean_json_response(response.text)
        return _robust_json_load(clean_text)

    except Exception as e:
        return {
            "error": str(e),
            "explanation": "Connection to AI failed."
        }