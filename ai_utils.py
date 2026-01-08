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
    model = genai.GenerativeModel('gemini-3-pro-preview')
    
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
You are an expert physics educator with deep knowledge of classical mechanics, kinematics, dynamics, and motion analysis. Your task is to analyze a sequence of video frames and extract physics insights.

## Input Context
- **Number of frames:** {len(pil_images)}
- **Target audience level:** {analysis_level}
  - If "beginner": Use simple language, avoid jargon, explain concepts intuitively
  - If "intermediate": Use standard physics terminology, include relevant equations
  - If "advanced": Use technical language, discuss subtleties, reference advanced concepts

## Analysis Requirements

### Step 1: Object Identification
Examine all {len(pil_images)} frames carefully and identify:
- The PRIMARY moving object (the most significant object in motion)
- Distinguish between the object of interest and background/stationary elements
- If multiple objects are moving, select the one exhibiting the clearest physics principle

### Step 2: Physics Principle Classification
Determine which physics principle(s) best explain the motion. Consider:
- **Projectile Motion**: Object follows parabolic trajectory under gravity
- **Conservation of Momentum**: Collisions or interactions between objects
- **Circular Motion**: Object moving in circular/curved path with centripetal force
- **Simple Harmonic Motion**: Oscillatory motion (pendulum, spring, etc.)
- **Free Fall**: Object accelerating under gravity alone
- **Friction/Drag**: Motion affected by resistive forces
- **Rotational Dynamics**: Spinning or rolling motion
- **Other**: Specify if different principle applies

### Step 3: Velocity Estimation
Estimate the approximate speed of the object:
- Use visual cues: displacement between frames, blur, trajectory shape
- Provide realistic magnitude with appropriate units (m/s, km/h, mph)
- Include brief reasoning if unusual (e.g., "fast due to steep trajectory")
- If velocity changes significantly, note this (e.g., "10-25 m/s, accelerating")

### Step 4: Explanation Generation
Craft an explanation tailored to {analysis_level}:
- **Beginner**: Use analogies, avoid equations, explain "what" and "why" in simple terms
- **Intermediate**: Include key physics terms, mention relevant forces, reference equations conceptually
- **Advanced**: Discuss vector components, reference frames, energy transformations, simplifying assumptions

Keep explanation concise (2-4 sentences) but insightful.

### Step 5: Key Motion Vectors
Identify 2-4 critical moments in the motion sequence where force direction is notable:
- **frame_index**: Integer from 0 to {len(pil_images)-1} indicating which frame
- **description**: Brief label for this moment (e.g., "launch point", "apex", "before collision", "at rest")
- **force_direction**: Primary force direction using compass directions or combinations
  - Examples: "down" (gravity), "up-right" (initial thrust), "left" (friction opposing motion)
  - Use descriptive terms: "downward", "horizontal-right", "toward center", "tangential"

Select frames that show:
- Initial state (frame 0 or near beginning)
- Critical transition points (apex, collision, direction change)
- Final state (frame {len(pil_images)-1} or near end)

## Output Format Requirements

**CRITICAL:** Return ONLY raw JSON. Do NOT wrap in markdown code blocks. Do NOT include ```json or ``` markers.

The JSON must be valid and parseable. Use this exact structure:

{{
    "main_object": "descriptive name of the primary moving object",
    "physics_principle": "Name of the dominant physics principle (e.g., Projectile Motion, Conservation of Momentum, Circular Motion)",
    "velocity_estimation": "Approximate speed with units and brief context (e.g., '15 m/s, typical throwing speed', '5-8 m/s, decelerating due to friction')",
    "explanation": "A {analysis_level}-appropriate explanation of the motion. Describe what's happening physically, why it happens, and what forces are involved. Tailor complexity to the audience level.",
    "vectors": [
        {{
            "frame_index": 0,
            "description": "brief description of motion state at this moment",
            "force_direction": "primary force direction (e.g., 'up-right', 'downward', 'horizontal-left')"
        }},
        {{
            "frame_index": {len(pil_images)-1},
            "description": "brief description of motion state at this moment",
            "force_direction": "primary force direction"
        }}
    ]
}}

## Quality Checks Before Responding

Before outputting your JSON, verify:
- ✅ Is the identified object actually the PRIMARY moving object?
- ✅ Does the physics principle accurately describe the motion?
- ✅ Is the velocity estimation physically reasonable?
- ✅ Is the explanation appropriate for {analysis_level}?
- ✅ Do frame indices fall within valid range [0, {len(pil_images)-1}]?
- ✅ Are force directions physically accurate for those moments?
- ✅ Is the output PURE JSON without any markdown formatting?

## Example Output (Structure Reference Only)

{{
    "main_object": "basketball",
    "physics_principle": "Projectile Motion",
    "velocity_estimation": "12 m/s at release, typical basketball shot speed",
    "explanation": "The basketball follows a parabolic trajectory after being released. Gravity pulls it downward while its horizontal velocity remains roughly constant (ignoring air resistance). The arc shape results from the combination of initial upward velocity and constant downward gravitational acceleration.",
    "vectors": [
        {{
            "frame_index": 0,
            "description": "release from hands",
            "force_direction": "up-right at 60 degrees"
        }},
        {{
            "frame_index": 15,
            "description": "apex of trajectory",
            "force_direction": "downward (gravity only)"
        }},
        {{
            "frame_index": 29,
            "description": "approaching hoop",
            "force_direction": "downward"
        }}
    ]
}}

---

Now analyze the {len(pil_images)} frames and return your physics analysis as raw JSON.
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