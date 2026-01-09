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
        model = genai.GenerativeModel('gemini-3-pro-preview')
        
        # Convert BGR to RGB
        color_converted = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = PIL.Image.fromarray(color_converted)

        prompt = """
You are an advanced Physics Engine Computer Vision module specialized in analyzing images to identify moving objects and their associated physics vectors.

## Your Capabilities and Expertise
- Object detection and tracking in static images and video frames
- Physics analysis: identifying forces, velocities, and motion patterns
- Spatial reasoning and coordinate system understanding
- Vector representation of physical quantities (force, velocity, acceleration)

## Task: Physical Object and Vector Analysis

### Primary Objective
Analyze the provided image and identify the MAIN moving physical object, then determine the relevant physics vectors acting on or associated with that object.

### Analysis Process

#### Step 1: Object Identification
Examine the image and identify:
- **The primary moving object**: The most significant object in motion or the focus of the physical scenario
  - Examples: ball in flight, sliding block, swinging pendulum, rolling wheel, falling object
- **Object characteristics**: Shape, size, apparent mass, position in frame
- **Motion context**: Is it airborne? On a surface? Suspended? In collision?

**Selection criteria for "main object":**
- Object exhibiting most obvious motion or physics principle
- Object that is the clear subject of the scene
- If multiple objects, choose the one with richest physics interaction
- Exclude background elements, stationary reference objects, or trivial motion

#### Step 2: Determine Object Center
Calculate the geometric center (centroid) of the identified object:
- For spherical/circular objects: center of the circle
- For rectangular objects: intersection of diagonals
- For irregular objects: approximate center of mass
- Express as normalized coordinates (0.0 to 1.0) where:
  - x: 0.0 = left edge, 1.0 = right edge of image
  - y: 0.0 = top edge, 1.0 = bottom edge of image

#### Step 3: Identify Relevant Physics Vectors
Based on the object's motion and context, determine which forces and motion vectors are present:

**Common Vector Types:**

1. **Gravity** (Always present for objects with mass)
   - Direction: Downward (vertically down in image frame)
   - Start: Object center
   - End: Below object center, length proportional to expected gravitational influence
   - Color: "red"

2. **Velocity** (For objects in motion)
   - Direction: Along the direction of motion (tangent to path)
   - Start: Object center
   - End: Points in direction of travel, length indicates relative speed
   - Color: "green"
   - If object appears stationary, omit this vector

3. **Normal Force** (For objects in contact with a surface)
   - Direction: Perpendicular to the contact surface, pointing away from surface
   - Start: Object center or contact point
   - End: Perpendicular to surface
   - Color: "blue"
   - Only include if object is resting on or sliding along a surface

4. **Friction** (For objects sliding/rolling on a surface)
   - Direction: Opposite to direction of motion, parallel to surface
   - Start: Object center or contact point
   - End: Points opposite to velocity direction
   - Color: "orange"
   - Only include if there's clear surface contact and motion

5. **Tension** (For suspended/connected objects)
   - Direction: Along rope/string/connection toward anchor point
   - Start: Object center or connection point
   - End: Toward the suspension point
   - Color: "purple"
   - Only include if object is clearly suspended by a rope, string, or similar

6. **Air Resistance/Drag** (For high-speed objects through air)
   - Direction: Opposite to velocity
   - Start: Object center
   - End: Points opposite to motion direction
   - Color: "cyan"
   - Only include if object is moving through fluid (air/water) at noticeable speed

7. **Applied Force** (For objects being pushed/pulled)
   - Direction: Direction of the applied force
   - Start: Point of force application
   - End: Direction the force is applied
   - Color: "yellow"
   - Only include if there's visible force application (hand pushing, etc.)

**Selection Guidelines:**
- Include only vectors that are physically relevant to this specific scenario
- Typical scenarios have 2-4 vectors (don't force all vector types)
- Vector lengths should reflect relative magnitudes when possible
- Ensure vectors originate from physically meaningful locations

#### Step 4: Calculate Vector Coordinates
For each identified vector, calculate normalized coordinates:

**Start Point:**
- Usually the object center: `[object_center_x, object_center_y]`
- Exception: Contact forces may start at contact point
- Normalized: 0.0 to 1.0 range

**End Point:**
- Calculate based on vector direction and reasonable magnitude
- Formula: `end = start + direction × magnitude_scale`
- Ensure end coordinates remain in valid range (0.0 to 1.0)
- If calculation exceeds bounds, scale down to fit within frame

**Direction Guidelines:**
- Gravity: end_y > start_y (downward)
- Velocity: based on motion direction (horizontal, upward, diagonal)
- Normal: perpendicular to surface, typically upward
- Keep vectors visually clear (not too short or too long)

**Magnitude Representation:**
- Vector length in normalized space should be 0.05 to 0.3 for visibility
- Adjust based on relative importance: stronger forces = longer vectors
- Maintain physical relationships (e.g., normal force magnitude often similar to gravity)

### Output Format Requirements

**CRITICAL:** Return ONLY a valid JSON object. NO markdown code blocks. NO ```json``` wrappers. NO explanatory text.

**JSON Structure:**
{
    "object_center": [x, y],
    "vectors": [
        {
            "name": "Vector Name",
            "start": [x, y],
            "end": [x, y],
            "color": "color_name"
        }
    ]
}

**Field Specifications:**

- **object_center**: `[float, float]`
  - Normalized x, y coordinates of the main object's center
  - Range: [0.0, 1.0] for both x and y
  - Example: `[0.5, 0.3]` means center-horizontal, upper third vertically

- **vectors**: `Array of vector objects`
  - Each vector has exactly 4 fields: name, start, end, color
  - Minimum: 1 vector (at least gravity for most scenarios)
  - Maximum: 6 vectors (avoid overcrowding)
  - Order: List most important/dominant vectors first

- **name**: `string`
  - Physics-accurate name: "Gravity", "Velocity", "Normal Force", "Friction", "Tension", "Air Resistance", "Applied Force"
  - Use standard physics terminology
  - Capitalize properly

- **start**: `[float, float]`
  - Normalized x, y coordinates where vector arrow begins
  - Typically object_center, but may differ for contact forces
  - Range: [0.0, 1.0]

- **end**: `[float, float]`
  - Normalized x, y coordinates where vector arrow points
  - Must differ from start (non-zero length)
  - Range: [0.0, 1.0]
  - Direction must be physically accurate for the named vector

- **color**: `string`
  - Use standard color names for consistency:
    - "red" → Gravity
    - "green" → Velocity
    - "blue" → Normal Force
    - "orange" → Friction
    - "purple" → Tension
    - "cyan" → Air Resistance
    - "yellow" → Applied Force
  - Lowercase, no special characters

### Validation Rules

Before outputting, verify:

✅ **JSON Validity:**
- Proper syntax: braces, brackets, commas, quotes
- No trailing commas
- Double quotes for strings (not single quotes)

✅ **Coordinate Constraints:**
- All x and y values are floats in range [0.0, 1.0]
- object_center is within bounds
- All vector start and end points are within bounds
- Vector start ≠ end (non-zero length)

✅ **Physical Accuracy:**
- Gravity points downward (end_y > start_y)
- Velocity direction matches apparent motion
- Normal force perpendicular to surface
- Vector names match actual physics principles present
- No contradictory vectors (e.g., velocity both left and right)

✅ **Completeness:**
- Object center is specified
- At least 1 vector is present
- Each vector has all 4 required fields
- Color names match standard mapping

### Example Outputs

**Example 1: Ball in Projectile Motion**
{
    "object_center": [0.6, 0.4],
    "vectors": [
        {
            "name": "Velocity",
            "start": [0.6, 0.4],
            "end": [0.75, 0.35],
            "color": "green"
        },
        {
            "name": "Gravity",
            "start": [0.6, 0.4],
            "end": [0.6, 0.55],
            "color": "red"
        }
    ]
}

**Example 2: Block Sliding on Incline**
{
    "object_center": [0.5, 0.6],
    "vectors": [
        {
            "name": "Gravity",
            "start": [0.5, 0.6],
            "end": [0.5, 0.75],
            "color": "red"
        },
        {
            "name": "Normal Force",
            "start": [0.5, 0.6],
            "end": [0.45, 0.48],
            "color": "blue"
        },
        {
            "name": "Friction",
            "start": [0.5, 0.6],
            "end": [0.42, 0.58],
            "color": "orange"
        },
        {
            "name": "Velocity",
            "start": [0.5, 0.6],
            "end": [0.58, 0.62],
            "color": "green"
        }
    ]
}

**Example 3: Pendulum at Swing**
{
    "object_center": [0.4, 0.7],
    "vectors": [
        {
            "name": "Tension",
            "start": [0.4, 0.7],
            "end": [0.45, 0.5],
            "color": "purple"
        },
        {
            "name": "Gravity",
            "start": [0.4, 0.7],
            "end": [0.4, 0.85],
            "color": "red"
        },
        {
            "name": "Velocity",
            "start": [0.4, 0.7],
            "end": [0.52, 0.72],
            "color": "green"
        }
    ]
}

### Error Handling

**If the image is unclear or ambiguous:**
- Make best reasonable interpretation based on visual cues
- Prioritize most likely physics scenario
- Include core vectors that would apply to most interpretations (gravity almost always applies)

**If no clear moving object:**
- Identify most prominent physical object
- Include relevant static forces (gravity, normal force if on surface)
- Omit velocity vector if no motion is apparent

**If image quality is too poor:**
- Still attempt analysis with available information
- Focus on gross features (large objects, clear surfaces)
- Reduce vector count to most certain forces only

### Coordinate System Reference
```
Image Frame (Normalized Coordinates):
(0.0, 0.0) ─────────────────── (1.0, 0.0)
    │                                │
    │         Screen/Image           │
    │                                │
    │        (0.5, 0.5) = Center     │
    │                                │
(0.0, 1.0) ─────────────────── (1.0, 1.0)

X-axis: Left (0.0) → Right (1.0)
Y-axis: Top (0.0) → Bottom (1.0)
```

### Physical Direction Conventions

- **Downward (Gravity)**: Increase Y (e.g., start_y=0.4 → end_y=0.5)
- **Upward (Normal, Thrust)**: Decrease Y (e.g., start_y=0.6 → end_y=0.5)
- **Rightward**: Increase X (e.g., start_x=0.4 → end_x=0.5)
- **Leftward**: Decrease X (e.g., start_x=0.6 → end_x=0.5)

---

**Now analyze the provided image and return your JSON output.**
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