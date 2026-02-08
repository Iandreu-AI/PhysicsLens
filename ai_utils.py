import google.generativeai as genai
import streamlit as st
from PIL import Image
import json
import cv2
import numpy as np
import re
import ast
from ultralytics import YOLO

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

# Global YOLO model cache
_yolo_model = None

def _get_yolo_model():
    """Get or initialize YOLO model (singleton pattern)."""
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO('yolov8n.pt')
        _yolo_model.conf = 0.15
    return _yolo_model


# ============================================================================
# STEP 1: DETERMINISTIC ANCHOR DETECTION WITH HUMAN REJECTION
# ============================================================================

def detect_object_centroid(frame, method='yolo_strict'):
    """
    Calculate the TRUE centroid of the primary moving object.
    NOW WITH HUMAN REJECTION LOGIC - Will NEVER track persons.
    
    Args:
        frame: BGR image (numpy array)
        method: 'yolo_strict' (RECOMMENDED), 'contour', 'color_threshold'
    
    Returns:
        (cx, cy): Pixel coordinates of centroid
        bbox: Bounding box [x, y, w, h] for debugging
    """
    h, w = frame.shape[:2]
    frame_area = h * w
    
    if method == 'yolo_strict':
        # === HUMAN REJECTION PIPELINE ===
        model = _get_yolo_model()
        results = model(frame, verbose=False)[0]
        
        if len(results.boxes) == 0:
            # Fallback: geometric center
            return (w // 2, h // 2), [w//4, h//4, w//2, h//2]
        
        # Filter candidates with 3-layer rejection
        candidates = []
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            
            # === LAYER 1: CLASS BLACKLIST ===
            # CRITICAL: Immediately reject Class 0 (Person)
            if cls_id == 0:
                continue
            
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            box_w = x2 - x1
            box_h = y2 - y1
            box_area = box_w * box_h
            area_ratio = box_area / frame_area
            aspect_ratio = box_w / max(box_h, 1)
            
            # === LAYER 2: AREA FILTER ===
            # Reject large boxes (>25% = likely human) and tiny boxes (<0.5% = noise)
            if area_ratio < 0.005 or area_ratio > 0.25:
                continue
            
            # === LAYER 3: ASPECT RATIO CHECK ===
            # Reject tall rectangles (human signature: height >> width)
            if aspect_ratio < 0.3 or aspect_ratio > 3.0:
                continue
            
            # === SCORING SYSTEM ===
            # Prioritize sports balls (Class 32) and other physics objects
            score = conf
            
            # Physics objects whitelist
            physics_classes = {32, 39, 40, 41, 45, 29, 47, 49, 33}
            if cls_id in physics_classes:
                score *= 2.0
            
            # Triple priority for sports balls (baseball, tennis ball, etc.)
            if cls_id == 32:
                score *= 3.0
            
            candidates.append({
                'box': [int(x1), int(y1), int(box_w), int(box_h)],
                'cx': int((x1 + x2) / 2),
                'cy': int((y1 + y2) / 2),
                'score': score
            })
        
        # Select best candidate
        if candidates:
            best = max(candidates, key=lambda x: x['score'])
            return (best['cx'], best['cy']), best['box']
        
        # No valid objects found - use fallback
        return (w // 2, h // 2), [w//4, h//4, w//2, h//2]
    
    elif method == 'contour':
        # Original contour-based detection (fallback)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Get bounding box
                x, y, bw, bh = cv2.boundingRect(largest)
                return (cx, cy), [x, y, bw, bh]
    
    elif method == 'color_threshold':
        # Detect bright/colored objects
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for bright colors (adjust ranges as needed)
        lower = np.array([0, 50, 50])
        upper = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Find center of mass
        moments = cv2.moments(mask, True)
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            
            # Estimate bbox from mask
            coords = cv2.findNonZero(mask)
            x, y, bw, bh = cv2.boundingRect(coords)
            return (cx, cy), [x, y, bw, bh]
    
    # Fallback: geometric center
    cx, cy = w // 2, h // 2
    return (cx, cy), [w//4, h//4, w//2, h//2]

# ============================================================================
# STEP 2: COORDINATE NORMALIZATION SYSTEM
# ============================================================================

class CoordinateTransformer:
    """Handles all coordinate transformations between AI grid and pixel space."""
    
    def __init__(self, frame_width, frame_height, ai_grid_size=1000):
        self.W = frame_width
        self.H = frame_height
        self.GRID = ai_grid_size
    
    def normalize_to_grid(self, pixel_x, pixel_y):
        """Convert pixel coordinates to AI grid [0-1000]."""
        grid_x = int((pixel_x / self.W) * self.GRID)
        grid_y = int((pixel_y / self.H) * self.GRID)
        return grid_x, grid_y
    
    def denormalize_from_grid(self, grid_x, grid_y):
        """Convert AI grid coordinates back to pixels."""
        pixel_x = int((grid_x / self.GRID) * self.W)
        pixel_y = int((grid_y / self.GRID) * self.H)
        return pixel_x, pixel_y
    
    def denormalize_from_ratio(self, ratio_x, ratio_y):
        """Convert [0.0-1.0] ratios to pixels."""
        pixel_x = int(ratio_x * self.W)
        pixel_y = int(ratio_y * self.H)
        return pixel_x, pixel_y

# --- CORE FUNCTIONS ---

def get_physics_vectors(frame, prev_frame=None, frame_index=0):
    h, w = frame.shape[:2]
    transformer = CoordinateTransformer(w, h)
    
    # STEP 1: Detect centroid using CV WITH HUMAN REJECTION
    centroid_px, bbox = detect_object_centroid(frame, method='yolo_strict')
    cx_px, cy_px = centroid_px
    
    # STEP 2: Convert to AI grid coordinates
    cx_grid, cy_grid = transformer.normalize_to_grid(cx_px, cy_px)
    
    # STEP 3: Ask AI for force directions (using the grid coordinates as hint)
    try:
        model = genai.GenerativeModel('gemini-3-pro-preview')
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        prompt_text = """
# SYSTEM IDENTITY
**Role:** Newtonian Telemetry Engine (Physics Vector Calculator)
**Capability:** High-fidelity analysis of kinematic forces based on visual data and motion context.

## INPUT DATA
You are provided with a snapshot of a physics experiment.
*   **Object Center:** [{{cx_grid}}, {{cy_grid}}] (This is your anchor point).
*   **Motion Context:** "{{motion_description}}" (CRITICAL: Use this to determine Velocity direction).
*   **Object Type:** "{{object_name}}"

# COORDINATE SYSTEM RULES
*   **Canvas:** Screen Space (Y-Axis points DOWN).
*   **Rotation:**
    *   $0^\circ$ = Right (Positive X)
    *   $90^\circ$ = Down (Positive Y)
    *   $180^\circ$ = Left (Negative X)
    *   $270^\circ$ = Up (Negative Y)
*   **Magnitude:** Visual scale 0-300. (Standard Gravity = 150).

# PHYSICS INFERENCE LOGIC
Apply these rules to generate the vector list:

1.  **Gravity ($F_g$):**
    *   *Condition:* ALWAYS applied for terrestrial objects.
    *   *Angle:* Exactly $90^\circ$ (Down).
    *   *Magnitude:* Fixed at 150.
    *   *Color:* `#EF4444` (Red).

2.  **Velocity ($v$):**
    *   *Condition:* Active if `Motion Context` implies movement.
    *   *Angle:* Derive strictly from the `Motion Context` (e.g., "Moving Up-Right" $\approx 315^\circ$).
    *   *Magnitude:* 150 (Slow) to 300 (Fast).
    *   *Color:* `#22C55E` (Green).

3.  **Normal Force ($F_N$):**
    *   *Condition:* Active ONLY if the object is clearly resting on or sliding along a surface.
    *   *Angle:* Perpendicular to surface (usually $270^\circ$ Up).
    *   *Magnitude:* Equal to Gravity (150) unless on a slope.
    *   *Color:* `#3B82F6` (Blue).

4.  **Friction ($F_f$):**
    *   *Condition:* Active ONLY if "Sliding" or "Rolling" on a surface.
    *   *Angle:* Exactly $180^\circ$ opposite to Velocity.
    *   *Color:* `#F59E0B` (Orange).

# OUTPUT SCHEMA
Return **ONLY** the raw JSON object. No markdown formatting. No code blocks.

{
  "vectors": [
    {
      "name": "String",
      "color": "HexCode",
      "angle_degrees": Integer,
      "magnitude_grid": Integer
    }
  ]
}
"""

        response = model.generate_content([prompt_text, pil_img])
        clean_text = response.text.strip()
        clean_text = re.sub(r'^```(?:json)?\s*\n?', '', clean_text)
        clean_text = re.sub(r'\n?```\s*$', '', clean_text)
        
        ai_data = json.loads(clean_text)
        
    except Exception as e:
        print(f"AI call failed: {e}. Using default gravity vector.")
        ai_data = {
            "vectors": [
                {"name": "Gravity", "color": "#3B82F6", "angle_degrees": 90, "magnitude_grid": 150}
            ]
        }
    
    # STEP 4: Convert AI vectors to pixel coordinates
    vectors_px = []
    
    for vec in ai_data.get('vectors', []):
        angle_rad = np.deg2rad(vec['angle_degrees'])
        mag_grid = vec['magnitude_grid']
        
        # Calculate end point in grid space
        end_x_grid = cx_grid + int(mag_grid * np.cos(angle_rad))
        end_y_grid = cy_grid + int(mag_grid * np.sin(angle_rad))
        
        # Clamp to grid bounds
        end_x_grid = np.clip(end_x_grid, 0, 1000)
        end_y_grid = np.clip(end_y_grid, 0, 1000)
        
        # Convert back to pixel space
        end_x_px, end_y_px = transformer.denormalize_from_grid(end_x_grid, end_y_grid)
        
        vectors_px.append({
            'name': vec['name'],
            'start_px': (cx_px, cy_px),  # ALWAYS anchored to detected centroid
            'end_px': (end_x_px, end_y_px),
            'color': vec['color'],
            'magnitude': mag_grid
        })
    
    return {
        'centroid_px': centroid_px,
        'bbox': bbox,
        'vectors': vectors_px,
        'frame_index': frame_index
    }
    
    # STEP 4: Convert AI vectors to pixel coordinates
    vectors_px = []
    
    for vec in ai_data.get('vectors', []):
        angle_rad = np.deg2rad(vec['angle_degrees'])
        mag_grid = vec['magnitude_grid']
        
        # Calculate end point in grid space
        end_x_grid = cx_grid + int(mag_grid * np.cos(angle_rad))
        end_y_grid = cy_grid + int(mag_grid * np.sin(angle_rad))
        
        # Clamp to grid bounds
        end_x_grid = np.clip(end_x_grid, 0, 1000)
        end_y_grid = np.clip(end_y_grid, 0, 1000)
        
        # Convert back to pixel space
        end_x_px, end_y_px = transformer.denormalize_from_grid(end_x_grid, end_y_grid)
        
        vectors_px.append({
            'name': vec['name'],
            'start_px': (cx_px, cy_px),  # ALWAYS anchored to detected centroid
            'end_px': (end_x_px, end_y_px),
            'color': vec['color'],
            'magnitude': mag_grid
        })
    
    return {
        'centroid_px': centroid_px,
        'bbox': bbox,
        'vectors': vectors_px,
        'frame_index': frame_index
    }


def draw_vectors_with_debug(frame, vector_data, show_debug=True):
    """
    Draw vectors with visual debugging aids.
    
    Args:
        frame: BGR image to draw on
        vector_data: Output from get_physics_vectors()
        show_debug: If True, draw centroid marker and bounding box
    """
    output = frame.copy()
    cx_px, cy_px = vector_data['centroid_px']
    
    # Debug: Draw centroid anchor point (BLUE DOT)
    if show_debug:
        cv2.circle(output, (cx_px, cy_px), 8, (255, 0, 0), -1)  # Blue filled circle
        cv2.circle(output, (cx_px, cy_px), 10, (255, 255, 255), 2)  # White outline
    
    # Draw vectors
    for vec in vector_data['vectors']:
        start = vec['start_px']
        end = vec['end_px']
        
        # Convert hex color to BGR
        hex_color = vec['color'].lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        color_bgr = (b, g, r)
        
        # Draw arrow
        cv2.arrowedLine(output, start, end, color_bgr, 3, tipLength=0.3)
        
        # Draw label
        label_pos = (end[0] + 10, end[1] - 10)
        cv2.putText(output, vec['name'], label_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
    
    return output


# ============================================================================
# JSON PARSING HELPERS
# ============================================================================

def _clean_json_response(text):
    """Remove markdown fences and clean JSON response."""
    clean_text = text.strip()
    clean_text = re.sub(r'^```(?:json)?\s*\n?', '', clean_text)
    clean_text = re.sub(r'\n?```\s*$', '', clean_text)
    return clean_text

def _robust_json_load(text):
    """Attempt to parse JSON with multiple strategies."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to fix common issues
        try:
            # Remove trailing commas
            fixed = re.sub(r',\s*}', '}', text)
            fixed = re.sub(r',\s*]', ']', fixed)
            return json.loads(fixed)
        except:
            return None


def analyze_physics_with_gemini(keyframes, difficulty="High School Physics"):
    """Analyzes video frames for educational text."""
    try:
        # Use Pro for better text analysis
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        if not keyframes:
            return {"error": "No frames"}
            
        mid_idx = len(keyframes) // 2
        ref_frame_rgb = keyframes[mid_idx]['frame']
        pil_image = Image.fromarray(ref_frame_rgb)

        prompt_text = f"""
        # IDENTITY & MISSION
        You are **VectorForge Pro** - an expert physics visualization engine with elite precision in vector rendering.

        **Core Mission:** Analyze visual frames and generate mathematically rigorous, physically accurate vector overlays with pixel-perfect geometry.

        ---

        # COORDINATE SYSTEM SPECIFICATION

        ## Grid Properties
        - **Canvas Size:** 1000×1000 grid units
        - **Origin Point:** `(0, 0)` at **Top-Left corner**
        - **Axis Orientation:**
        - **+X Axis:** Extends RIGHT (increasing horizontal)
        - **+Y Axis:** Extends DOWN (aligned with gravitational direction)
        - **Unit System:** Pure grid coordinates (dimensionless)

        ## Anchor Point Protocol
        - **Object Center:** `[{grid_x}, {grid_y}]`
        - All vectors **originate** from this anchor point unless explicitly specified otherwise
        - Boundary enforcement: All coordinates MUST satisfy `0 ≤ x,y ≤ 1000`

        ---

        # MAGNITUDE CALIBRATION SYSTEM

        ## Base Reference Scale
        - **Gravitational Standard:** `M_g = 150` grid units (canonical reference)
        - **Scaling Law:** All force magnitudes calculated relative to `M_g`

        ## Dynamic Magnitude Rules
        1. **Proportional Forces:** Scale linearly with `M_g`
        - Normal force on flat surface: `M_N = M_g`
        - Weight component on θ° slope: `M_x = M_g × sin(θ)`

        2. **Velocity Vectors:** Scale based on motion context
        - Slow motion: `0.5 × M_g` to `1.0 × M_g`
        - Medium motion: `1.0 × M_g` to `2.0 × M_g`
        - High-speed: `2.0 × M_g` to `4.0 × M_g`

        3. **Friction Forces:** Calculate from context
        - Kinetic friction: `μ_k × M_N` (typical μ_k = 0.2-0.8)
        - Scale to grid: multiply by `(M_g / 150)` for display

        ---

        # VECTOR GENERATION LOGIC ENGINE

        ## Conditional Rendering Protocol
        **CRITICAL:** Only generate vectors that satisfy ALL conditions in their trigger ruleset.

        ### Vector Type 1: Gravitational Force (`F_g`)
        **Trigger Conditions:**
        - ✓ Object exists in planetary environment (NOT deep space/orbit)
        - ✓ NOT explicitly stated to be "weightless"

        **Vector Properties:**
        {{
        "name": "Gravity (F_g)",
        "start_x": {grid_x},
        "start_y": {grid_y},
        "end_x": {grid_x},
        "end_y": {grid_y} + 150,
        "magnitude": 150,
        "angle_deg": 270,  // Straight down
        "color": "#FF0000",
        "style": "solid"
        }}
        ```

        ---

        ### Vector Type 2: Velocity (`v`)
        **Trigger Conditions:**
        - ✓ Object shows motion blur, trajectory path, or positional change
        - ✓ OR context indicates movement (rolling, flying, sliding, falling)
        - ✗ EXCLUDE if object is stationary/at rest

        **Directional Logic:**
        - **Direction:** Tangent to motion path
        - **Freefall:** Angle = 270° (straight down)
        - **Projectile:** Angle depends on trajectory phase
        - **Rolling:** Parallel to surface contact

        **Vector Properties:**
        ```javascript
        {{
        "name": "Velocity (v)",
        "start_x": {grid_x},
        "start_y": {grid_y},
        "end_x": {grid_x} + (magnitude × cos(angle_rad)),
        "end_y": {grid_y} + (magnitude × sin(angle_rad)),
        "magnitude": <contextual>,
        "angle_deg": <calculated>,
        "color": "#00FF00",
        "style": "solid"
        }}
        ```

        ---

        ### Vector Type 3: Normal Force (`F_N`)
        **Trigger Conditions:**
        - ✓ Object is in contact with a solid surface
        - ✓ Surface provides reactive support force
        - ✗ EXCLUDE if object is airborne/in freefall

        **Directional Algorithm:**
        1. Identify contact surface orientation
        2. Calculate perpendicular (90° outward from surface)
        3. For flat surface: Angle = 90° (straight up)
        4. For θ° incline: Angle = (90° + θ)

        **Magnitude Calculation:**
        - Flat surface: `M_N = M_g`
        - Inclined surface: `M_N = M_g × cos(θ)`

        **Vector Properties:**
        ```javascript
        {{
        "name": "Normal Force (F_N)",
        "start_x": {grid_x},
        "start_y": {grid_y},
        "end_x": {grid_x} + (magnitude × cos(angle_rad)),
        "end_y": {grid_y} + (magnitude × sin(angle_rad)),
        "magnitude": <calculated>,
        "angle_deg": <perpendicular_to_surface>,
        "color": "#0000FF",
        "style": "solid"
        }}
        ```

        ---

        ### Vector Type 4: Friction Force (`F_f`)
        **Trigger Conditions:**
        - ✓ Object is moving relative to contact surface (kinetic)
        - ✓ OR has tendency to move while at rest (static)
        - ✗ EXCLUDE if no surface contact OR no motion/motion tendency

        **Directional Law:**
        - **Direction:** Exactly opposite to velocity vector
        - **Mathematical:** `angle_friction = angle_velocity + 180°`

        **Magnitude Estimation:**
        - Kinetic: `0.3 × M_N` to `0.8 × M_N` (context dependent)
        - Static: Up to `1.0 × M_N` (maximum before motion)

        **Vector Properties:**
        ```javascript
        {{
        "name": "Friction (F_f)",
        "start_x": {grid_x},
        "start_y": {grid_y},
        "end_x": {grid_x} + (magnitude × cos(angle_rad)),
        "end_y": {grid_y} + (magnitude × sin(angle_rad)),
        "magnitude": <calculated>,
        "angle_deg": <opposite_to_velocity>,
        "color": "#FFA500",
        "style": "dashed"
        }}
        ```

        ---

        ### Vector Type 5: Weight Components (`W_x`, `W_y`)
        **STRICT Trigger Conditions:**
        - ✓ Object is on an inclined plane/ramp (angle θ > 5°)
        - ✓ Gravitational decomposition is relevant to analysis
        - ✗ **NEVER generate on flat surfaces (θ = 0°)**
        - ✗ **NEVER generate in freefall scenarios**

        **Component Calculations:**
        - **Parallel Component (`W_x`):**
        - Magnitude: `M_g × sin(θ)`
        - Direction: Down the slope (along surface)
        
        - **Perpendicular Component (`W_y`):**
        - Magnitude: `M_g × cos(θ)`
        - Direction: Into the slope (perpendicular)

        **Vector Properties:**
        ```javascript
        // Parallel component
        {{
        "name": "Weight Parallel (W_x)",
        "start_x": {grid_x},
        "start_y": {grid_y},
        "end_x": {grid_x} + (M_g × sin(θ) × cos(slope_angle)),
        "end_y": {grid_y} + (M_g × sin(θ) × sin(slope_angle)),
        "magnitude": <M_g × sin(θ)>,
        "angle_deg": <along_slope>,
        "color": "#FF00FF",
        "style": "dashed"
        }}

        // Perpendicular component
        {{
        "name": "Weight Perpendicular (W_y)",
        "start_x": {grid_x},
        "start_y": {grid_y},
        "end_x": {grid_x} + (M_g × cos(θ) × cos(perp_angle)),
        "end_y": {grid_y} + (M_g × cos(θ) × sin(perp_angle)),
        "magnitude": <M_g × cos(θ)>,
        "angle_deg": <into_slope>,
        "color": "#FFFF00",
        "style": "dashed"
        }}
        ```

        ---

        # COMPUTATIONAL REQUIREMENTS

        ## Angle-to-Coordinate Conversion
        **Standard Formula:**
        end_x = start_x + (magnitude × cos(angle_radians))
        end_y = start_y + (magnitude × sin(angle_radians))

        **Angle Convention:**
        - 0° = East (+X direction)
        - 90° = South (+Y direction)
        - 180° = West (-X direction)
        - 270° = North (-Y direction)

        ## Integer Rounding Protocol
        - Calculate in floating point
        - Round final coordinates: `Math.round(value)`
        - Ensure integers in JSON output

        ## Boundary Validation
        ```javascript
        // Clamp all coordinates
        end_x = Math.max(0, Math.min(1000, end_x));
        end_y = Math.max(0, Math.min(1000, end_y));
        ```

        ---

        # OUTPUT SPECIFICATION

        ## JSON Structure (Strict Schema)
        ```json
        {{
        "object_name": "string - Brief descriptor of analyzed object",
        "physics_state": "string - Current mechanical state (e.g., 'Freefall', 'Sliding on Incline', 'Static Equilibrium')",
        "surface_angle_deg": "number|null - Angle of contact surface if applicable",
        "analysis_notes": "string - Brief reasoning for vector choices",
        "vectors": [
            {{
            "name": "string - Force/vector identifier",
            "start_x": "integer - Origin X coordinate [0-1000]",
            "start_y": "integer - Origin Y coordinate [0-1000]",
            "end_x": "integer - Terminal X coordinate [0-1000]",
            "end_y": "integer - Terminal Y coordinate [0-1000]",
            "magnitude": "number - Length in grid units",
            "angle_deg": "number - Direction in degrees [0-360]",
            "color": "string - Hex color code",
            "style": "string - 'solid' or 'dashed'"
            }}
        ]
        }}
        ```

        ## Quality Assurance Checklist
        Before outputting, verify:
        - [ ] All coordinates are integers within [0, 1000]
        - [ ] Each vector satisfies its trigger conditions
        - [ ] Magnitudes follow calibration rules
        - [ ] Angles are mathematically consistent
        - [ ] No contradictory vectors (e.g., Normal + Freefall)
        - [ ] JSON is valid and parseable

        ---

        # COGNITIVE WORKFLOW

        ## Step-by-Step Analysis Protocol
        1. **Scene Recognition:** Identify object and environmental context
        2. **State Classification:** Determine physics state (static/kinetic/airborne)
        3. **Surface Detection:** Locate contact points and surface angles
        4. **Trigger Evaluation:** Check each vector type's conditions
        5. **Magnitude Calculation:** Apply scaling laws
        6. **Geometric Computation:** Convert angles to coordinates
        7. **Validation:** Run quality checks
        8. **JSON Generation:** Output formatted data

        ## Example Reasoning Chain
        """

        response = model.generate_content(
            [prompt_text, pil_image],
            safety_settings=safety_settings 
        )
        
        clean_text = _clean_json_response(response.text)
        result = _robust_json_load(clean_text)
        
        if result is None:
             return {
                "error": "AI Response Parse Error",
                "explanation": "The AI analysis could not be processed."
            }
        return result

    except Exception as e:
        return {
            "error": "API Error", 
            "explanation": f"Connection failed. Error: {str(e)}"
        }

def get_chat_response_stream(chat_history, user_input, vector_context, text_context = None):
    """
    Generates a streaming response for the Physics Tutor chat.
    """
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')

         # 1. Format the Data Context for the AI
        data_report = "NO EXPERIMENT DATA AVAILABLE."
        
        # Check if text_context exists (this contains the data from the UI)
        if text_context and isinstance(text_context, dict) and "error" not in text_context:
            
            # --- EXTRACT KEY METRICS (Matching your UI) ---
            subject = text_context.get('main_object', 'Unknown Object')
            motion_type = text_context.get('motion_type', 'Unknown Motion')
            principle = text_context.get('physics_principle', 'Analyzing...')
            
            # Extract additional details for context
            forces = ", ".join(text_context.get('active_forces', []))
            formula = text_context.get('key_formula', 'N/A')
            explanation = text_context.get('explanation', '')

            # Build the "Lab Report" that the AI reads silently
            data_report = f"""
            === VISUAL ANALYSIS LOG ===
            [KEY METRICS]
            Subject:   {subject}
            Type:      {motion_type}
            Principle: {principle}

            [DETAILED TELEMETRY]
            Active Forces: {forces}
            Governing Formula: {formula}
            System Notes: {explanation}
            """
        
        system_instruction = f"""
        # ROLE: Professor Lens
        You are an expert Physics Tutor. You have access to real-time analysis of a video the user uploaded.

        # CURRENT EXPERIMENT DATA
        You are currently looking at the following analysis:
        {data_report}

        # INSTRUCTIONS
        1. **Context Awareness:** The user sees the 'Key Metrics' listed above. If they ask "What is this?", refer to the Subject and Principle.
        2. **Math Support:** If the user asks about the math, explain the 'Governing Formula' provided in the data.
        3. **Tone:** Helpful, encouraging, and scientific.
        4. **Format:** Keep answers concise. Use inline LaTeX for math (e.g., $F=ma$).
        
        # USER QUERY
        "{user_input}"
        """
        
        # 3. Build Chat History
        history_gemini = []
        
        # Seed with system instruction
        history_gemini.append({
            "role": "user",
            "parts": [system_instruction]
        })
        history_gemini.append({
            "role": "model",
            "parts": ["Understood. I have the Key Metrics for this experiment. I am ready to answer."]
        })
        
        # Append previous conversation (Last 6 messages for context)
        if chat_history:
            for msg in chat_history[-6:]: 
                role = "user" if msg["role"] == "user" else "model"
                history_gemini.append({"role": role, "parts": [msg["content"]]})
            
        # 4. Generate Response
        chat = model.start_chat(history=history_gemini)
        response = chat.send_message(user_input, stream=True)
        
        for chunk in response:
            if chunk.text:
                yield chunk.text
                
    except Exception as e:
        yield f"I'm having trouble accessing the metrics right now. Error: {str(e)}"