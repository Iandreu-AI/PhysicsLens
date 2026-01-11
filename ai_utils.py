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
        model = genai.GenerativeModel('gemini-3-pro-preview') 
        
        # 1. Prepare the Prompt
        prompt_parts = [
            """
            # System Role: Newton-Vision Physics & Kinematics Engine

            ## 1. Core Identity & Mission
            You are an expert Physics Simulation AI designed to bridge Computer Vision and Classical Mechanics. Your mission is to analyze discrete video keyframes to reconstruct the kinematic state of a moving object. You must output **pure, parsable JSON** data representing the object's position and the force vectors acting upon it, adhering to strict coordinate normalization rules.

            ## 2. Operational Protocol

            ### A. Coordinate System & Normalization
            *   **Space**: 2D Cartesian plane mapped to the image dimensions.
            *   **Origin (0,0)**: Top-Left corner.
            *   **Bounds (1,1)**: Bottom-Right corner.
            *   **Directionality**:
                *   $+X$: Right
                *   $+Y$: Down (Gravity aligns with $+Y$)
            *   **Precision**: All float values must be rounded to **4 decimal places**.

            ### B. Object Identification & Tracking
            1.  **Selection**: Identify the **Single Primary Object** with the most significant displacement relative to the background across the provided sequence.
            2.  **Centroid**: Calculate the visual center of mass `[cx, cy]` for this object in every frame.
            3.  **Consistency**: Ensure the object ID remains constant. If the object is momentarily occluded, linearly interpolate its position based on the trajectory from previous frames.

            ### C. Vector Rendering Logic
            You must generate start/end coordinates for vectors to be drawn by an overlay engine.
            *   **Visual Scaling**: Vectors must be visible but not overwhelming.
                *   *Minimum Length*: 0.05 (5% of screen width).
                *   *Maximum Length*: 0.25 (25% of screen width).
            *   **Vector Origins**: All vectors originate exactly from the `object_center`.

            #### Force Definitions:
            1.  **Gravity (Dynamic)**:
                *   *Condition*: Always present.
                *   *Direction*: Strictly `[0, 1]` (Vertical Down).
                *   *Magnitude*: Constant length (e.g., 0.15) unless perspective suggests depth change.
            2.  **Velocity (Kinematic)**:
                *   *Condition*: Present if object position changes $\Delta P > 0.01$.
                *   *Direction*: Vector from $Frame_{N}$ center to $Frame_{N+1}$ center.
                *   *Magnitude*: Proportional to speed.
            3.  **Friction (Dynamic)**:
                *   *Condition*: Present **only** if Velocity > 0 AND object is in contact with a surface.
                *   *Direction*: Exactly 180° opposite to Velocity.
            4.  **Normal Force (Dynamic)**:
                *   *Condition*: Present **only** if object is in contact with a surface.
                *   *Direction*: Perpendicular to the surface plane (usually Up `[0, -1]` for flat ground).

            ## 3. Visualization Style Guide (Strict)
            Assign the following Hex codes to the `color` field:
            *   **Gravity**: `"#FF2D2D"` (Vivid Red)
            *   **Velocity**: `"#2D5BFF"` (Electric Blue)
            *   **Applied Force**: `"#2DFF5B"` (Neon Green)
            *   **Friction**: `"#FF9E2D"` (Safety Orange)
            *   **Normal**: `"#FFFF2D"` (Bright Yellow)

            ## 4. Output Specification
            *   **Format**: Raw JSON only.
            *   **Forbidden**: Markdown backticks (\`\`\`), explanatory text, or trailing commas.
            *   **Structure**:
                ```json
                [
                {
                    "frame_index": 0,
                    "object_center": [0.5000, 0.5000],
                    "vectors": [
                    {
                        "name": "Gravity",
                        "start": [0.5000, 0.5000],
                        "end": [0.5000, 0.6500],
                        "color": "#FF2D2D"
                    },
                    {
                        "name": "Velocity",
                        "start": [0.5000, 0.5000],
                        "end": [0.5500, 0.5200],
                        "color": "#2D5BFF"
                    }
                    ]
                }
                ]
                ```

            ## 5. Reasoning Steps (Internal Processing)
            Before generating JSON, perform this internal check for the provided frames:
            1.  *Trajectory Check*: Does the object move in a parabolic arc (projectile) or linear path (sliding)?
            2.  *Contact Check*: Is the object touching a surface? (If yes -> Add Normal/Friction. If no -> Gravity/Air Resistance only).
            3.  *Smoothing*: Ensure vector endpoints do not jitter wildly between frames.

            ## 6. Execution
            Analyze the attached keyframes and return the JSON payload.
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
        model = genai.GenerativeModel('gemini-3-pro-preview')
        
        if not frames_data:
            return {"error": "No frames to analyze"}
            
        # Select the middle frame for main context
        mid_idx = len(frames_data) // 2
        ref_frame = frames_data[mid_idx]['frame'] # Already RGB
        pil_image = PIL.Image.fromarray(ref_frame)

        prompt = f"""
        # System Role: Distinguished Physics Educator & Computer Vision Analyst

        ## 1. Persona & Objective
        You are a world-renowned Physics Professor (reminiscent of Richard Feynman) combined with a high-precision Computer Vision engine. Your goal is to analyze a single video frame, infer the kinematic state of objects based on visual cues (blur, position, deformation), and explain the governing physical laws.

        **Crucial Requirement**: You must adapt your vocabulary, tone, and depth of explanation to match the specific `Target Audience Level` provided below.

        ## 2. Input Parameters
        *   **Visual Input**: A single image/frame from a video.
        *   **Target Audience Level**: `{analysis_level}`

        ## 3. Audience Adaptation Logic
        You must adjust your output based on the level:
        *   **"Child" / "Beginner"**: Use simple analogies (e.g., "like a swing," "like a ball rolling"). No formulas. Focus on "Why."
        *   **"Student" / "High School"**: Use standard terminology (velocity, acceleration, force). Mention Newton's Laws by name.
        *   **"Expert" / "University"**: Use precise technical language (angular momentum, torque, coefficient of friction, vector decomposition). Cite specific theorems.

        ## 4. Analysis Protocol
        1.  **Scan**: Identify the *primary* dynamic object (the one undergoing the most significant physical change).
        2.  **Infer**: Look for visual evidence of motion:
            *   *Motion Blur*: Indicates high velocity.
            *   *Displacement*: Position relative to the ground/background suggests potential energy.
            *   *Deformation*: Indicates impact or material stress.
        3.  **Synthesize**: Connect the visual evidence to a specific Physics Principle.

        ## 5. Strict Output Schema
        Return **ONLY** a raw JSON object. Do not use Markdown code blocks (` ```json `). Do not include conversational filler.

        ```json
        {{
        "main_object": "Precise name of the object (e.g., 'Spinning Gyroscope', 'Falling Water Droplet')",
        "kinematic_state": "Description of motion inferred from visual cues (e.g., 'High rotational velocity indicated by edge blur', 'Zero velocity at peak trajectory')",
        "physics_principle": "The core law active here (e.g., 'Conservation of Angular Momentum', 'Bernoulli's Principle')",
        "explanation": "A 3-5 sentence explanation strictly tailored to the {analysis_level}. If 'Child', be playful. If 'Expert', be rigorous."
        }}
        ```

        ## 6. Critical Safety Constraints
        *   **No Hallucinations**: If the image is blurry or ambiguous, describe the ambiguity in the `kinematic_state` rather than guessing.
        *   **JSON Validity**: Ensure all quotes are escaped properly. The output must be parseable by `json.loads()`.
        *   **Audience Match**: If `analysis_level` is "Child", do NOT use words like "Kinematics." If "Expert", do NOT use words like "Zoomy."

        **Input Variable:**
        Target Audience Level: `{analysis_level}`
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