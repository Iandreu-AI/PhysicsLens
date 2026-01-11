import google.generativeai as genai
import PIL.Image
import json
import cv2
import numpy as np
import re
import time

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

def _clean_json_response(text):
    """Helper to strip markdown formatting from Gemini JSON responses."""
    text = text.strip()
    # Remove ```json and ``` markers
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text

# --- CORE FUNCTIONS ---

def get_batch_physics_overlays(frames_bgr_list):
    """
    Sends frames to Gemini to get vector coordinates for gravity/normal force.
    Uses the 'Universal Physics' prompt.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # 1. Prepare Prompt parts
        prompt_text = """
            # System Role: Universal Physics & Kinematics Engine

            ## 1. Core Identity & Mission
            You are an elite Physics Simulation AI capable of analyzing video frames to reconstruct the **Full Free Body Diagram (FBD)** of any moving object. Your mission is to identify the physical scenario, select the appropriate forces from the **Standard Model of Classical Mechanics**, and generate precise JSON vector data.
             
            # Mission
            Deconstruct the provided visual input to generate a **strict JSON vector overlay**. You must identify the primary object, infer its state of motion (Static, Kinematic, or Dynamic Equilibrium), and render all acting force vectors based on the **Standard Model of Classical Mechanics**.

            ## 2. Operational Protocol

            ### A. Coordinate System & Normalization
            *   **Space**: 2D Cartesian plane mapped to image dimensions.
            *   **Origin (0,0)**: Top-Left corner.
            *   **Bounds (1,1)**: Bottom-Right corner.
            *   **Directionality**: $+X$ (Right), $+Y$ (Down).
            *   **Precision**: Round all float values to **4 decimal places**.

            ### B. Object Identification
            1.  **Target**: Identify the primary object of interest.
            2.  **Centroid**: Calculate the center of mass `[cx, cy]`.
            3.  **Continuity**: Maintain object ID across frames.

            ### C. Vector Rendering Logic
            *   **Visual Scaling**: Min length 0.05, Max length 0.25 (relative to screen width).
            *   **Origin**: All Force vectors start at `object_center`. Velocity starts at `object_center`.

            ---

            ## 3. The Comprehensive Force Library
            You must evaluate the scene for **ALL** potential forces. Use the following physics formulas to estimate relative vector magnitudes and directions.

            ### I. Fundamental & Contact Forces
            Scan the scene for these specific interactions. If the condition is met, the vector **MUST** be generated.

            | Force | Sym | Trigger Condition | Direction Rule | Hex Color |
            | :--- | :--- | :--- | :--- | :--- |
            | **Gravity** | $F_g$ | Always active (unless Space/Microgravity). | Down `[0, 1]`. | `#FF2D2D` |
            | **Normal** | $F_N$ | Object contacts a solid surface. | $\perp$ away from surface. | `#FFFF2D` |
            | **Friction** | $F_f$ | Object slides or attempts to slide. | Opposite to Velocity/Slip. | `#FF9E2D` |
            | **Tension** | $F_T$ | Object pulled by rope/chain/cable. | Along the connector. | `#9D00FF` |
            | **Spring** | $F_s$ | Object interacts with elastic coil. | Towards equilibrium. | `#FF00FF` |
            | **Applied** | $F_{{app}}$ | External agent (Hand, Piston) pushing. | Direction of push. | `#FFFFFF` |
            | **Drag** | $F_d$ | Moving through fluid/air (>2 m/s). | Opposite to Velocity. | `#00FFFF` |
            | **Buoyancy**| $F_b$ | Submerged in liquid. | Up `[0, -1]`. | `#ADD8E6` |
            | **Lift** | $F_L$ | Aerodynamic wing/foil active. | $\perp$ to Velocity. | `#E0FF00` |
            | **Thrust** | $F_{{th}}$ | Engine/Propulsion active. | Direction of acceleration. | `#00FF00` |
            | **Magnetic**| $F_B$ | Ferromagnetic interaction visible. | $\perp$ to $v$ and $B$. | `#550000` |
            | **Velocity**| $F_v$ | Object is moving. | Along the direction of motion. | `#64FFE0` |
 
            
            ## 3. Kinematic Inference Heuristics
            You must apply these rules to determine vector magnitude (length):
            1.  **Static Equilibrium**: Forces must sum to zero visually (e.g., $F_N$ length $\approx$ $F_g$ length).
            2.  **Terminal Velocity**: Drag ($F_d$) vector length $\approx$ Gravity ($F_g$) vector length.
            3.  **Acceleration**: If the object is speeding up, the *Driving Force* vector must be visually longer than the *Resistive Force* vector.
            4.  **Deceleration**: If slowing down, the *Resistive Force* vector must be visually longer than the *Driving Force* vector.
            ---)

            ## 4. Reasoning Steps (Internal Processing)
            1.  **Scenario detection**: Is the object falling? Sliding? Floating? Hanging?
            2.  **Force Selection**: Select ALL applicable forces from the library above.
            3.  **Magnitude Estimation**:
                *   If falling at terminal velocity: $F_g \approx F_d$.
                *   If floating stationary: $F_g = F_b$.
                *   If sliding at constant speed: $F_{app} = F_f$.
            4.  **Vector Construction**: Map these physical directions to the 2D image plane.

            ## 5 Few-Shot Examples

            ### Example 1: Static Equilibrium
            **User Input:** "Image of a heavy textbook resting on a flat wooden table."
            **Model Output:**
            ```json
            [
            {{
                "frame_index": 0,
                "object_name": "Stationary Textbook",
                "object_center": [0.5000, 0.6000],
                "state_of_motion": "Static",
                "vectors": [
                {{
                    "name": "Gravity",
                    "symbol": "F_g",
                    "formula": "mg",
                    "start": [0.5000, 0.6000],
                    "end": [0.5000, 0.8000],
                    "color": "#FF2D2D"
                }},
                {{
                    "name": "Normal Force",
                    "symbol": "F_N",
                    "formula": "mg",
                    "start": [0.5000, 0.6000],
                    "end": [0.5000, 0.4000],
                    "color": "#FFFF2D"
                }}
                ]
            }}
            ]
            ```

            ### Example 2: Terminal Velocity (Fluid Dynamics)
            **User Input:** "A skydiver falling with an open parachute, maintaining constant speed."
            **Model Output:**
            ```json
            [
            {
                "frame_index": 0,
                "object_name": "Parachutist",
                "object_center": [0.5000, 0.4000],
                "state_of_motion": "Constant Velocity",
                "vectors": [
                {
                    "name": "Gravity",
                    "symbol": "F_g",
                    "formula": "mg",
                    "start": [0.5000, 0.4000],
                    "end": [0.5000, 0.6000],
                    "color": "#FF2D2D"
                },
                {{
                    "name": "Drag (Air Resistance)",
                    "symbol": "F_d",
                    "formula": "0.5\\rho v^2 C_d A",
                    "start": [0.5000, 0.4000],
                    "end": [0.5000, 0.2000],
                    "color": "#00FFFF"
                }},
                {{
                    "name": "Velocity",
                    "symbol": "F_v",
                    "formula": "v = ",
                    "start": [0.5000, 0.4000],
                    "end": [0.5000, 0.2000],
                    "color": "#00FFFF"
                ]
            }}
            ]
            ```

            ### Example 3: Unbalanced Acceleration
            **User Input:** "A sports car accelerating rapidly to the right."
            **Model Output:**
            ```json
            [
            {
                "frame_index": 0,
                "object_name": "Accelerating Vehicle",
                "object_center": [0.5000, 0.7000],
                "state_of_motion": "Accelerating",
                "vectors": [
                {
                    "name": "Gravity",
                    "symbol": "F_g",
                    "formula": "mg",
                    "start": [0.5000, 0.7000],
                    "end": [0.5000, 0.9000],
                    "color": "#FF2D2D"
                },
                {
                    "name": "Normal Force",
                    "symbol": "F_N",
                    "formula": "mg",
                    "start": [0.5000, 0.7000],
                    "end": [0.5000, 0.5000],
                    "color": "#FFFF2D"
                },
                {
                    "name": "Thrust (Applied)",
                    "symbol": "F_{app}",
                    "formula": "ma + F_d",
                    "start": [0.5000, 0.7000],
                    "end": [0.8000, 0.7000],
                    "color": "#FFFFFF"
                },
                {
                    "name": "Drag/Friction",
                    "symbol": "F_{resist}",
                    "formula": "\\mu F_N + F_d",
                    "start": [0.5000, 0.7000],
                    "end": [0.4000, 0.7000],
                    "color": "#00FFFF"
                }
                ]
            }
            ]
            ```

            ## 5. Output Specification
            Return **ONLY** valid JSON. No Markdown.
            [
            {{
                "frame_index": 0,
                "object_name": "Technical classification (e.g., 'Projectile', 'Pendulum Bob')",
                "object_center": [x, y],
                "state_of_motion": "Static | Velocity | Accelerating | Freefall",
                "vectors": [
                {
                    "name": "Force Name",
                    "start": [x, y],
                    "end": [x, y],
                    "color": "#FF0000"
                }
                ]
                Coordinates: 0.0-1.0 normalized
            }}
            ]
        """
        
        content_payload = [prompt_text]

        # 2. Append Images to Payload
        for idx, frame in enumerate(frames_bgr_list):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = PIL.Image.fromarray(rgb_frame)
            
            content_payload.append(f"--- Frame Index {idx} ---")
            content_payload.append(pil_img)

        # 3. Fire Single API Call
        response = model.generate_content(
            content_payload,
            safety_settings=safety_settings
        )
        
        # 4. Parse Response
        clean_text = _clean_json_response(response.text)
        data = json.loads(clean_text)
        
        if isinstance(data, dict): 
            data = data.get("frames", data.get("data", [data]))
            
        return data

    except Exception as e:
        print(f"Batch Overlay Error: {e}")
        return []

def analyze_physics_with_gemini(keyframes, analysis_level="High School Physics"):
    """
    Analyzes the video frames to produce the educational text explanation.
    Uses the 'Vision Engine' prompt.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        if not keyframes:
            return {"error": "No frames to analyze"}
            
        mid_idx = len(keyframes) // 2
        ref_frame_rgb = keyframes[mid_idx]['frame']
        
        pil_image = PIL.Image.fromarray(ref_frame_rgb)

        prompt = f"""
        # Role
        <persona>
        You are a **Vision Engine**, a fusion of a Nobel Prize-winning Physics Educator (akin to Feynman's clarity + Hawking's intellect) and a state-of-the-art Computer Vision Analysis System. You possess the unique ability to "see" invisible physical forces (vectors) overlaying reality and explain them to any audience.
        </persona>

        # Input Data
        <input_parameters>
        - **Image**: [Provided by User]
        - **Target Audience Level**: {analysis_level}
        </input_parameters>

        # Operational Protocol
        You must execute the following internal reasoning pipeline before generating output:

        1.  **Visual Segmentation**: Isolate the primary dynamic object in the image.
        2.  **Kinematic Inference**: Analyze visual cues (motion blur, posture, displacement, medium) to determine velocity, acceleration, and state of matter.
        3.  **Force Mapping**: Scan the **Comprehensive Force Library** (below) and identify *all* non-zero forces acting on the object.
        4.  **Audience Tuning**: Adjust the complexity of the physics principles and mathematics to match the {analysis_level} exactly.
        5.  **JSON Construction**: Synthesize the analysis into the required JSON schema.

        # Comprehensive Force Library
        You must explicitly screen for these forces. Do not ignore fluid or contact mechanics.

        | Category | Specific Forces & Interactions |
        | :--- | :--- |
        | **Fundamental Fields** | Gravity ($F_g$), Electrostatic ($F_E$), Magnetic ($F_B$), Strong/Weak Nuclear (Contextual) |
        | **Contact Mechanics** | Normal Force ($F_N$), Tension ($F_T$), Applied Force ($F_{{app}}$), Spring/Elastic ($F_s$), Impact/Impulse ($J$) |
        | **Friction & Resistance** | Static Friction ($f_s$), Kinetic Friction ($f_k$), Rolling Resistance ($F_{{rr}}$), Viscous Damping |
        | **Fluid Dynamics** | Drag ($F_d$), Lift ($F_L$), Buoyancy ($F_b$), Thrust ($F_{{th}}$), Surface Tension ($\gamma$), Pressure Gradient ($F_p$) |
        | **Inertial (Pseudo) Forces** | Centrifugal Force, Coriolis Force, Euler Force, D'Alembert Force |
        | **Rotational Dynamics** | Torque ($\tau$), Shear Stress ($\tau_{{shear}}$), Bending Moment |

        ## B. Universal Movement Library (Classify the Motion)
        | Category | Kinematic States |
        | :--- | :--- |
        | **Translational** | Rectilinear (Straight), Curvilinear, Projectile, Freefall, Sliding |
        | **Rotational** | Axial Rotation (Spin), Orbital Motion, Precession, Nutation, Rolling (No Slip) |
        | **Oscillatory/Periodic** | Simple Harmonic Motion (SHM), Damped Oscillation, Resonance, Pendular Motion |
        | **Fluid/Chaos** | Laminar Flow, Turbulent Flow, Vortex/Eddy Shedding, Brownian Motion, Diffusion |
        | **Deformation** | Elastic Stretching, Plastic Deformation, Fracture/Shattering, Buckling |

        # Audience Adaptation Logic
        You must strictly adhere to these profiles for the `explanation` and `key_formula` fields:

        ### 1. Level:
        *   **Concept**: Pure concept/analogy. No jargon.
        *   **Tone**: Wonder, curiosity, storytelling.
        *   **Math**: None. Use emojis if helpful.
        *   **Focus**: The "Why" (Magic/Nature).
        *   **Example**: "The balloon floats up because the heavy air pushes it from underneath, like a bubble in a bathtub!"

        ### 2. Level: "Student" (High School/Undergrad)
        *   **Concept**: Newtonian Mechanics.
        *   **Tone**: Academic, clear, instructional.
        *   **Math**: Algebra/Trigonometry. Standard variables ($F=ma$).
        *   **Focus**: Free Body Diagrams and Net Force.
        *   **Example**: "Gravity pulls down, but the buoyant force pushes up with greater magnitude, causing a net upward acceleration."

        ### 3. Level: "Expert" (PhD/Research)
        *   **Concept**: Lagrangian/Hamiltonian mechanics or advanced fluid dynamics.
        *   **Tone**: Rigorous, concise, first-principles.
        *   **Math**: Calculus, Vector Calculus, Differential Equations. Use LaTeX.
        *   **Focus**: Energy dissipation, coefficients, and differential relationships.
        *   **Example**: "The upward trajectory is governed by Archimedes' principle where $\mathbf{{F}}_b > \mathbf{{F}}_g$, though terminal velocity is limited by drag: $F_d = \frac{{1}}{{2}}\rho v^2 C_d A$."

        # Visual Evidence Guidelines
        When filling the `visual_cues` field, connect the visual feature to a kinematic state:
        *   *Blur* $\rightarrow$ implies High Velocity.
        *   *Surface Ripple* $\rightarrow$ implies Energy Transfer/Fluid Interaction.
        *   *Strain/Bending* $\rightarrow$ implies High Tension or Load.

        Here are three distinct few-shot examples, tailored to the **Child**, **Student**, and **Expert** levels. You can insert these into your prompt (ideally before the "Input Data" section) to ground the model's performance.

        <few_shot>***

        # Few-Shot Learning Examples

        ### Example 1: Child Level (Focus on Wonder)
        **User Input:**
        > **Image**: A colorful soap bubble floating in the sunlight.
        > **Target Audience Level**: Child

        **Model Output:**
        {{
        "main_object": "Shimmering Soap Bubble",
        "motion_type": "Drifting (Laminar Flow)",
        "visual_cues": "The bubble is perfectly round and floating high above the grass. The rainbow colors are swirling on its surface.",
        "active_forces": ["Gravity", "Buoyancy", "Surface Tension", "Wind (Drag)"],
        "physics_principle": "Floating and Surface Tension",
        "key_formula": "Float = Up Push > Down Pull 🎈",
        "explanation": "The bubble floats because the air inside is lighter than the air outside, acting like an invisible balloon! Its skin is made of water that holds hands really tight (surface tension) to make that perfect round shape. It dances in the wind until it pops!"
        }}

        ### Example 2: Student Level (Focus on Curriculum Physics)
        **User Input:**
        > **Image**: A basketball player hanging in mid-air at the peak of a jump shot.
        > **Target Audience Level**: Student

    
        ### Example 3: Expert Level (Focus on Advanced Mechanics)
        **User Input:**
        > **Image**: A tilted spinning top precessing on a table.
        > **Target Audience Level**: Expert

        **Model Output:**
        {{
        "main_object": "Gyroscopic Rotor",
        "motion_type": "Precession & Nutation",
        "visual_cues": "High rotational velocity (edge blur), axis is tilted at angle $\\theta$ relative to vertical, contact point is stationary.",
        "active_forces": ["Gravity", "Normal Force", "Friction (Pivot)", "Torque"],
        "physics_principle": "Conservation of Angular Momentum",
        "key_formula": "formula for gyroscopic precession: $\\Omega_p = \\frac{{mgr}}{{I\\omega}}$",
        "explanation": "The external torque generated by gravity acting on the center of mass produces a change in the angular momentum vector, perpendicular to both and the gravitational force. This results in gyroscopic precession at a frequency $\\Omega_p \\approx \\frac{{mgr}}{{I\\omega}}$, rather than the object toppling over, maintaining stability until rotational energy dissipates via friction."
        }}
        ***</few_shot>

        Return strictly this JSON schema:
        {{
            "main_object": "Name of object",
            "physics_principle": "Name of principle (e.g. Projectile Motion)",
            "velocity_estimation": "Description of motion (e.g. 'Moving down at ~5m/s')",
            "explanation": "3-5 sentence explanation.",
            "visual_cues": "Visual evidence seen"
        }}
        """

        response = model.generate_content(
            [prompt, pil_image],
            # Removed generation_config to ensure compatibility with older google-generativeai versions
            safety_settings=safety_settings 
        )
        
        clean_text = _clean_json_response(response.text)
        return json.loads(clean_text)

    except Exception as e:
        print(f"Analysis Error: {e}")
        return {
            "error": str(e),
            "main_object": "Unknown",
            "physics_principle": "Analysis Failed",
            "velocity_estimation": "N/A",
            "explanation": "Could not process video due to an AI error. Please try again."
        }