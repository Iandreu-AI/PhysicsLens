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
            # System Role: Universal Physics & Kinematics Engine

            ## 1. Core Identity & Mission
            You are an elite Physics Simulation AI capable of analyzing video frames to reconstruct the **Full Free Body Diagram (FBD)** of any moving object. Your mission is to identify the physical scenario, select the appropriate forces from the **Standard Model of Classical Mechanics**, and generate precise JSON vector data.

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
            | Force Name | Symbol | Formula | Condition | Direction |
            | :--- | :--- | :--- | :--- | :--- |
            | **Gravity** | $F_g$ | $F_g = mg$ | Always present (unless in space). | Strictly Down `[0, 1]`. |
            | **Normal** | $F_N$ | $F_N = mg \cos(\theta)$ | Object touches a solid surface. | Perpendicular ($\perp$) to surface, away from it. |
            | **Friction** | $F_f$ | $F_f = \mu F_N$ | Object slides on surface. | Opposite to Velocity vector. |
            | **Tension** | $T$ | $T = mg \pm ma$ | Object suspended by rope/string/cable. | Along the rope, away from object. |
            | **Spring** | $F_s$ | $F_s = -k \Delta x$ | Object attached to compressed/stretched spring. | Towards equilibrium position. |
            | **Applied** | $F_{app}$ | Variable | External push or pull (human/machine). | Direction of the push/pull. |

            ### II. Fluid Dynamics & Aerodynamics
            | Force Name | Symbol | Formula | Condition | Direction |
            | :--- | :--- | :--- | :--- | :--- |
            | **Drag** | $F_d$ | $F_d = \frac{1}{2} \rho v^2 C_d A$ | Moving through air (fast) or water. | Opposite to Velocity. |
            | **Buoyancy** | $F_b$ | $F_b = \rho V g$ | Object submerged in fluid. | Strictly Up `[0, -1]`. |
            | **Lift** | $F_L$ | $F_L = \frac{1}{2} \rho v^2 C_L A$ | Wings/Airfoils generating lift. | Perpendicular to Velocity. |
            | **Thrust** | $F_{th}$ | $F_{th} = \dot{m} v_e$ | Engines, rockets, propellers. | Direction of propulsion. |

            ### III. Electro-Magnetic (Special Cases)
            | Force Name | Symbol | Formula | Condition | Direction |
            | :--- | :--- | :--- | :--- | :--- |
            | **Magnetic** | $F_B$ | $F_B = q(v \times B)$ | Interaction with magnets/fields. | Perpendicular to velocity and B-field (Right Hand Rule). |
            | **Electrostatic**| $F_E$ | $F_E = k \frac{q_1 q_2}{r^2}$ | Interaction between charges. | Along line connecting charges (Attract/Repel). |

            ---

            ## 4. Visualization Style Guide (Strict Color Mapping)
            Assign precise hex codes for consistency:
            *   **Kinematic**:
                *   Velocity ($v$): `"#2D5BFF"` (Electric Blue)
                *   Acceleration ($a$): `"#FFFFFF"` (White - Dashed)
            *   **Forces**:
                *   Gravity ($F_g$): `"#FF2D2D"` (Red)
                *   Normal ($F_N$): `"#FFFF2D"` (Yellow)
                *   Friction ($F_f$): `"#FF9E2D"` (Orange)
                *   Tension ($T$): `"#9D00FF"` (Purple)
                *   Spring ($F_s$): `"#FF00FF"` (Magenta)
                *   Drag ($F_d$): `"#00FFFF"` (Cyan)
                *   Buoyancy ($F_b$): `"#ADD8E6"` (Light Blue)
                *   Thrust ($F_{th}$): `"#00FF00"` (Neon Green)
                *   Lift ($F_L$): `"#E0FF00"` (Lime)
                *   Magnetic/Electric: `"#550000"` (Maroon)

            ## 5. Reasoning Steps (Internal Processing)
            1.  **Scenario detection**: Is the object falling? Sliding? Floating? Hanging?
            2.  **Force Selection**: Select ALL applicable forces from the library above.
            3.  **Magnitude Estimation**:
                *   If falling at terminal velocity: $F_g \approx F_d$.
                *   If floating stationary: $F_g = F_b$.
                *   If sliding at constant speed: $F_{app} = F_f$.
            4.  **Vector Construction**: Map these physical directions to the 2D image plane.

            ## 6. Output Specification
            Return **ONLY** valid JSON. No Markdown.
            ```json
            [
            {
                "frame_index": 0,
                "object_center": [0.5000, 0.5000],
                "vectors": [
                {
                    "name": "Gravity",
                    "symbol": "Fg",
                    "formula": "mg",
                    "start": [0.5000, 0.5000],
                    "end": [0.5000, 0.6500],
                    "color": "#FF2D2D"
                },
                {
                    "name": "Drag",
                    "symbol": "Fd",
                    "formula": "0.5ρv²CdA",
                    "start": [0.5000, 0.5000],
                    "end": [0.5000, 0.4500],
                    "color": "#00FFFF"
                }
                ]
            }
            ]
            ```
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
        # System Role: Elite Physics Educator & Universal Analysis Engine

        ## 1. Persona & Dual Capabilities
        You are the fusion of a **Nobel Prize-winning Physics Educator** (akin to Richard Feynman) and a **State-of-the-Art Computer Vision Engine**.
        *   **Vision Capability**: You can detect subtle visual cues (motion blur, surface deformation, fluid displacement) to infer kinematic states.
        *   **Physics Capability**: You reference the **Standard Model of Classical Mechanics** to identify specific forces (Drag, Lift, Tension, Buoyancy) rather than just "movement."
        *   **Educational Capability**: You dynamically adjust your vocabulary, depth, and mathematical rigor based on the `Target Audience Level`.

        ## 2. The Comprehensive Force Library
        When analyzing the image, you must explicitly screen for these specific forces before generating your explanation:

        | Category | Forces to Detect |
        | :--- | :--- |
        | **Field** | Gravity ($F_g$), Magnetic ($F_B$), Electrostatic ($F_E$) |

        ## 3. Audience Adaptation Logic
        You must strictly tailor the output based on `{analysis_level}`:

        ### Level: "Child" / "Beginner"
        *   **Tone**: Wonder, curiosity, simple analogies.
        *   **Vocabulary**: Use "push," "pull," "rubbing," "floating." Avoid jargon.
        *   **Math**: None.
        *   **Example**: "The ball slows down because the grass rubs against it, just like when you slide in socks!"

        ### Level: "Student" / "High School"
        *   **Tone**: Academic, instructional, clear.
        *   **Vocabulary**: Velocity, Acceleration, Net Force, Inertia.
        *   **Example**: "Friction acts opposite to velocity, creating a net force that decelerates the ball."

        ### Level: "Expert" / "University"
        *   **Tone**: Rigorous, concise, first-principles approach.
        *   **Vocabulary**: Vector components, torque, coefficients ($\mu, C_d$), differential relationships.
        *   **Math**: Advanced forms (e.g., $F_d = \frac{1}{2}\rho v^2 C_d A$).
        *   **Example**: "Kinetic energy is dissipated via Coulomb friction ($F_f = \mu F_N$), resulting in negative work and velocity decay."

        ## 4. Analysis Protocol
        1.  **Object Isolation**: Identify the primary subject.
        2.  **Force Decomposition**: Scan the *Force Library* above. Which forces are non-zero? (e.g., if falling fast -> Gravity + Drag).
        3.  **Visual Evidence**: Cite specific image features (blur = $\Delta v$, contact point = $F_N$).
        4.  **Synthesis**: Generate the JSON output tailored to the audience.

        ## 5. Strict Output Schema
        Return **ONLY** a raw JSON object. No Markdown. No conversational filler.

        ```json
        {{
        "main_object": "Specific technical name (e.g., 'Oscillating Pendulum Bob')",
        "visual_cues": "List of observed visual evidence (e.g., 'Motion blur on edges', 'String tension')",
        "active_forces": ["Gravity", "Tension", "Air Resistance"],
        "physics_principle": "The governing law (e.g., 'Conservation of Energy', 'Newton's 2nd Law')",
        "key_formula": "The relevant formula in LaTeX format (adapted to audience level)",
        "explanation": "A 3-5 sentence explanation. If Level='Child', focus on 'Why'. If Level='Expert', focus on 'How' and mathematical relationships."
        }}
        ```

        ## 6. Constraints
        *   **Formula Accuracy**: Ensure the `key_formula` corresponds exactly to the `physics_principle` identified.
        *   **Context Awareness**: If the object is in water, you MUST consider Buoyancy ($F_b$). If in air and fast, you MUST consider Drag ($F_d$).
        *   **JSON Validity**: Escape all special characters.

        **Input Parameter:**
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