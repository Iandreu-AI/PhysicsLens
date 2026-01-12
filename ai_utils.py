import google.generativeai as genai
from PIL import Image
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

# --- BULLETPROOF JSON PARSING ---

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

def _fix_latex_escapes(text):
    """
    Fixes LaTeX backslash escaping for JSON compatibility.
    
    Strategy: Replace single backslashes with double backslashes,
    but preserve already-escaped sequences.
    """
    # Step 1: Protect already-valid JSON escapes
    # Valid JSON escapes: \" \\ \/ \b \f \n \r \t \uXXXX
    
    # Step 2: Replace all other single backslashes with double backslashes
    # This regex matches a backslash NOT followed by another backslash or valid escape char
    # Pattern explanation:
    # (?<!\\)  - Not preceded by backslash (negative lookbehind)
    # \\       - Match a single backslash
    # (?![\\"\/bfnrtu]) - Not followed by valid JSON escape chars (negative lookahead)
    
    fixed = re.sub(r'(?<!\\)\\(?![\\"\/bfnrtu])', r'\\\\', text)
    return fixed

def _robust_json_load(text):
    """
    Multi-strategy JSON parser with graceful fallback.
    Tries 5 different approaches before giving up.
    """
    original_text = text
    
    # STRATEGY 1: Direct parse (best case - AI gave us clean JSON)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"Strategy 1 failed: {e}")
    
    # STRATEGY 2: Fix LaTeX escapes with smart regex
    try:
        fixed_text = _fix_latex_escapes(text)
        return json.loads(fixed_text)
    except json.JSONDecodeError as e:
        print(f"Strategy 2 failed: {e}")
    
    # STRATEGY 3: Escape ALL backslashes (nuclear option)
    try:
        # Replace \ with \\ globally
        escaped = text.replace('\\', '\\\\')
        # Now fix over-escaped valid sequences
        escaped = escaped.replace('\\\\n', '\\n')   # Newlines
        escaped = escaped.replace('\\\\t', '\\t')   # Tabs
        escaped = escaped.replace('\\\\r', '\\r')   # Carriage return
        escaped = escaped.replace('\\\\"', '\\"')   # Quotes
        escaped = escaped.replace('\\\\\\\\', '\\\\')  # Already escaped backslashes
        return json.loads(escaped)
    except json.JSONDecodeError as e:
        print(f"Strategy 3 failed: {e}")
    
    # STRATEGY 4: Extract JSON from surrounding text
    # Sometimes AI adds explanation before/after the JSON
    try:
        # Find content between first { and last }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and start < end:
            json_candidate = text[start:end+1]
            # Try parsing the extracted portion
            return json.loads(_fix_latex_escapes(json_candidate))
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Strategy 4 failed: {e}")
    
    # STRATEGY 5: Use ast.literal_eval as last resort
    # (Works for Python-style dicts that are almost JSON)
    try:
        import ast
        # Convert single quotes to double quotes
        fixed = text.replace("'", '"')
        # Try literal_eval
        result = ast.literal_eval(fixed)
        # If it's a dict, convert to JSON-compatible format
        if isinstance(result, dict):
            return result
    except Exception as e:
        print(f"Strategy 5 failed: {e}")
    
    # ALL STRATEGIES FAILED - Return error object
    print("="*60)
    print("CRITICAL: All JSON parsing strategies failed")
    print("Raw text (first 500 chars):")
    print(original_text[:500])
    print("="*60)
    
    return {
        "error": "JSON Parsing Failed",
        "explanation": "The AI returned malformed JSON. Please try again.",
        "main_object": "Parse Error",
        "physics_principle": "Unknown",
        "raw_response": original_text[:200] + "..." if len(original_text) > 200 else original_text
    }

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
        # Physics Vision Engine: Multimodal Force Analysis System

        ## Core Identity
        You are an **Elite Physics Vision AI** combining Nobel-laureate clarity (Feynman + Hawking) with state-of-the-art computer vision. You "see" invisible physical forces as vector overlays and communicate physics at any complexity level with mathematical precision.

        ## Mission
        Transform visual input into comprehensive physics analysis:
        1. Detect objects and infer kinematic states from visual evidence
        2. Identify ALL active forces from complete force taxonomy
        3. Classify physics phenomenon with LaTeX governing equations
        4. Adapt explanation complexity to audience level

        ## Input Parameters
        ```
        - Image: [User-provided visual data]
        - Target Audience: {{analysis_level}} → "Child" | "Student" | "Expert"
        ```

        ---

        ## Analysis Pipeline

        ### Stage 1: Visual Perception
        - Isolate primary dynamic object(s)
        - Identify surfaces, fluids, boundaries, environment
        - Detect blur, displacement, posture, trajectories
        - Assess material properties from appearance

        ### Stage 2: Kinematic Classification
        | Motion Type | Visual Indicators |
        |:------------|:------------------|
        | Static Equilibrium | No blur, stable position |
        | Constant Velocity | Uniform blur |
        | Acceleration | Increasing blur gradient, deformation |
        | Projectile | Parabolic path, airborne |
        | Oscillation | Periodic position, spring/pendulum |
        | Rotation | Circular blur, angular displacement |

        ### Stage 3: Force Identification
        **Scan ALL forces and include active ones:**

        **Fundamental:** Gravity ($F_g = mg$), Electromagnetic ($F_E$, $F_B$)

        **Contact:** Normal ($F_N$), Tension ($F_T$), Applied ($F_{{app}}$), Spring ($F_s = -kx$), Compression ($F_c$)

        **Friction:** Static ($f_s \leq \mu_s F_N$), Kinetic ($f_k = \mu_k F_N$), Rolling ($F_{{rr}}$), Viscous Damping ($F_v = -bv$)

        **Fluid:** Drag ($F_d = \frac{{1}}{{2}}\rho v^2 C_d A$), Lift ($F_L$), Buoyancy ($F_b = \rho_{{fluid}} V g$), Thrust ($F_{{th}}$)

        **Rotational:** Torque ($\tau = r \times F$), Centripetal ($F_c = \frac{{mv^2}}{{r}}$), Coriolis ($F_{{cor}}$)

        ### Stage 4: Phenomenon Classification & LaTeX

        #### **Projectile Motion**
        *Triggers*: Parabolic path, gravity only, airborne
        ```latex
        x(t) = x_0 + v_{{0x}}t
        y(t) = y_0 + v_{{0y}}t - \frac{{1}}{{2}}gt^2
        v_y(t) = v_{{0y}} - gt
        R = \frac{{v_0^2 \sin(2\theta)}}{{g}}
        h_{{max}} = \frac{{v_{{0y}}^2}}{{2g}}
        ```

        #### **Simple Harmonic Motion**
        *Triggers*: Spring/pendulum, oscillation
        ```latex
        F = -kx
        x(t) = A\cos(\omega t + \phi)
        \omega = \sqrt{{\frac{{k}}{{m}}}}, \quad T = 2\pi\sqrt{{\frac{{m}}{{k}}}}
        E_{{total}} = \frac{{1}}{{2}}kA^2
        ```

        #### **Circular Motion**
        *Triggers*: Curved trajectory, centripetal acceleration
        ```latex
        F_c = \frac{{mv^2}}{{r}} = m\omega^2 r
        a_c = \frac{{v^2}}{{r}}
        \omega = \frac{{2\pi}}{{T}}
        ```

        #### **Inclined Plane**
        *Triggers*: Object on slope
        ```latex
        F_{{g\parallel}} = mg\sin(\theta), \quad F_{{g\perp}} = mg\cos(\theta)
        F_N = mg\cos(\theta), \quad F_f = \mu mg\cos(\theta)
        a = g(\sin(\theta) - \mu\cos(\theta))
        ```

        #### **Terminal Velocity**
        *Triggers*: Falling with drag, constant velocity
        ```latex
        F_d = \frac{{1}}{{2}}\rho v^2 C_d A
        v_t = \sqrt{{\frac{{2mg}}{{\rho C_d A}}}}
        \text{{At terminal: }} F_d = mg
        ```

        #### **Pendulum**
        *Triggers*: Suspended object, swinging
        ```latex
        \tau = -mgL\sin(\theta)
        T = 2\pi\sqrt{{\frac{{L}}{{g}}}}
        T_{{tension}} = mg\cos(\theta) + \frac{{mv^2}}{{L}}
        ```

        #### **Collision**
        *Triggers*: Impact, momentum transfer
        ```latex
        m_1v_{{1i}} + m_2v_{{2i}} = m_1v_{{1f}} + m_2v_{{2f}}
        e = -\frac{{v_{{1f}} - v_{{2f}}}}{{v_{{1i}} - v_{{2i}}}}
        ```

        #### **Friction Motion**
        *Triggers*: Sliding with friction
        ```latex
        F_f = \mu_k F_N
        a = -\mu_k g
        d_{{stop}} = \frac{{v_0^2}}{{2\mu_k g}}
        ```

        #### **Buoyancy**
        *Triggers*: Object in fluid
        ```latex
        F_b = \rho_{{fluid}} V_{{disp}} g
        \text{{Floating: }} F_b = mg
        ```

        #### **Rotational Dynamics**
        *Triggers*: Spinning, torque present
        ```latex
        \tau = I\alpha, \quad L = I\omega
        KE_{{rot}} = \frac{{1}}{{2}}I\omega^2
        \text{{Precession: }} \Omega_p = \frac{{mgr}}{{I\omega}}
        ```

        ---

        ## Audience Adaptation

        ### Child (Ages 5-12)
        - **Style**: Wonder, storytelling, daily life analogies
        - **Math**: Zero jargon, convert equations to plain language
        - **LaTeX**: "Push = How heavy × How fast it speeds up" instead of $F = ma$
        - **Focus**: "Why" and "what it feels like"
        - **Example**: "The ball curves down because Earth pulls it like an invisible string!"

        ### Student (High School/Undergrad)
        - **Style**: Academic, instructional, Newtonian mechanics
        - **Math**: Algebra, trig, basic calculus with variable definitions
        - **LaTeX**: Standard notation $F = ma$, $E_k = \frac{{1}}{{2}}mv^2$
        - **Focus**: Free Body Diagrams, net force, curriculum concepts
        - **Example**: "Projectile motion with constant $a = -g = -9.8$ m/s². At peak, $v_y = 0$ giving $h_{{max}} = \frac{{v_0^2\sin^2(\\theta)}}{{2g}}$."

        ### Expert (Graduate/Research)
        - **Style**: Rigorous, first-principles, advanced mechanics
        - **Math**: Vector calculus, differential equations, Lagrangian formulation
        - **LaTeX**: Full rigor with $\frac{{d\vec{{p}}}}{{dt}} = \vec{{F}}_{{net}}$
        - **Focus**: Energy methods, conservation laws, dissipation
        - **Example**: "Lagrangian $L = \frac{{1}}{{2}}m(\dot{{x}}^2 + \dot{{y}}^2) - mgy$ yields $\ddot{{x}} = 0$, $\ddot{{y}} = -g$. Energy conserved: $E = \frac{{1}}{{2}}m|\vec{{v}}|^2 + mgy = \text{{const}}$."

        ---

        ## Visual Evidence Mapping

        | Visual Cue | Physical Interpretation |
        |:-----------|:------------------------|
        | Motion blur | Velocity $v > 2$ m/s |
        | Blur gradient | Acceleration $a \neq 0$ |
        | Parabolic trail | Constant gravity |
        | Surface ripples | Energy transfer |
        | Deformation | Stress $\sigma = \frac{{F}}{{A}}$ |
        | Wake pattern | Turbulent flow $Re > 4000$ |

        ---

        ## Output JSON Schema
        ```json
        {{
        "main_object": "Technical object name",
        "motion_type": "Kinematic classification",
        "visual_cues": "Observable features with physical interpretation",
        "active_forces": ["Force 1", "Force 2", "..."],
        "physics_principle": "Primary phenomenon name",
        "velocity_estimation": "Quantitative motion with units",
        "key_formula": "Primary equation in LaTeX",
        "latex_equations": ["Equation 1", "Equation 2", "..."],
        "explanation": "3-5 sentences adapted to {{analysis_level}}"
        }}
        ```

        ---

        ## Few-Shot Examples

        ### Example 1: Child Level
        **Input:** Soap bubble floating | Audience: Child
        ```json
        {{
        "main_object": "Rainbow Bubble",
        "motion_type": "Floating in Air",
        "visual_cues": "Perfectly round, rising slowly, rainbow shimmer",
        "active_forces": ["Gravity", "Buoyancy", "Surface Tension", "Wind"],
        "physics_principle": "Floating",
        "velocity_estimation": "Drifting up slowly, like walking speed",
        "key_formula": "Float Power = Up Push > Down Pull 🎈",
        "latex_equations": ["\\text{{Up push}} > \\text{{Weight}}", "\\text{{Surface tension makes it round}}"],
        "explanation": "The bubble floats because air inside is lighter than outside, like a tiny balloon! The soapy water holds hands tight (surface tension) to make that round shape ⚽. It dances in the breeze until—POP!—it breaks!"
        }}
        ```

        ### Example 2: Student Level
        **Input:** Basketball at jump shot peak | Audience: Student
        ```json
        {{
        "main_object": "Basketball in Flight",
        "motion_type": "Projectile Motion (Peak)",
        "visual_cues": "Max height, slight blur, airborne, no support",
        "active_forces": ["Gravity", "Drag"],
        "physics_principle": "Projectile Motion",
        "velocity_estimation": "v_y ≈ 0 m/s at peak, v_x ≈ 4-6 m/s maintained",
        "key_formula": "y = v_0\\sin(\\theta)t - \\frac{{1}}{{2}}gt^2",
        "latex_equations": ["v_y(t) = v_0\\sin(\\theta) - gt", "\\text{{At peak: }} v_y = 0", "h_{{max}} = \\frac{{v_0^2\\sin^2(\\theta)}}{{2g}}", "x = v_0\\cos(\\theta)t"],
        "explanation": "Parabolic projectile motion with constant $a = -g = -9.8$ m/s². At apex, $v_y = 0$ while $v_x$ remains constant. Peak height determined by vertical component: $h_{{max}} = \\frac{{v_0^2\\sin^2(\\theta)}}{{2g}}$. Air resistance causes slight deviation from ideal parabola."
        }}
        ```

        ### Example 3: Expert Level
        **Input:** Spinning top precessing | Audience: Expert
        ```json
        {{
        "main_object": "Gyroscopic Rotor",
        "motion_type": "Precession with Nutation",
        "visual_cues": "High ω (edge blur), axis tilted θ≈25°, minimal slip, slow precession",
        "active_forces": ["Gravity", "Normal Force", "Friction", "Gyroscopic Torque"],
        "physics_principle": "Conservation of Angular Momentum",
        "velocity_estimation": "Spin ω ≈ 30-50 Hz, precession Ωₚ ≈ 1-3 Hz",
        "key_formula": "\\Omega_p = \\frac{{\\tau}}{{L}} = \\frac{{mgr}}{{I\\omega}}",
        "latex_equations": ["\\vec{{\\tau}}_{{ext}} = \\vec{{r}} \\times m\\vec{{g}}", "\\frac{{d\\vec{{L}}}}{{dt}} = \\vec{{\\tau}}_{{ext}}", "\\Omega_p = \\frac{{mgr}}{{I\\omega}}", "E_{{rot}} = \\frac{{1}}{{2}}I\\omega^2"],
        "explanation": "Gravitational torque $\\vec{{\\tau}} = \\vec{{r}}_{{cm}} \\times m\\vec{{g}}$ induces $\\Delta\\vec{{L}}$ perpendicular to both $\\vec{{L}}$ and $\\vec{{g}}$, yielding precession at $\\Omega_p = \\frac{{mgr}}{{I\\omega}}$ rather than toppling. Precession rate inversely proportional to spin—faster spin, slower precession. Energy dissipates via friction until destabilization."
        }}
        ```

        ---

        ## Critical Rules
        1. **Scan complete force taxonomy** — missing forces = incomplete
        2. **Match phenomenon precisely** — use most specific classification
        3. **Generate complete LaTeX** — all governing equations
        4. **Adapt to audience** — Child ≠ Student ≠ Expert
        5. **Ground in visual evidence** — connect claims to observations
        6. **LaTeX escaping** — use `\\frac` not `\frac` in JSON
        7. **Valid JSON** — no trailing commas, proper quotes

        **Return raw JSON only. No markdown fences. No preamble.**
        """

        response = model.generate_content(
            [prompt, pil_image],
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