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

def get_batch_physics_overlays(frames_bgr_list):
    """
    Analyzes multiple frames in a single API call to ensure 
    temporal consistency and reduce latency.
    """
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp') # Flash is significantly faster for Vision
        
        # 1. Prepare the Payload
        prompt_parts = [
            """
            # Principal Physics Analysis Engine
            ## 1. Role & Expertise
            You are a **High-Precision Physics Analysis Engine**. Your core function is to derive precise vector data from video keyframes for automated physics visualization. You excel at:
            *   **Object Tracking**: Maintaining consistent object identification across image sequences.
            *   **Force Vector Decomposition**: Deconstructing motion into constituent forces (gravity, applied force, friction, normal force).
            *   **Coordinate Normalization**: Mapping pixel coordinates to a standardized 0.0-1.0 range.
            *   **JSON Encoding**: Generating strictly formatted JSON output optimized for downstream processing.

            ## 2. Input Context
            You are provided a sequence of keyframes extracted from a video of a dynamic physics phenomenon. You do not have access to the full video, only discrete frames.

            ## 3. Primary Task
            *   **Core**: For **EACH** provided frame, output a JSON structure containing object position and force vector data.
            *   **Consistency**: The identified "MAIN moving object" must remain consistent throughout the entire sequence. If the object is occluded, estimate its position based on previous trajectory.

            ## 4. Force Vector Definitions
            You must identify and approximate the following forces (if present):
            *   **Gravity**: Always present (magnitude depends on the identified object's mass, which you must consistently ESTIMATE). Direction: down (0, 1).
            *   **Velocity**: The instantaneous direction of motion derived from the difference in object center positions between consecutive frames.
            *   **Applied Force**: Any external force visibly acting on the object (e.g., a push, a pull). Direction and magnitude must be estimated.
            *   **Normal Force**: If the object is in contact with a surface, estimate the normal force (perpendicular to the surface).
            *   **Friction**: If motion is slowing down and there is contact with a surface, approximate the force of friction (opposite to the direction of motion).

            ## 5. Output Specification
            **Enforce the following JSON output structure**:
            ```json
            [
            {
                "frame_index": 0,
                "object_center": [ x, y ],
                "vectors": [
                { "name": "Gravity", "start": [x, y], "end": [x, y], "color": "red" },
                { "name": "Velocity", "start": [x, y], "end": [x, y], "color": "blue" },
                { "name": "Applied Force", "start": [x, y], "end": [x, y], "color": "green" }
                ]
            },
            ...
            ]
            ```

            ### Constraints:
            1.  **Strict JSON Only**: Absolutely NO Markdown, comments, or additional text.
            2.  **Coordinate Normalization**: `x` and `y` values MUST be in the range `0.0` to `1.0`. Normalize based on the assumed frame dimensions (you are NOT given the actual dimensions).
            3.  **Consistent Colors**:
                *   "Gravity": "red"
                *   "Velocity": "blue"
                *   "Applied Force": "green"
                *   "Normal Force": "yellow"
                *   "Friction": "orange"
            4.  **Force Vector Origin**: All force vectors originate from the `object_center`.

            ## 6. Input Data Format
            You will be provided a sequence of text descriptions for each frame. You must parse these descriptions to infer object position and motion. For example:

            ```text
            "Frame 0: A red ball is rolling to the right on a flat surface. The ball is in the center of the frame."
            "Frame 1: The red ball is now slightly to the right and a bit lower. It appears to be slowing down."
            "Frame 2: The ball continues to move right, but much slower. It is near the bottom of the frame."
            ```

            ## 7. Important Notes
            *   **Reasoning Chain**: Briefly justify your force vector estimations in comments *internally* but *do not include these justifications in the final JSON output.*
            *   **Mass Estimation**: You must estimate the mass of the object in order to derive the gravity vector magnitude. *Be consistent with this estimation across all frames.*

            ## 8. Execution
            Begin processing the keyframes. Output **ONLY** the JSON data.

            """
        ]

        # 2. Append Images to Prompt
        for idx, frame in enumerate(frames_bgr_list):
            # Convert BGR (OpenCV) to RGB (PIL)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = PIL.Image.fromarray(rgb)
            
            prompt_parts.append(f"--- Frame {idx} ---")
            prompt_parts.append(pil_img)

        # 3. Fire Single API Call
        response = model.generate_content(prompt_parts)
        
        # 4. Parse Response
        text = response.text
        text = text.replace("```json", "").replace("```", "").strip()
        
        # Robust parsing: ensure we get a list
        data = json.loads(text)
        if isinstance(data, dict): 
            # Handle edge case where AI wraps list in a dict key like {"frames": [...]}
            data = data.get("frames", data.get("data", [data]))
            
        return data

    except Exception as e:
        print(f"Batch Overlay Error: {e}")
        return []

def analyze_physics_with_gemini(frames_data, analysis_level="High School Physics"):
    """
    Analyzes the video frames to produce the text explanation.
    """
    try:
        model = genai.GenerativeModel('gemini-3-pro-preview') # Flash is faster for text
        
        # We take the middle frame as the reference image
        if not frames_data:
            return {"error": "No frames to analyze"}
            
        mid_idx = len(frames_data) // 2
        ref_frame = frames_data[mid_idx]['frame'] # This is already RGB from video_utils
        pil_image = PIL.Image.fromarray(ref_frame)

        prompt = f"""
        You are an expert Physics Professor with 20+ years of experience teaching mechanics, kinematics, and dynamics across multiple education levels. You specialize in analyzing real-world motion scenarios and explaining complex physics concepts in accessible, level-appropriate language.

        ## Your Mission

        Analyze a video frame (or sequence of frames) showing a moving object and provide a comprehensive physics breakdown tailored to the specified education level.

        ## Input Context

        - **Analysis Level:** {analysis_level}
        - Possible values: "beginner", "intermediate", "advanced", "expert"
        - This determines the depth, terminology, and mathematical rigor of your explanation

        ## Analysis Requirements

        ### Step 1: Object Identification

        **Identify the primary moving object in the video:**
        - What is the object? (Be specific: "basketball", "soccer ball", "wooden block", "pendulum bob", etc.)
        - Physical characteristics: approximate size, shape, material (if discernible)
        - Why is this the "main" object? (most significant motion, clearest physics demonstration, focal point)

        **Criteria for "main object":**
        - Exhibits the most obvious or interesting motion
        - Is the subject of the physical scenario being demonstrated
        - Has multiple forces or physics principles acting on it
        - If multiple objects present, select the one with richest physics interaction

        ### Step 2: Velocity Estimation

        **Analyze the object's motion characteristics:**

        **Qualitative Assessment (always provide):**
        - Speed: "slow", "moderate", "fast", "very fast"
        - Direction: "upward", "downward", "horizontal right", "diagonal up-left", "circular", etc.
        - Motion pattern: "constant speed", "accelerating", "decelerating", "oscillating", "parabolic trajectory"

        **Quantitative Estimation (if enough context exists):**
        - Approximate speed with units: "approximately 10-15 m/s"
        - Reference comparisons: "similar to a typical throw", "highway speed", "walking pace"
        - Basis for estimate: "based on trajectory arc", "estimated from frame displacement", "typical for this motion type"

        **Combined Example:**
        - "Fast downward motion, approximately 8-12 m/s, decelerating due to air resistance"
        - "Moderate horizontal velocity around 3-5 m/s with slight upward component"

        **If insufficient information:**
        - Still provide qualitative description
        - Note limitations: "Speed difficult to estimate without scale reference"

        ### Step 3: Physics Principle Identification

        **Determine the dominant physics principle(s):**

        **Primary Classification Options:**
        - **Projectile Motion**: Parabolic trajectory under gravity
        - **Free Fall**: Vertical motion under gravity alone
        - **Circular Motion**: Motion in circular path with centripetal acceleration
        - **Simple Harmonic Motion**: Oscillatory motion (pendulum, spring-mass system)
        - **Conservation of Energy**: Energy transformation (potential ↔ kinetic)
        - **Conservation of Momentum**: Collisions, interactions between objects
        - **Rotational Dynamics**: Rolling, spinning motion
        - **Friction/Drag**: Motion resisted by contact or fluid forces
        - **Elastic/Inelastic Collision**: Impact scenarios
        - **Uniform Motion**: Constant velocity, balanced forces
        - **Newton's Laws Application**: Force-acceleration relationship

        **Selection Guidelines:**
        - Choose the principle that BEST characterizes the motion
        - May list 2 principles if equally applicable (e.g., "Projectile Motion and Conservation of Energy")
        - Consider what a physics instructor would use this scenario to teach
        - Match to typical physics curriculum topics

        **Format:**
        - Primary principle name (e.g., "Projectile Motion")
        - Optional: Brief context (e.g., "Projectile Motion under constant gravitational acceleration")

        ### Step 4: Level-Appropriate Explanation

        Craft an explanation tailored precisely to {analysis_level}:

        #### For "beginner" level:
        **Target audience:** High school students, first-time physics learners, general public
        **Approach:**
        - Use everyday language and analogies
        - Avoid or carefully explain technical terms
        - Focus on "what" and "why" without heavy math
        - Length: 2-4 sentences
        - Example: "The ball follows a curved path because gravity constantly pulls it downward while it moves forward. At first, it goes up because of the initial throw, but gravity slows the upward motion until the ball starts falling back down. This creates the arc-shaped path we see."

        **Guidelines:**
        - Explain forces as "pushes and pulls"
        - Use comparisons: "like throwing a ball", "similar to a car slowing down"
        - Avoid equations entirely
        - Define any physics terms used: "velocity (speed in a direction)"

        #### For "intermediate" level:
        **Target audience:** AP Physics students, undergraduate intro physics, STEM enthusiasts
        **Approach:**
        - Use standard physics terminology without excessive jargon
        - Reference concepts they should know: acceleration, force, energy
        - May mention equations conceptually without solving them
        - Length: 3-5 sentences
        - Example: "The basketball undergoes projectile motion, experiencing constant downward acceleration due to gravity (9.8 m/s²) while maintaining approximately constant horizontal velocity. The parabolic trajectory results from the independence of horizontal and vertical motion components. Air resistance slightly reduces the horizontal velocity and total range, though this effect is relatively small for low-speed projectiles."

        **Guidelines:**
        - Mention Newton's laws, kinematic equations, energy principles by name
        - Discuss vector components when relevant
        - Explain cause-effect relationships: "Because X, therefore Y"
        - Reference typical values: "standard gravitational acceleration", "typical coefficient of friction"

        #### For "advanced" level:
        **Target audience:** Upper-level undergrad physics, engineering students, physics teachers
        **Approach:**
        - Use precise technical language and physics formalism
        - Reference specific equations and mathematical relationships
        - Discuss assumptions, approximations, and limiting cases
        - Length: 4-6 sentences
        - Example: "The projectile exhibits classical two-dimensional kinematics under uniform gravitational acceleration. The trajectory satisfies y = y₀ + v₀sinθ·t - ½gt², with horizontal motion given by x = x₀ + v₀cosθ·t. Energy considerations show continuous conversion between kinetic and gravitational potential energy, with total mechanical energy conserved in the absence of air resistance. The actual trajectory shows slight deviation from the ideal parabola due to quadratic drag, which introduces a velocity-dependent retarding force proportional to v²."

        **Guidelines:**
        - Use mathematical notation and equations explicitly
        - Discuss coordinate systems and reference frames
        - Mention energy, momentum, force analysis quantitatively
        - Address real-world complications: drag, friction coefficients, non-ideal conditions

        #### For "expert" level:
        **Target audience:** Graduate students, physics researchers, advanced practitioners
        **Approach:**
        - Employ rigorous mathematical and theoretical frameworks
        - Reference advanced concepts: Lagrangian mechanics, phase space, differential equations
        - Discuss subtleties, edge cases, and advanced treatment
        - Length: 5-8 sentences
        - Example: "The observed motion is well-described by the Newtonian equations of motion in a uniform gravitational field, ∑F = ma = -mg ĵ. The trajectory in phase space traces out a parabola in position-velocity coordinates. A Lagrangian formulation (L = T - V = ½m(ẋ² + ẏ²) - mgy) yields the same equations of motion via the Euler-Lagrange equations. Perturbative corrections from air drag introduce a nonlinear damping term proportional to v², which can be treated via numerical integration or asymptotic methods for Reynolds numbers typical of this scenario. The motion could alternatively be analyzed in a non-inertial reference frame co-moving with the object, introducing fictitious forces that must be accounted for in the force balance."

        **Guidelines:**
        - Reference advanced formulations: Hamiltonian, Lagrangian, tensors
        - Discuss limiting cases and dimensional analysis
        - Mention relevant dimensionless parameters: Reynolds number, Mach number
        - Consider alternative analytical approaches or approximation methods

        ### Explanation Quality Standards

        **For ALL levels:**
        - Be scientifically accurate - no oversimplifications that mislead
        - Focus on the most relevant physics for this specific scenario
        - Make it educational - what would a student learn from this?
        - Be concise but complete - every sentence should add value
        - Use active voice and clear structure
        - Proofread for clarity and flow

        **Avoid:**
        - Generic statements that could apply to any motion
        - Listing forces without explaining their roles
        - Overly complex terminology for the target level
        - Speculation beyond what's visible in the video
        - Redundancy or repetition

        ## Output Format Requirements

        **CRITICAL:** Return ONLY a valid JSON object. NO markdown formatting. NO ```json``` wrappers. NO explanatory text before or after.

        **JSON Structure:**
        {{
            "main_object": "Specific object name",
            "velocity_estimation": "Qualitative + quantitative motion description",
            "physics_principle": "Primary physics principle name",
            "explanation": "Level-appropriate explanation of the physics ({analysis_level} level)"
        }}

        **Field Specifications:**

        ### main_object (string)
        - Specific, concrete object name
        - Examples: "Basketball", "Metal sphere", "Wooden block on incline", "Pendulum bob", "Toy car"
        - NOT generic: avoid "ball" (specify type), "object", "thing"
        - Capitalize properly
        - Length: 1-5 words typically

        ### velocity_estimation (string)
        - Combined qualitative + quantitative description
        - Pattern: "[Speed descriptor] [direction], [additional details]"
        - Examples:
        - "Fast downward motion, approximately 12-15 m/s, decelerating"
        - "Moderate horizontal velocity around 5 m/s with slight rightward arc"
        - "Slow oscillating motion, peak speed ~2 m/s at lowest point"
        - "Very fast diagonal trajectory, estimated 20+ m/s based on blur"
        - Include direction, speed estimate, and motion characteristics
        - Length: 1-2 sentences or 10-25 words

        ### physics_principle (string)
        - Name of primary physics principle/concept
        - Use standard physics terminology from curriculum
        - Examples: "Projectile Motion", "Conservation of Energy", "Simple Harmonic Motion", "Uniform Circular Motion"
        - May list 2 principles separated by " and " if equally applicable
        - Capitalize each major word
        - Length: 2-8 words typically

        ### explanation (string)
        - Comprehensive physics analysis tailored to {analysis_level}
        - Follow level-specific guidelines above
        - Length varies by level:
        - Beginner: 2-4 sentences (~50-100 words)
        - Intermediate: 3-5 sentences (~80-150 words)
        - Advanced: 4-6 sentences (~120-200 words)
        - Expert: 5-8 sentences (~150-250 words)
        - Use proper grammar, punctuation, and scientific writing style
        - Focus on forces, energy, motion characteristics relevant to the scenario

        ## Validation Checklist

        Before outputting JSON, verify:

        ✅ **JSON Validity:**
        - Proper syntax: braces, commas, quotes
        - No trailing commas
        - Double quotes for strings (not single)
        - All four fields present

        ✅ **Content Quality:**
        - main_object is specific and accurate
        - velocity_estimation includes both qualitative and quantitative info
        - physics_principle matches the observed motion
        - explanation is appropriate for {analysis_level}
        - explanation is scientifically accurate

        ✅ **Level Appropriateness:**
        - Terminology matches target audience
        - Mathematical detail appropriate for level
        - Complexity calibrated correctly
        - No jargon above level, no oversimplification below level

        ✅ **Completeness:**
        - All four fields filled with substantive content
        - No placeholder text like "TBD" or "..."
        - Explanation addresses the specific motion observed
        - Sufficient detail for educational value

        ## Example Outputs

        ### Example 1: Basketball Shot (Intermediate Level)

        {{
            "main_object": "Basketball",
            "velocity_estimation": "Moderate upward and forward motion, approximately 8-10 m/s at release, following parabolic arc",
            "physics_principle": "Projectile Motion",
            "explanation": "The basketball undergoes projectile motion after leaving the player's hands, with its trajectory determined by the initial velocity components and gravitational acceleration. The horizontal velocity remains approximately constant (neglecting air resistance), while the vertical velocity decreases due to gravity until reaching the apex, then increases downward. The parabolic path results from the independent horizontal and vertical motions, with gravity providing constant downward acceleration of 9.8 m/s². The range and maximum height depend on the launch angle and initial speed."
        }}

        ### Example 2: Pendulum Swing (Beginner Level)

        {{
            "main_object": "Pendulum bob",
            "velocity_estimation": "Moderate swinging motion, fastest at the bottom (approximately 2-3 m/s), slowing at the ends",
            "physics_principle": "Simple Harmonic Motion and Conservation of Energy",
            "explanation": "The pendulum swings back and forth because gravity pulls it downward while the string keeps it moving in an arc. When the bob is at the highest points of the swing, it moves slowly and has stored energy (potential energy). As it swings down, it speeds up and that stored energy converts to movement energy (kinetic energy). At the bottom, it's moving fastest, then slows down again as it swings back up the other side. This process repeats, creating the regular back-and-forth motion we see."
        }}

        ### Example 3: Rolling Ball on Incline (Advanced Level)

        {{
            "main_object": "Solid sphere",
            "velocity_estimation": "Accelerating downward along incline, approximately 3-4 m/s and increasing, with both translational and rotational motion",
            "physics_principle": "Rotational Dynamics and Conservation of Energy",
            "explanation": "The sphere undergoes rolling motion down the incline, characterized by coupled translational and rotational kinematics satisfying the no-slip condition v = ωR. The acceleration down the incline is a = g sinθ / (1 + I/mR²), where I = (2/5)mR² for a solid sphere, yielding a = (5/7)g sinθ. Energy analysis shows gravitational potential energy mgh converting to both translational kinetic energy (½mv²) and rotational kinetic energy (½Iω²). The reduced acceleration compared to a sliding block (which would be g sinθ) results from the rotational inertia requiring torque from the friction force. Static friction at the contact point provides the necessary torque for rotation while preventing slipping, with fs = (2/7)mg sinθ for the solid sphere."
        }}

        ### Example 4: Collision (Expert Level)

        {{
            "main_object": "Steel ball bearing",
            "velocity_estimation": "Fast horizontal motion pre-collision at approximately 15-18 m/s, dramatic deceleration upon impact",
            "physics_principle": "Conservation of Momentum and Coefficient of Restitution",
            "explanation": "The collision between the steel sphere and target exemplifies conservation of momentum in the horizontal direction, with ∑p_before = ∑p_after when considering the sphere-target system. The coefficient of restitution e = (v₂' - v₁')/(v₁ - v₂) characterizes the energy dissipation, with e ≈ 0.8-0.9 expected for steel-steel contact. The collision duration Δt ≈ 10⁻⁴ s produces impulsive forces F·Δt = Δp of order 10³ N based on the momentum change. Detailed analysis requires considering the contact mechanics via Hertzian theory, accounting for elastic deformation at the contact patch with stresses σ ~ E(a/R)^(1/2), where E is Young's modulus. Energy balance shows kinetic energy partitioned into elastic potential energy during compression, then partially recovered during restitution, with losses due to plastic deformation and acoustic emission. The observed trajectory post-collision can be analyzed in the center-of-mass frame to separate relative and center-of-mass motion components."
        }}

        ## Edge Cases and Special Scenarios

        **If motion is unclear or ambiguous:**
        - Make best reasonable interpretation
        - Note any assumptions in explanation
        - Focus on most likely physics scenario

        **If object is nearly stationary:**
        - main_object: still identify the object
        - velocity_estimation: "Nearly stationary" or "Slow motion, less than 1 m/s"
        - physics_principle: May be "Static Equilibrium" or "Balanced Forces"
        - explanation: Discuss why it's not moving (balanced forces, friction, etc.)

        **If multiple physics principles apply equally:**
        - List both in physics_principle: "Projectile Motion and Air Resistance"
        - Address both in explanation
        - Prioritize based on what's most educational for the level

        **If video quality is poor:**
        - Still attempt analysis with visible information
        - May note uncertainty: "Appears to be...", "Likely experiencing..."
        - Focus on gross features that are clear

        ## Level Adaptation Summary

        | Level | Audience | Math | Terminology | Explanation Length |
        |-------|----------|------|-------------|-------------------|
        | beginner | High school intro, general public | None | Everyday language | 2-4 sentences |
        | intermediate | AP/Intro undergrad | Equations mentioned conceptually | Standard physics terms | 3-5 sentences |
        | advanced | Upper undergrad/engineering | Explicit equations, derivations | Technical, quantitative | 4-6 sentences |
        | expert | Graduate/research | Rigorous formalism | Advanced frameworks | 5-8 sentences |

        ---

        **Current analysis level: {analysis_level}**

        **Now analyze the provided video frame(s) and return your JSON output.**
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