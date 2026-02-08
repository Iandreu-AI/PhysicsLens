import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict


class EnhancedPhysicsFrameSelector:
    """
    Production-grade semantic frame selection with STRICT human rejection.
    
    Key improvements:
    1. ZERO TOLERANCE for person detections (Class ID 0)
    2. Strict area and aspect ratio filtering
    3. Prioritizes sports balls and physics objects
    4. Multiple fallback strategies
    5. Detailed debug logging
    """
    
    # Expanded physics-relevant classes
    PHYSICS_OBJECTS = {
        # Projectiles (PRIORITY - Baseball related)
        'sports ball': 32,        # ⚾ Baseball, tennis ball, etc. - TOP PRIORITY
        'baseball bat': 39,       # Baseball bat
        'baseball glove': 40,     # Baseball glove
        
        # Other projectiles
        'frisbee': 29, 'apple': 47, 'orange': 49,
        'tennis racket': 38,
        
        # Containers & liquids
        'bottle': 39, 'wine glass': 40, 'cup': 41, 'bowl': 45,
        
        # Vehicles
        'car': 2, 'motorcycle': 3, 'bicycle': 1, 'skateboard': 41,
        
        # Tools & objects
        'umbrella': 25, 'handbag': 26, 'suitcase': 28, 'kite': 33,
        'clock': 74, 'vase': 75, 'scissors': 76, 'teddy bear': 77,
        'book': 73, 'cell phone': 67, 'remote': 65, 'mouse': 64,
        
        # Animals
        'bird': 14, 'cat': 15, 'dog': 16,
    }
    
    # CRITICAL: Baseball (sports ball) is class 32 in COCO
    BASEBALL_CLASS = 32
    PERSON_CLASS = 0  # HUMAN CLASS - MUST BE REJECTED
    
    def __init__(
        self,
        confidence_threshold=0.15,
        max_area_ratio=0.25,  # STRICT: Max 25% of frame (blocks humans)
        min_area_ratio=0.005,  # Min 0.5% of frame (filters noise)
        min_aspect_ratio=0.3,  # Reject tall rectangles (human signature)
        max_aspect_ratio=3.0,  # Reject very wide objects
        enable_tracking=True,
        enable_ensemble=False,
        debug=False
    ):
        self.conf_threshold = confidence_threshold
        self.max_area_ratio = max_area_ratio
        self.min_area_ratio = min_area_ratio
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.enable_tracking = enable_tracking
        self.enable_ensemble = enable_ensemble
        self.debug = debug
        
        print("Loading YOLO model...")
        self.model = YOLO('yolov8n.pt')
        self.model.conf = 0.1  # Force even lower internal threshold
        
        if enable_ensemble:
            try:
                self.model_small = YOLO('yolov8s.pt')
                self.has_ensemble = True
                print("✓ Ensemble mode enabled")
            except:
                self.has_ensemble = False
                print("⚠ Ensemble unavailable, using nano only")
        else:
            self.has_ensemble = False
        
        self.detection_count = 0
        self.rejection_reasons = defaultdict(int)
        
    def _is_valid_detection(self, result, frame_area, frame_shape):
        """
        STRICT validation with 3-layer human rejection.
        
        Returns:
            (is_valid, box, class_id, centroid, confidence_score)
        """
        if len(result.boxes) == 0:
            self.rejection_reasons['no_boxes'] += 1
            return False, None, None, None, 0.0
        
        frame_h, frame_w = frame_shape[:2]
        
        # === LAYER 1: HUMAN REJECTION (CLASS BLACKLIST) ===
        # CRITICAL: Filter out ALL person detections immediately
        non_human_boxes = [
            box for box in result.boxes 
            if int(box.cls[0]) != self.PERSON_CLASS
        ]
        
        if len(non_human_boxes) == 0:
            self.rejection_reasons['all_humans_rejected'] += 1
            if self.debug:
                print("  ❌ ALL DETECTIONS WERE HUMANS - Frame rejected")
            return False, None, None, None, 0.0
        
        # Find best detection (prioritize physics objects)
        candidates = []
        
        for box in non_human_boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height
            
            area_ratio = box_area / frame_area
            aspect_ratio = box_width / max(box_height, 1)
            
            # === LAYER 2: AREA FILTER ===
            # Reject boxes that are too large (>25% likely human/background)
            # or too small (<0.5% noise)
            if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
                if self.debug and area_ratio > self.max_area_ratio:
                    print(f"  ❌ REJECTED: Area {area_ratio:.3f} TOO LARGE (likely human in background)")
                continue
            
            # === LAYER 3: ASPECT RATIO CHECK ===
            # Reject tall rectangles (human signature: height >> width)
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                if self.debug and aspect_ratio < 0.5:
                    print(f"  ❌ REJECTED: Aspect ratio {aspect_ratio:.2f} (vertical = human-like)")
                continue
            
            # === SCORING SYSTEM ===
            # Boost score if it's a known physics object
            score = conf
            
            if cls_id in self.PHYSICS_OBJECTS.values():
                score *= 2.0  # Prioritize known physics objects
            
            # EXTRA boost for baseball (sports ball class 32)
            if cls_id == self.BASEBALL_CLASS:
                score *= 3.0  # Highest priority for baseball
                if self.debug:
                    print(f"    ⚾ BASEBALL detected! conf={conf:.2f}, boosted_score={score:.2f}")
            
            candidates.append({
                'box': [x1, y1, x2, y2],
                'cls_id': cls_id,
                'conf': conf,
                'score': score,
                'area_ratio': area_ratio,
                'aspect_ratio': aspect_ratio
            })
        
        if len(candidates) == 0:
            self.rejection_reasons['no_valid_non_human'] += 1
            return False, None, None, None, 0.0
        
        # Pick best candidate
        best = max(candidates, key=lambda x: x['score'])
        
        x1, y1, x2, y2 = best['box']
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        self.detection_count += 1
        
        if self.debug and self.detection_count % 10 == 0:
            class_name = "sports_ball(⚾)" if best['cls_id'] == 32 else f"class_{best['cls_id']}"
            print(f"  Detection #{self.detection_count}: {class_name}, "
                  f"conf={best['conf']:.2f}, area={best['area_ratio']:.4f}, "
                  f"aspect={best['aspect_ratio']:.2f}")
        
        return True, np.array(best['box']), best['cls_id'], (cx, cy), best['conf']
    
    def optimize_frames(self, video_path):
        """
        Robust detection pipeline with human rejection and multiple fallback strategies.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_area = width * height
        
        print(f"Video: {total_frames} frames, {width}x{height}, {fps:.1f} FPS")
        
        valid_detections = []
        frame_idx = 0
        
        # Sample EVERY frame if video is short
        sample_rate = 1 if total_frames < 100 else 2
        
        print(f"Analyzing frames (sampling every {sample_rate})...")
        print("🚫 HUMAN REJECTION ACTIVE - Only tracking physics objects")
        
        TRIM_IN = 18
        TRIM_OUT = 12

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # === EXCLUSION LOGIC ===
            # Skip the first 18 frames AND the last 12 frames
            if frame_idx < TRIM_IN or frame_idx > (total_frames - TRIM_OUT):
                frame_idx += 1
                continue
            # =======================

            if frame_idx % sample_rate == 0:
                # Run detection with very low confidence threshold
                results = self.model(frame, conf=0.1, verbose=False)[0]
            
            if frame_idx % sample_rate == 0:
                # Run detection with very low confidence threshold
                results = self.model(frame, conf=0.1, verbose=False)[0]
                
                is_valid, box, cls_id, centroid, conf = self._is_valid_detection(
                    results, frame_area, frame.shape
                )
                
                if is_valid:
                    valid_detections.append({
                        'idx': frame_idx,
                        'box': box,
                        'class_id': cls_id,
                        'cx': centroid[0],
                        'cy': centroid[1],
                        'confidence': conf
                    })
            
            frame_idx += 1
            
            # Progress indicator
            if frame_idx % 100 == 0:
                print(f"  Processed {frame_idx}/{total_frames} frames, "
                      f"found {len(valid_detections)} detections")
        
        cap.release()
        
        print(f"\n✓ Detection complete: {len(valid_detections)} valid frames")
        
        if self.debug:
            print("\n=== Rejection Statistics ===")
            for reason, count in self.rejection_reasons.items():
                print(f"  {reason}: {count}")
        
        # CRITICAL: If we found very few detections, use fallback strategy
        if len(valid_detections) < 3:
            print("\n⚠ WARNING: Few detections found, using uniform sampling fallback")
            return self._fallback_uniform_sampling(video_path, fps, total_frames)
        
        # Select kinematic triad
        final_indices = self._select_kinematic_triad(
            valid_detections, total_frames
        )
        
        # Extract final frames
        results = []
        labels = ["Entry (Impulse)", "Peak (Max Displacement)", "Exit (Impact)"]
        
        cap = cv2.VideoCapture(video_path)
        for i, target_idx in enumerate(final_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results.append({
                    'frame': frame_rgb,
                    'frame_bgr': frame,
                    'timestamp': target_idx / fps,
                    'label': labels[i],
                    'frame_index': target_idx
                })
        
        cap.release()
        
        # Verify frames aren't empty
        if all(r['frame'] is not None for r in results):
            print("✓ All frames extracted successfully")
        else:
            print("⚠ Some frames may be missing")
        
        return results
    
    def _fallback_uniform_sampling(self, video_path, fps, total_frames):
        """
        Fallback: Extract frames at fixed intervals when detection fails.
        """
        print("Using uniform sampling: 20%, 50%, 80% through video")
        
        indices = [
            int(total_frames * 0.2),
            int(total_frames * 0.5),
            int(total_frames * 0.8)
        ]
        
        cap = cv2.VideoCapture(video_path)
        results = []
        labels = ["Entry (Start)", "Middle (Peak)", "Exit (End)"]
        
        for i, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results.append({
                    'frame': frame_rgb,
                    'frame_bgr': frame,
                    'timestamp': idx / fps,
                    'label': labels[i],
                    'frame_index': idx
                })
        
        cap.release()
        return results
    
    def _select_kinematic_triad(self, detections, total_frames):
        """
        Select 3 representative frames from detections.
        """
        if len(detections) < 3:
            # Not enough detections, space them out
            if len(detections) == 1:
                d = detections[0]
                return [d['idx'], d['idx'], d['idx']]
            elif len(detections) == 2:
                return [detections[0]['idx'], detections[0]['idx'], detections[1]['idx']]
        
        # Sort by frame index
        detections.sort(key=lambda x: x['idx'])
        
        # Strategy: Start, furthest from start, end
        start_det = detections[0]
        end_det = detections[-1]
        
        # Find frame with maximum displacement from start
        start_x, start_y = start_det['cx'], start_det['cy']
        max_dist = 0
        peak_det = detections[len(detections) // 2]  # Default to middle
        
        for det in detections:
            dist = np.sqrt(
                (det['cx'] - start_x)**2 + 
                (det['cy'] - start_y)**2
            )
            if dist > max_dist:
                max_dist = dist
                peak_det = det
        
        idx1 = start_det['idx']
        idx2 = peak_det['idx']
        idx3 = end_det['idx']
        
        # Ensure separation
        min_gap = max(10, total_frames // 20)
        
        if idx2 - idx1 < min_gap:
            idx2 = min(idx1 + min_gap, total_frames - min_gap - 1)
        if idx3 - idx2 < min_gap:
            idx3 = min(idx2 + min_gap, total_frames - 1)
        
        # Clamp to valid range
        idx2 = min(max(idx2, 0), total_frames - 1)
        idx3 = min(max(idx3, 0), total_frames - 1)
        
        print(f"Selected frames: {idx1}, {idx2}, {idx3}")
        print(f"  Displacement: {max_dist:.1f} pixels")
        
        return [idx1, idx2, idx3]


# Public API
def optimize_frames(video_path, debug=False):
    """
    Robust frame selection with automatic human rejection and fallbacks.
    
    Args:
        video_path: Path to physics experiment video
        debug: Enable debug logging
        
    Returns:
        List of 3 frame dictionaries with RGB frames and metadata
    """
    selector = EnhancedPhysicsFrameSelector(debug=debug)
    return selector.optimize_frames(video_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python frame_optimizer.py <video_path> [--debug]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    debug = '--debug' in sys.argv
    
    print(f"Processing: {video_path}\n")
    print("=" * 60)
    print("HUMAN REJECTION LOGIC ENABLED")
    print("=" * 60)
    frames = optimize_frames(video_path, debug=debug)
    
    print("\n=== Selected Frames ===")
    for frame_data in frames:
        print(f"{frame_data['label']:30} @ {frame_data['timestamp']:6.2f}s "
              f"(Frame {frame_data['frame_index']})")
    
    # Save frames with visual confirmation
    print("\nSaving frames...")
    for i, frame_data in enumerate(frames):
        output_name = f"frame_{i+1}_{frame_data['label'].split()[0].lower()}.jpg"
        success = cv2.imwrite(output_name, frame_data['frame_bgr'])
        if success:
            h, w = frame_data['frame_bgr'].shape[:2]
            print(f"✓ {output_name} ({w}x{h})")
        else:
            print(f"✗ Failed to save {output_name}")
    
    print("\n✓ Processing complete!")