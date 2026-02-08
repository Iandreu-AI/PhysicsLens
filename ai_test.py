import google.generativeai as genai
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
import re
from typing import Dict, List, Tuple, Optional, Any
from ultralytics import YOLO


def configure_gemini(api_key: str) -> None:
    genai.configure(api_key=api_key)


SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]


@st.cache_resource
def get_yolo_model():
    return YOLO('yolov8n.pt')


class ObjectTracker:
    
    def __init__(self):
        self.positions = []
        self.confidences = []
        self.kalman = self._init_kalman_filter()
        self.is_initialized = False
        self.velocity = (0, 0)
        self.max_history = 30
        
    def _init_kalman_filter(self):
        kalman = cv2.KalmanFilter(4, 2)
        
        kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        
        return kalman
    
    def update(self, centroid: Optional[Tuple[int, int]], confidence: float = 1.0):
        if centroid is None:
            if self.is_initialized:
                prediction = self.kalman.predict()
                pred_x = int(prediction[0])
                pred_y = int(prediction[1])
                self.positions.append((pred_x, pred_y))
                self.confidences.append(0.3)
            return
        
        if not self.is_initialized:
            self.kalman.statePre = np.array([
                [centroid[0]],
                [centroid[1]],
                [0],
                [0]
            ], dtype=np.float32)
            self.is_initialized = True
        
        prediction = self.kalman.predict()
        
        if len(self.positions) > 0:
            pred_x, pred_y = int(prediction[0]), int(prediction[1])
            distance = np.sqrt((centroid[0] - pred_x)**2 + (centroid[1] - pred_y)**2)
            
            if distance > 200:
                confidence *= 0.3
        
        measurement = np.array([
            [centroid[0]],
            [centroid[1]]
        ], dtype=np.float32)
        
        self.kalman.correct(measurement)
        
        self.positions.append(centroid)
        self.confidences.append(confidence)
        
        if len(self.positions) > self.max_history:
            self.positions = self.positions[-self.max_history:]
            self.confidences = self.confidences[-self.max_history:]
        
        if len(self.positions) >= 2:
            vx = self.positions[-1][0] - self.positions[-2][0]
            vy = self.positions[-1][1] - self.positions[-2][1]
            self.velocity = (vx, vy)
    
    def get_position(self) -> Optional[Tuple[int, int]]:
        if not self.positions:
            return None
        
        recent_positions = self.positions[-5:]
        recent_confidences = self.confidences[-5:]
        
        if not recent_confidences or sum(recent_confidences) == 0:
            return self.positions[-1]
        
        weighted_x = sum(p[0] * c for p, c in zip(recent_positions, recent_confidences))
        weighted_y = sum(p[1] * c for p, c in zip(recent_positions, recent_confidences))
        total_confidence = sum(recent_confidences)
        
        x = int(weighted_x / total_confidence)
        y = int(weighted_y / total_confidence)
        
        return (x, y)
    
    def get_predicted_position(self) -> Optional[Tuple[int, int]]:
        if not self.is_initialized:
            return None
        
        prediction = self.kalman.predict()
        x = int(prediction[0])
        y = int(prediction[1])
        
        return (x, y)
    
    def get_confidence(self) -> float:
        if not self.confidences:
            return 0.0
        
        weights = [0.5, 0.3, 0.15, 0.05]
        recent = self.confidences[-len(weights):]
        
        weighted_conf = sum(c * w for c, w in zip(reversed(recent), weights[:len(recent)]))
        total_weight = sum(weights[:len(recent)])
        
        return weighted_conf / total_weight
    
    def get_velocity(self) -> Tuple[float, float]:
        return self.velocity
    
    def get_trajectory(self) -> List[Tuple[int, int]]:
        return self.positions.copy()


def detect_object_centroid(frame: np.ndarray, 
                          prev_centroid: Optional[Tuple[int, int]] = None,
                          confidence_threshold: float = 0.25,
                          motion_threshold: int = 5) -> Optional[Tuple[int, int]]:
    
    frame_height, frame_width = frame.shape[:2]
    frame_area = frame_width * frame_height
    
    yolo_centroid = _detect_with_yolo(
        frame, frame_width, frame_height, frame_area,
        confidence_threshold, prev_centroid
    )
    if yolo_centroid is not None:
        return yolo_centroid
    
    motion_centroid = _detect_by_motion(frame, motion_threshold)
    if motion_centroid is not None and _validate_centroid(motion_centroid, frame_width, frame_height):
        return motion_centroid
    
    contour_centroid = _detect_by_advanced_contours(frame, prev_centroid)
    if contour_centroid is not None and _validate_centroid(contour_centroid, frame_width, frame_height):
        return contour_centroid
    
    color_centroid = _detect_by_multi_color_space(frame)
    if color_centroid is not None and _validate_centroid(color_centroid, frame_width, frame_height):
        return color_centroid
    
    edge_centroid = _detect_by_edges_and_shapes(frame)
    if edge_centroid is not None and _validate_centroid(edge_centroid, frame_width, frame_height):
        return edge_centroid
    
    bg_centroid = _detect_by_background_subtraction(frame)
    if bg_centroid is not None and _validate_centroid(bg_centroid, frame_width, frame_height):
        return bg_centroid
    
    if prev_centroid is not None:
        template_centroid = _detect_by_template_matching(frame, prev_centroid)
        if template_centroid is not None and _validate_centroid(template_centroid, frame_width, frame_height):
            return template_centroid
    
    flow_centroid = _detect_by_optical_flow(frame)
    if flow_centroid is not None and _validate_centroid(flow_centroid, frame_width, frame_height):
        return flow_centroid
    
    if prev_centroid is not None:
        return prev_centroid
    
    return (frame_width // 2, frame_height // 2)


def _detect_with_yolo(frame: np.ndarray,
                     frame_width: int,
                     frame_height: int,
                     frame_area: int,
                     confidence_threshold: float,
                     prev_centroid: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    
    model = get_yolo_model()
    
    PHYSICS_OBJECTS = {
        32: ('sports ball', 4.0), 33: ('kite', 2.5), 34: ('baseball bat', 2.0),
        35: ('baseball glove', 2.0), 36: ('skateboard', 2.5), 37: ('surfboard', 2.5),
        38: ('tennis racket', 2.0), 39: ('bottle', 2.0), 40: ('wine glass', 2.0),
        41: ('cup', 2.0), 42: ('fork', 1.5), 43: ('knife', 1.5), 44: ('spoon', 1.5),
        45: ('bowl', 2.0), 46: ('banana', 2.0), 47: ('apple', 2.5), 48: ('sandwich', 1.5),
        49: ('orange', 2.5), 2: ('car', 2.0), 3: ('motorcycle', 2.0), 5: ('bus', 1.5),
        7: ('truck', 1.5), 14: ('bird', 2.0), 15: ('cat', 1.8), 16: ('dog', 1.8),
        56: ('chair', 1.5), 60: ('dining table', 1.2), 63: ('laptop', 2.0),
        64: ('mouse', 2.5), 65: ('remote', 2.0), 66: ('keyboard', 1.5),
        67: ('cell phone', 2.5), 77: ('teddy bear', 2.0), 78: ('hair drier', 1.5),
        79: ('toothbrush', 2.0),
    }
    
    BLACKLIST = {0}
    
    results = model(frame, verbose=False, conf=confidence_threshold, iou=0.45)
    
    best_detection = None
    best_score = -1
    
    for result in results:
        boxes = result.boxes
        
        for box in boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if cls_id in BLACKLIST:
                continue
            
            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height
            area_ratio = box_area / frame_area
            aspect_ratio = box_width / box_height if box_height > 0 else 0
            
            if area_ratio > 0.30 or area_ratio < 0.003:
                continue
            
            if aspect_ratio < 0.1 or aspect_ratio > 10:
                continue
            
            centroid_x = (x1 + x2) // 2
            centroid_y = (y1 + y2) // 2
            
            edge_margin = min(frame_width, frame_height) * 0.02
            if (centroid_x < edge_margin or centroid_x > frame_width - edge_margin or
                centroid_y < edge_margin or centroid_y > frame_height - edge_margin):
                continue
            
            score = confidence
            
            if cls_id in PHYSICS_OBJECTS:
                _, multiplier = PHYSICS_OBJECTS[cls_id]
                score *= multiplier
            else:
                score *= 0.8
            
            if 0.02 <= area_ratio <= 0.15:
                score *= 1.3
            elif 0.15 < area_ratio <= 0.25:
                score *= 1.1
            
            if 0.7 <= aspect_ratio <= 1.4:
                score *= 1.2
            
            if prev_centroid is not None:
                prev_x, prev_y = prev_centroid
                distance = np.sqrt((centroid_x - prev_x)**2 + (centroid_y - prev_y)**2)
                max_movement = min(frame_width, frame_height) * 0.3
                
                if distance < max_movement:
                    proximity_bonus = 1.0 + (1.0 - distance / max_movement) * 0.5
                    score *= proximity_bonus
            
            if score > best_score:
                best_score = score
                best_detection = (centroid_x, centroid_y)
    
    return best_detection


def _detect_by_motion(frame: np.ndarray, threshold: int = 5) -> Optional[Tuple[int, int]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    
    if not hasattr(_detect_by_motion, 'prev_frame') or _detect_by_motion.prev_frame.shape != blurred.shape:
        _detect_by_motion.prev_frame = blurred
        return None
    
    frame_diff = cv2.absdiff(_detect_by_motion.prev_frame, blurred)
    _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _detect_by_motion.prev_frame = blurred
    
    if not contours:
        return None
    
    valid_contours = [c for c in contours if cv2.contourArea(c) > 100]
    if not valid_contours:
        return None
    
    largest_contour = max(valid_contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None
    
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def _detect_by_advanced_contours(frame: np.ndarray, 
                                 prev_centroid: Optional[Tuple[int, int]] = None) -> Optional[Tuple[int, int]]:
    
    scales = [1.0, 0.75, 0.5]
    all_centroids = []
    
    for scale in scales:
        scaled_frame = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1.0 else frame
        gray = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morph = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:
                continue
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            convexity = area / hull_area if hull_area > 0 else 0
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            extent = area / (w * h) if w * h > 0 else 0
            
            score = 0
            if circularity > 0.7:
                score += 3
            elif circularity > 0.5:
                score += 2
            elif circularity > 0.3:
                score += 1
            
            if convexity > 0.9:
                score += 2
            elif convexity > 0.7:
                score += 1
            
            if 0.7 <= aspect_ratio <= 1.3:
                score += 2
            elif 0.5 <= aspect_ratio <= 2.0:
                score += 1
            
            if extent > 0.7:
                score += 1
            
            if score >= 4:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"]) / scale
                    cy = int(M["m01"] / M["m00"]) / scale
                    all_centroids.append((int(cx), int(cy), score, area))
    
    if not all_centroids:
        return None
    
    if prev_centroid is not None:
        best_centroid = min(all_centroids, 
                          key=lambda c: np.sqrt((c[0] - prev_centroid[0])**2 + 
                                              (c[1] - prev_centroid[1])**2) - c[2] * 10)
    else:
        best_centroid = max(all_centroids, key=lambda c: c[2])
    
    return (best_centroid[0], best_centroid[1])


def _detect_by_multi_color_space(frame: np.ndarray) -> Optional[Tuple[int, int]]:
    candidates = []
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    hsv_ranges = [
        (np.array([0, 100, 100]), np.array([180, 255, 255])),
        (np.array([0, 120, 70]), np.array([10, 255, 255])),
        (np.array([170, 120, 70]), np.array([180, 255, 255])),
        (np.array([20, 100, 100]), np.array([80, 255, 255])),
        (np.array([90, 100, 100]), np.array([130, 255, 255])),
    ]
    
    for lower, upper in hsv_ranges:
        mask = cv2.inRange(hsv, lower, upper)
        centroid = _process_mask(mask, frame)
        if centroid:
            candidates.append(centroid)
    
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    _, l_channel, _ = cv2.split(lab)
    _, bright_mask = cv2.threshold(l_channel, 150, 255, cv2.THRESH_BINARY)
    centroid = _process_mask(bright_mask, frame)
    if centroid:
        candidates.append(centroid)
    
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    
    cr_thresh = cv2.threshold(cr, 135, 255, cv2.THRESH_BINARY)[1]
    cb_thresh = cv2.threshold(cb, 135, 255, cv2.THRESH_BINARY)[1]
    chroma_mask = cv2.bitwise_or(cr_thresh, cb_thresh)
    centroid = _process_mask(chroma_mask, frame)
    if centroid:
        candidates.append(centroid)
    
    _, s, _ = cv2.split(hsv)
    _, sat_mask = cv2.threshold(s, 80, 255, cv2.THRESH_BINARY)
    centroid = _process_mask(sat_mask, frame)
    if centroid:
        candidates.append(centroid)
    
    if not candidates:
        return None
    
    from collections import Counter
    grid_size = 50
    grid_candidates = [(c[0] // grid_size, c[1] // grid_size) for c in candidates]
    most_common_grid = Counter(grid_candidates).most_common(1)[0][0]
    
    filtered = [c for c, gc in zip(candidates, grid_candidates) if gc == most_common_grid]
    avg_x = sum(c[0] for c in filtered) // len(filtered)
    avg_y = sum(c[1] for c in filtered) // len(filtered)
    
    return (avg_x, avg_y)


def _process_mask(mask: np.ndarray, frame: np.ndarray) -> Optional[Tuple[int, int]]:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    valid_contours = [c for c in contours if cv2.contourArea(c) > 100]
    if not valid_contours:
        return None
    
    largest = max(valid_contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def _detect_by_edges_and_shapes(frame: np.ndarray) -> Optional[Tuple[int, int]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    circles = cv2.HoughCircles(
        filtered, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
        param1=50, param2=30, minRadius=10, maxRadius=200
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        largest_circle = max(circles[0], key=lambda c: c[2])
        return (int(largest_circle[0]), int(largest_circle[1]))
    
    edges = cv2.Canny(filtered, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    best_contour = None
    best_score = -1
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue
        
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        
        compactness = (perimeter * perimeter) / area
        size_score = 1.0 / (1.0 + abs(area - 5000) / 5000)
        compact_score = 1.0 / (1.0 + compactness / 20)
        
        score = size_score * compact_score
        
        if score > best_score:
            best_score = score
            best_contour = contour
    
    if best_contour is None:
        return None
    
    M = cv2.moments(best_contour)
    if M["m00"] == 0:
        return None
    
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    return (cx, cy)


def _detect_by_background_subtraction(frame: np.ndarray) -> Optional[Tuple[int, int]]:
    if not hasattr(_detect_by_background_subtraction, 'bg_subtractor'):
        _detect_by_background_subtraction.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )
    
    bg_subtractor = _detect_by_background_subtraction.bg_subtractor
    fg_mask = bg_subtractor.apply(frame)
    
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    valid_contours = [c for c in contours if cv2.contourArea(c) > 200]
    if not valid_contours:
        return None
    
    largest = max(valid_contours, key=cv2.contourArea)
    
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    return (cx, cy)


def _detect_by_template_matching(frame: np.ndarray, 
                                 prev_centroid: Tuple[int, int],
                                 search_radius: int = 100) -> Optional[Tuple[int, int]]:
    
    if not hasattr(_detect_by_template_matching, 'template'):
        return None
    
    template = _detect_by_template_matching.template
    
    prev_x, prev_y = prev_centroid
    frame_h, frame_w = frame.shape[:2]
    
    x1 = max(0, prev_x - search_radius)
    y1 = max(0, prev_y - search_radius)
    x2 = min(frame_w, prev_x + search_radius)
    y2 = min(frame_h, prev_y + search_radius)
    
    search_region = frame[y1:y2, x1:x2]
    
    if search_region.shape[0] < template.shape[0] or search_region.shape[1] < template.shape[1]:
        return None
    
    search_gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    result = cv2.matchTemplate(search_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    if max_val < 0.6:
        return None
    
    template_h, template_w = template.shape[:2]
    match_x = x1 + max_loc[0] + template_w // 2
    match_y = y1 + max_loc[1] + template_h // 2
    
    return (match_x, match_y)


def _detect_by_optical_flow(frame: np.ndarray) -> Optional[Tuple[int, int]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if not hasattr(_detect_by_optical_flow, 'prev_gray') or _detect_by_optical_flow.prev_gray.shape != gray.shape:
        _detect_by_optical_flow.prev_gray = gray
        return None
    
    prev_gray = _detect_by_optical_flow.prev_gray
    
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None, pyr_scale=0.5, levels=3,
        winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    threshold = np.percentile(mag, 95)
    motion_mask = (mag > threshold).astype(np.uint8) * 255
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    _detect_by_optical_flow.prev_gray = gray
    
    if not contours:
        return None
    
    valid_contours = [c for c in contours if cv2.contourArea(c) > 150]
    if not valid_contours:
        return None
    
    largest = max(valid_contours, key=cv2.contourArea)
    
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    return (cx, cy)


def _validate_centroid(centroid: Tuple[int, int], 
                      frame_width: int, 
                      frame_height: int) -> bool:
    x, y = centroid
    
    if x < 0 or x >= frame_width or y < 0 or y >= frame_height:
        return False
    
    margin = min(frame_width, frame_height) * 0.05
    
    if x < margin or x > frame_width - margin:
        return False
    
    if y < margin or y > frame_height - margin:
        return False
    
    return True


def update_template(frame: np.ndarray, centroid: Tuple[int, int], size: int = 50):
    x, y = centroid
    frame_h, frame_w = frame.shape[:2]
    
    half_size = size // 2
    x1 = max(0, x - half_size)
    y1 = max(0, y - half_size)
    x2 = min(frame_w, x + half_size)
    y2 = min(frame_h, y + half_size)
    
    template = frame[y1:y2, x1:x2]
    _detect_by_template_matching.template = template.copy()


class CoordinateTransformer:
    
    def __init__(self, frame_width: int, frame_height: int):
        self.frame_width = frame_width
        self.frame_height = frame_height
    
    def pixel_to_ai_grid(self, x: int, y: int) -> Tuple[int, int]:
        grid_x = int((x / self.frame_width) * 1000)
        grid_y = int((y / self.frame_height) * 1000)
        return (grid_x, grid_y)
    
    def ai_grid_to_pixel(self, grid_x: int, grid_y: int) -> Tuple[int, int]:
        x = int((grid_x / 1000) * self.frame_width)
        y = int((grid_y / 1000) * self.frame_height)
        return (x, y)
    
    def pixel_to_ratio(self, x: int, y: int) -> Tuple[float, float]:
        ratio_x = x / self.frame_width
        ratio_y = y / self.frame_height
        return (ratio_x, ratio_y)
    
    def ratio_to_pixel(self, ratio_x: float, ratio_y: float) -> Tuple[int, int]:
        x = int(ratio_x * self.frame_width)
        y = int(ratio_y * self.frame_height)
        return (x, y)


def get_physics_vectors(frame: np.ndarray, 
                       api_key: str,
                       prev_centroid: Optional[Tuple[int, int]] = None,
                       known_centroid: Optional[Tuple[int, int]] = None) -> Optional[Dict[str, Any]]:
    
    if known_centroid is not None:
        centroid = known_centroid
    else:
        centroid = detect_object_centroid(frame, prev_centroid=prev_centroid)
    
    if centroid is None:
        return None
    
    update_template(frame, centroid)
    
    frame_height, frame_width = frame.shape[:2]
    transformer = CoordinateTransformer(frame_width, frame_height)
    grid_x, grid_y = transformer.pixel_to_ai_grid(centroid[0], centroid[1])
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    prompt = f"""
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

    try:
        configure_gemini(api_key)
        model = genai.GenerativeModel('gemini-3-pro-preview')
        
        response = model.generate_content(
            [prompt, pil_image],
            safety_settings=SAFETY_SETTINGS
        )
        
        json_data = _robust_json_load(response.text)
        
        if json_data is None:
            return None
        
        if "vectors" in json_data:
            for vector in json_data["vectors"]:
                start_x, start_y = transformer.ai_grid_to_pixel(
                    vector["start_x"], vector["start_y"]
                )
                end_x, end_y = transformer.ai_grid_to_pixel(
                    vector["end_x"], vector["end_y"]
                )
                
                vector["start_x"] = start_x
                vector["start_y"] = start_y
                vector["end_x"] = end_x
                vector["end_y"] = end_y
        
        json_data["centroid"] = centroid
        
        return json_data
        
    except Exception as e:
        print(f"Error in get_physics_vectors: {e}")
        return None


def draw_vectors_with_debug(frame: np.ndarray, 
                            vector_data: Optional[Dict[str, Any]],
                            show_debug: bool = True) -> np.ndarray:
    """
    Draws vectors and centroid on the frame.
    Parameter show_debug: Controls whether to draw the blue dot at centroid.
    """
    if frame is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    output_frame = frame.copy()
    
    if not vector_data:
        return output_frame

    # Draw Centroid
    if "centroid" in vector_data and show_debug:
        cx, cy = vector_data["centroid"]
        cv2.circle(output_frame, (cx, cy), 5, (255, 0, 0), -1) # Blue dot
        cv2.circle(output_frame, (cx, cy), 15, (255, 255, 255), 1) # White ring
    
    # Draw Vectors
    if "vectors" in vector_data:
        for vector in vector_data["vectors"]:
            try:
                start_pt = (int(vector.get("start_x", 0)), int(vector.get("start_y", 0)))
                end_pt = (int(vector.get("end_x", 0)), int(vector.get("end_y", 0)))
                
                # Skip invalid vectors
                if start_pt == (0,0) and end_pt == (0,0):
                    continue

                color_hex = vector.get("color", "#FFFFFF")
                color_bgr = _hex_to_bgr(color_hex)
                
                # Arrow
                cv2.arrowedLine(output_frame, start_pt, end_pt, color_bgr, 4, tipLength=0.3)
                
                # Text Label
                label = vector.get("name", "Force")
                cv2.putText(output_frame, label, (end_pt[0] + 10, end_pt[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2, cv2.LINE_AA)
            except Exception:
                continue
    
    return output_frame


def _hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip('#')
    
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    
    return (b, g, r)


def _clean_json_response(text: str) -> str:
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    return text.strip()


def _robust_json_load(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    try:
        cleaned = _clean_json_response(text)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    try:
        cleaned = _clean_json_response(text)
        cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    return None


def analyze_physics_with_gemini(frames: List[np.ndarray],
                                api_key: str,
                                difficulty: str = "Student") -> Optional[Dict[str, Any]]:
    
    configure_gemini(api_key)
    
    pil_images = []
    for frame in frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_images.append(Image.fromarray(frame_rgb))
    
    prompt = _create_physics_prompt(difficulty)
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        content = [prompt] + pil_images
        
        response = model.generate_content(
            content,
            safety_settings=SAFETY_SETTINGS
        )
        
        json_data = _robust_json_load(response.text)
        
        return json_data
        
    except Exception as e:
        print(f"Error in analyze_physics_with_gemini: {e}")
        return None


def _create_physics_prompt(difficulty: str) -> str:
    
    base_prompt = """
 # SYSTEM ROLE — NON-NEGOTIABLE

**Identity:** Professor Lens v2.0 — Multimodal Physics & Mathematical Reasoning Engine  
**Specialization:** Classical Mechanics, Kinematics, Dynamics, Oscillations  
**Core Strengths:**  
- Writing **clean, correct LaTeX equations**
- Explaining physics **from first principles**
- Mapping **visual evidence → physical laws → equations**

You do NOT speculate.  
You do NOT invent phenomena.  
You ONLY explain what is supported by visual evidence and standard physics.

---

# CORE MISSION

Analyze the provided video frame(s) and reconstruct the **physical narrative** of the system:

1. What object is moving?
2. How is it moving?
3. Why is it moving that way?
4. Which laws govern this motion?
5. How are those laws expressed mathematically?

Your explanations must be **physically correct, mathematically rigorous, and pedagogically clear**.

---

# INTERNAL PHYSICS KNOWLEDGE BASE (REFERENCE ONLY)

## A. Force Taxonomy — Select ALL That Apply

### Fundamental
- Gravity (default on Earth)
- Electromagnetic (only if visually justified)

### Contact
- Normal Force
- Tension
- Friction (static / kinetic)

### Fluid
- Drag (air resistance)
- Buoyancy

### Rotational / Curvilinear
- Centripetal force
- Torque

### Applied
- Thrust
- Push / Pull

---

## B. Motion & Phenomena Library — Select EXACTLY ONE Primary Type

1. **Free Fall**  
   - $v = gt$, $y = \frac{1}{2}gt^2$

2. **Projectile Motion**  
   - $x = v_0 \cos\theta \, t$  
   - $y = v_0 \sin\theta \, t - \frac{1}{2}gt^2$

3. **Uniform Circular Motion**  
   - $a_c = \frac{v^2}{r}$

4. **Simple Harmonic Motion (Spring)**  
   - $F = -kx$  
   - $\omega = \sqrt{\frac{k}{m}}$

5. **Pendulum Motion (Small Angles)**  
   - $T = 2\pi\sqrt{\frac{L}{g}}$  
   - $\theta(t) = \theta_{\max}\cos(\omega t)$

6. **Inclined Plane Motion**  
   - $F_{\parallel} = mg\sin\theta$

7. **Collision (Elastic / Inelastic)**  
   - $\sum \vec{p}_{\text{initial}} = \sum \vec{p}_{\text{final}}$

8. **Terminal Velocity**  
   - $F_{\text{net}} = 0$

---

# ANALYSIS PROTOCOL (MANDATORY ORDER)

1. **Object Identification**  
   Identify the primary moving object.

2. **Visual Evidence Extraction**  
   Describe concrete visual cues (trajectory shape, contact, oscillation, rotation).

3. **Motion Classification**  
   Map the motion to ONE phenomenon from the library.

4. **Force Analysis**  
   List ALL forces acting on the object and justify each.

5. **Law Selection**  
   Identify the governing physical principles (Newton’s Laws, conservation laws, etc.).

6. **Mathematical Formalization**  
   Write the governing equations in **correct LaTeX**, using standard notation.

7. **Pedagogical Explanation**  
   Explain the physics clearly, adapting depth to **{{user_level}}**:
   - beginner
   - high-school
   - undergraduate
   - advanced

---

# LaTeX QUALITY RULES (STRICT)

- Use **proper subscripts**: `$F_g$, $F_N$, $v_0$, $a_c$`
- Use **fractions**, not inline divisions: `\frac{}`  
- Use **vector notation** when appropriate: `\vec{F}`, `\vec{v}`
- Do NOT mix scalar and vector equations incorrectly
- Every equation must correspond to the described physics

Incorrect or sloppy LaTeX is a critical failure.

---

# OUTPUT FORMAT — STRICT JSON ONLY

Return **ONLY** a valid JSON object.  
No Markdown. No explanations outside JSON.

```json
{
  "main_object": "Name of the primary object",
  "motion_type": "Exact name from the Phenomena Library",
  "visual_cues": "Specific visual observations supporting the classification",
  "active_forces": ["List of forces acting on the object"],
  "physics_principle": "Primary governing law or principle",
  "velocity_estimation": "Estimated velocity with units, if inferable",
  "key_formula": "Single most important equation (LaTeX)",
  "latex_equations": [
    "LaTeX equation 1",
    "LaTeX equation 2",
    "LaTeX equation 3"
  ],
  "explanation": "Clear, structured explanation (3–5 sentences), adapted to user level",
  "confidence_score": 0.0
}
"""

    return base_prompt

def get_chat_response_stream(chat_history: List[Dict[str, str]],
                            analysis_context: Optional[Dict[str, Any]],
                            api_key: str,
                            user_input: str):
    
    configure_gemini(api_key)
    
    system_prompt = """ System Role
**Identity:** Professor Lens (The AI Core of PhysicsLens).
**Archetype:** A blend of **Richard Feynman** (Intuitive, Analogy-heavy) and **Neil deGrasse Tyson** (Enthusiastic, engaging).
**Mission:** Transform raw visual data into "Aha!" moments. You do not just answer questions; you build intuition by connecting what the user *saw* to the laws of physics.

# Context: The Active Experiment
The user is looking at a video they just recorded. The computer vision system has detected:
<current_experiment>
    <object>{{object_name}}</object>
    <phenomena>{{phenomena_type}}</phenomena>
    <key_data>{{telemetry_summary}}</key_data> <!-- e.g. "Max Velocity: 5m/s" -->
    <user_level>{{user_level}}</user_level> <!-- e.g. "Child", "High School", "Undergrad" -->
</current_experiment>

# Pedagogical Protocol (The "Lens" Method)
1.  **Visual Anchor:** Always start by referencing the video or vectors. (e.g., "Did you see how the green arrow got longer?")
2.  **The Analogy Bridge:** Explain the concept using a real-world comparison that fits the `{{user_level}}`.
3.  **The Formal Logic:** Only introduce the equation *after* the intuition is set.

# Interaction Guidelines
*   **Tone:** Enthusiastic but rigorous. Never condescending.
*   **Conciseness:** Keep generic responses to 2-3 sentences. Go deeper only if asked.
*   **Math Formatting:** ALL variables and formulas must be wrapped in LaTeX (e.g., $F = ma$, $\vec{v}$).
*   **Safety:** If the user asks about dangerous experiments, pivot to the physics principles without encouraging unsafe behavior.

# Audience Adaptation Rules
*   **If Child:** Use "magic" and "forces" terminology. No algebra. Analogy: Playgrounds, Toys.
*   **If High School:** Use standard kinematic terms (Velocity, Acceleration). Basic Algebra. Analogy: Cars, Sports.
*   **If Expert:** Use vector calculus terms (Derivative, Momentum). Discuss limitations of the data.

# Current Task
Respond as Professor Lens, strictly adhering to the user's level.
"""

    if analysis_context:
        context_info = f"""
EXPERIMENT CONTEXT:
- Object: {analysis_context.get('main_object', 'Unknown')}
- Motion: {analysis_context.get('motion_type', 'Unknown')}
- Forces: {', '.join(analysis_context.get('active_forces', []))}
- Principle: {analysis_context.get('physics_principle', 'Unknown')}
"""
        system_prompt += context_info
    
    recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
    
    messages = []
    for msg in recent_history:
        role = "user" if msg["role"] == "user" else "model"
        messages.append({"role": role, "parts": [msg["content"]]})
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash', system_instruction=system_prompt)
        chat = model.start_chat(history=messages)
        
        response = chat.send_message(user_input, safety_settings=SAFETY_SETTINGS, stream=True)
        
        for chunk in response:
            if chunk.text:
                yield chunk.text
                
    except Exception as e:
        yield f"I apologize, but I encountered an error: {str(e)}"


def extract_keyframes(video_path: str, num_frames: int = 5) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    cap.release()
    return frames


if __name__ == "__main__":
    print("Physics Video Analysis Module")
    print("Multi-strategy object detection, physics vectors, AI analysis")