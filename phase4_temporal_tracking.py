"""
Phase 4: Temporal Tracking & Memory System
This module implements sophisticated tracking with occlusion handling and re-identification.

Key Features:
- IoU-based detection-to-track matching
- Kalman filter for motion prediction
- Temporal memory buffer for re-identification
- Multi-target tracking without ID switching
"""

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from collections import deque
import time
from typing import List, Dict, Tuple, Optional


class KalmanBoxTracker:
    """
    Kalman Filter for tracking bounding boxes.
    Predicts where an object will be in the next frame based on its motion history.
    """
    
    count = 0  # Global counter for unique IDs
    
    def __init__(self, bbox):
        """
        Initialize Kalman filter for a bounding box
        
        Args:
            bbox: [x1, y1, x2, y2]
        """
        # State: [x_center, y_center, area, aspect_ratio, dx, dy, darea]
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix (how state evolves)
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],  # x = x + dx
            [0,1,0,0,0,1,0],  # y = y + dy
            [0,0,1,0,0,0,1],  # area = area + darea
            [0,0,0,1,0,0,0],  # aspect ratio stays same
            [0,0,0,0,1,0,0],  # dx constant
            [0,0,0,0,0,1,0],  # dy constant
            [0,0,0,0,0,0,1]   # darea constant
        ])
        
        # Measurement matrix (what we can observe)
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ])
        
        # Measurement noise
        self.kf.R[2:,2:] *= 10.0
        
        # Process noise
        self.kf.P[4:,4:] *= 1000.0
        self.kf.P *= 10.0
        
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        
        # Initialize state
        self.kf.x[:4] = self._convert_bbox_to_z(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
    
    def update(self, bbox):
        """Update the state with a new detection"""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._convert_bbox_to_z(bbox))
    
    def predict(self):
        """Predict the next position"""
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        
        self.time_since_update += 1
        self.history.append(self._convert_x_to_bbox(self.kf.x))
        
        return self.history[-1]
    
    def get_state(self):
        """Get current bounding box estimate"""
        return self._convert_x_to_bbox(self.kf.x)
    
    def _convert_bbox_to_z(self, bbox):
        """Convert [x1,y1,x2,y2] to [x_center, y_center, area, aspect_ratio]"""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.0
        y = bbox[1] + h/2.0
        area = w * h
        aspect_ratio = w / float(h) if h != 0 else 1.0
        return np.array([x, y, area, aspect_ratio]).reshape((4, 1))
    
    def _convert_x_to_bbox(self, x):
        """Convert [x_center, y_center, area, aspect_ratio] to [x1,y1,x2,y2]"""
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w if w != 0 else 1.0
        return np.array([
            x[0] - w/2.0,
            x[1] - h/2.0,
            x[0] + w/2.0,
            x[1] + h/2.0
        ]).reshape((1, 4))[0]


class TemporalMemory:
    """
    Stores semantic descriptions and visual features of tracks over time.
    Used for re-identification when tracks are lost.
    """
    
    def __init__(self, buffer_size=100):
        """
        Args:
            buffer_size: Number of frames to remember
        """
        self.buffer_size = buffer_size
        # Structure: {track_id: deque of (frame_num, bbox, description, features)}
        self.memories = {}
    
    def add(self, track_id, frame_num, bbox, description, features=None):
        """Add a memory for a track"""
        if track_id not in self.memories:
            self.memories[track_id] = deque(maxlen=self.buffer_size)
        
        self.memories[track_id].append({
            'frame': frame_num,
            'bbox': bbox,
            'description': description,
            'features': features,
            'timestamp': time.time()
        })
    
    def get_latest(self, track_id):
        """Get most recent memory for a track"""
        if track_id in self.memories and len(self.memories[track_id]) > 0:
            return self.memories[track_id][-1]
        return None
    
    def get_description_history(self, track_id, n=5):
        """Get last n descriptions for a track"""
        if track_id not in self.memories:
            return []
        
        descriptions = [m['description'] for m in list(self.memories[track_id])[-n:]]
        return descriptions
    
    def find_similar(self, description, threshold=0.5):
        """Find tracks with similar descriptions (simple text matching)"""
        similar = []
        query_words = set(description.lower().split())
        
        for track_id, memory_queue in self.memories.items():
            if len(memory_queue) == 0:
                continue
            
            latest = memory_queue[-1]
            memory_words = set(latest['description'].lower().split())
            
            # Simple Jaccard similarity
            intersection = query_words & memory_words
            union = query_words | memory_words
            
            if len(union) > 0:
                similarity = len(intersection) / len(union)
                if similarity >= threshold:
                    similar.append((track_id, similarity, latest['description']))
        
        return sorted(similar, key=lambda x: x[1], reverse=True)
    
    def cleanup_old(self, max_age_seconds=60):
        """Remove very old memories"""
        current_time = time.time()
        to_remove = []
        
        for track_id, memory_queue in self.memories.items():
            if len(memory_queue) > 0:
                latest = memory_queue[-1]
                if current_time - latest['timestamp'] > max_age_seconds:
                    to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.memories[track_id]


class AdvancedSemanticTracker:
    """
    Advanced tracking system with:
    - IoU-based matching
    - Kalman filter prediction
    - Temporal memory for re-identification
    - Occlusion handling
    """
    
    def __init__(self, detector, vlm, 
                 max_age=30, min_hits=3, iou_threshold=0.3,
                 description_interval=10):
        """
        Args:
            detector: YOLO detector from Phase 2
            vlm: VisionLanguageDescriptor from Phase 3
            max_age: Frames to keep alive without detection
            min_hits: Minimum detections before track is confirmed
            iou_threshold: Minimum IoU for matching
            description_interval: Generate description every N frames
        """
        self.detector = detector
        self.vlm = vlm
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.description_interval = description_interval
        
        self.trackers = []  # List of KalmanBoxTracker
        self.memory = TemporalMemory()
        self.frame_count = 0
        
        # Statistics
        self.stats = {
            'total_tracks': 0,
            'active_tracks': 0,
            'reidentifications': 0
        }
        
        print("✅ Advanced Semantic Tracker initialized!")
    
    def update(self, frame):
        """
        Main update function - process a frame
        
        Args:
            frame: Video frame (numpy array)
            
        Returns:
            List of active tracks with semantic information
        """
        self.frame_count += 1
        
        # Get detections
        detections = self.detector.detect(frame)
        det_bboxes = np.array([d['bbox'] for d in detections]) if detections else np.empty((0, 4))
        
        # Predict all tracker positions
        for t in self.trackers:
            t.predict()
        
        # Match detections to existing trackers
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
            det_bboxes, [t.get_state() for t in self.trackers]
        )
        
        # Update matched trackers
        for det_idx, trk_idx in matched:
            self.trackers[trk_idx].update(det_bboxes[det_idx])
        
        # Create new trackers for unmatched detections
        for det_idx in unmatched_dets:
            trk = KalmanBoxTracker(det_bboxes[det_idx])
            self.trackers.append(trk)
            self.stats['total_tracks'] += 1
        
        # Generate semantic descriptions (not every frame to save compute)
        should_describe = (self.frame_count % self.description_interval == 0)
        
        # Prepare return data
        active_tracks = []
        i = len(self.trackers)
        
        for trk in reversed(self.trackers):
            i -= 1
            bbox = trk.get_state()
            
            # Only return confirmed tracks
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                
                # Generate description if needed
                description = None
                if should_describe:
                    try:
                        description = self.vlm.describe_object(
                            frame, 
                            bbox.astype(int),
                            context=f"frame {self.frame_count}"
                        )
                    except Exception as e:
                        print(f"⚠️ Description failed: {e}")
                        description = "object"
                else:
                    # Use last known description
                    mem = self.memory.get_latest(trk.id)
                    description = mem['description'] if mem else "object"
                
                # Store in memory
                self.memory.add(trk.id, self.frame_count, bbox, description)
                
                track_info = {
                    'id': trk.id,
                    'bbox': bbox.astype(int).tolist(),
                    'description': description,
                    'hits': trk.hits,
                    'age': trk.age,
                    'time_since_update': trk.time_since_update
                }
                
                active_tracks.append(track_info)
            
            # Remove dead trackers
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        
        self.stats['active_tracks'] = len(active_tracks)
        
        # Cleanup old memories periodically
        if self.frame_count % 100 == 0:
            self.memory.cleanup_old()
        
        return active_tracks
    
    def _associate_detections_to_trackers(self, detections, trackers, iou_threshold=None):
        """
        Assigns detections to tracked objects using Hungarian algorithm
        
        Returns:
            matched: [(det_idx, trk_idx), ...]
            unmatched_detections: [det_idx, ...]
            unmatched_trackers: [trk_idx, ...]
        """
        if iou_threshold is None:
            iou_threshold = self.iou_threshold
        
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)
        
        if len(detections) == 0:
            return np.empty((0, 2), dtype=int), np.empty((0,), dtype=int), np.arange(len(trackers))
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self._iou(det, trk)
        
        # Hungarian algorithm for optimal assignment
        if min(iou_matrix.shape) > 0:
            matched_indices = linear_sum_assignment(-iou_matrix)
            matched_indices = np.array(list(zip(*matched_indices)))
        else:
            matched_indices = np.empty((0, 2), dtype=int)
        
        # Filter out matches with low IoU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                continue
            matches.append(m.reshape(1, 2))
        
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        
        # Find unmatched detections and trackers
        unmatched_detections = []
        for d in range(len(detections)):
            if d not in matches[:, 0]:
                unmatched_detections.append(d)
        
        unmatched_trackers = []
        for t in range(len(trackers)):
            if t not in matches[:, 1]:
                unmatched_trackers.append(t)
        
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
    
    def _iou(self, bbox1, bbox2):
        """
        Calculate Intersection over Union (IoU)
        This is the core metric for determining if two boxes refer to the same object
        
        IoU = Area of Overlap / Area of Union
        
        Args:
            bbox1, bbox2: [x1, y1, x2, y2]
            
        Returns:
            IoU score (0-1)
        """
        # Get coordinates
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection rectangle
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        # Check if there's intersection
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        # Calculate areas
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def search_by_description(self, query):
        """
        Search for tracks matching a description
        
        Args:
            query: Natural language description (e.g., "person in red jacket")
            
        Returns:
            List of matching tracks with similarity scores
        """
        return self.memory.find_similar(query, threshold=0.3)
    
    def visualize(self, frame, tracks, show_predictions=True):
        """
        Enhanced visualization with track IDs, descriptions, and trajectories
        
        Args:
            frame: Video frame
            tracks: List of track dicts
            show_predictions: Show predicted next position
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for track in tracks:
            track_id = track['id']
            bbox = track['bbox']
            description = track['description']
            
            x1, y1, x2, y2 = bbox
            
            # Generate consistent color for track
            color = self._get_color(track_id)
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            
            # Draw track ID
            id_label = f"ID:{track_id}"
            cv2.putText(annotated, id_label, (x1, y1 - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Draw description
            desc_label = description[:40] + "..." if len(description) > 40 else description
            cv2.putText(annotated, desc_label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(annotated, (center_x, center_y), 5, color, -1)
            
            # Draw trajectory (last 30 positions)
            mem_history = self.memory.get_description_history(track_id, n=30)
            if track_id in self.memory.memories:
                points = []
                for mem in list(self.memory.memories[track_id])[-30:]:
                    b = mem['bbox']
                    cx = int((b[0] + b[2]) / 2)
                    cy = int((b[1] + b[3]) / 2)
                    points.append((cx, cy))
                
                # Draw trajectory line
                for i in range(1, len(points)):
                    cv2.line(annotated, points[i-1], points[i], color, 2)
        
        # Draw statistics overlay
        stats_y = 30
        cv2.rectangle(annotated, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.putText(annotated, f"Frame: {self.frame_count}", (20, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated, f"Active Tracks: {len(tracks)}", (20, stats_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated, f"Total Tracks: {self.stats['total_tracks']}", (20, stats_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated, "Press 'q' to quit, 's' to save", (20, stats_y + 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return annotated
    
    def _get_color(self, track_id):
        """Generate consistent color for track ID"""
        np.random.seed(track_id)
        return tuple(map(int, np.random.randint(50, 255, 3)))


def demo_advanced_tracking():
    """
    Demonstration of advanced tracking with all features
    """
    print("\n" + "="*60)
    print("🎬 ADVANCED SEMANTIC TRACKING DEMO")
    print("="*60)
    
    # Import required modules
    import sys
    sys.path.append('src')
    
    from detection.yolo_detector import YOLODetector
    from vlm.semantic_descriptor import VisionLanguageDescriptor
    
    # Initialize components
    print("\n1️⃣ Loading YOLO detector...")
    yolo = YOLODetector(model_size='n')
    
    print("\n2️⃣ Loading Vision-Language Model...")
    vlm = VisionLanguageDescriptor(model_name="Qwen/Qwen2-VL-2B-Instruct")
    
    print("\n3️⃣ Creating Advanced Semantic Tracker...")
    tracker = AdvancedSemanticTracker(
        yolo, vlm,
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        description_interval=15  # Describe every 15 frames
    )
    
    print("\n4️⃣ Starting tracking...")
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Save snapshot")
    print("  'p' - Pause/Resume")
    print("  'r' - Reset all tracks")
    
    cap = cv2.VideoCapture(0)
    paused = False
    
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update tracker
            tracks = tracker.update(frame)
            
            # Visualize
            annotated = tracker.visualize(frame, tracks, show_predictions=True)
        else:
            annotated = frame.copy()
            cv2.putText(annotated, "PAUSED", (frame.shape[1]//2 - 100, frame.shape[0]//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        
        cv2.imshow('Advanced Semantic Tracking', annotated)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f'outputs/snapshot_{tracker.frame_count}.jpg', annotated)
            print(f"💾 Saved snapshot at frame {tracker.frame_count}")
        elif key == ord('p'):
            paused = not paused
            print("⏸️ Paused" if paused else "▶️ Resumed")
        elif key == ord('r'):
            tracker.trackers = []
            KalmanBoxTracker.count = 0
            print("🔄 Reset all tracks")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n✅ Demo completed!")
    print(f"   Total frames: {tracker.frame_count}")
    print(f"   Total tracks created: {tracker.stats['total_tracks']}")


if __name__ == "__main__":
    demo_advanced_tracking()
