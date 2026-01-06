"""
Phase 3: Vision-Language Model Integration
This module adds "semantic understanding" to your tracking system.

Instead of tracking "box #5", we track "person in red jacket carrying backpack"
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import numpy as np
import cv2
from typing import List, Dict, Optional
import json
from pathlib import Path


class VisionLanguageDescriptor:
    """
    This class uses a Vision-Language Model to create semantic descriptions
    of detected objects. This helps maintain tracking even when objects
    are temporarily occluded.
    """
    
    def __init__(self, model_name="Qwen/Qwen2-VL-2B-Instruct", device='auto'):
        """
        Initialize the VLM
        
        Args:
            model_name: Hugging Face model name
                       Options: 
                       - "Qwen/Qwen2-VL-2B-Instruct" (smaller, faster)
                       - "Qwen/Qwen2-VL-7B-Instruct" (larger, more accurate)
            device: 'auto', 'cuda', or 'cpu'
        """
        print("🧠 Initializing Vision-Language Model...")
        print(f"   Model: {model_name}")
        
        self.device = self._setup_device(device)
        print(f"   Device: {self.device}")
        
        # Load model and processor
        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map=self.device
            )
            self.processor = AutoProcessor.from_pretrained(model_name)
            print("✅ VLM loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading VLM: {e}")
            print("\n💡 First time? The model needs to be downloaded (2-7GB)")
            print("   This is normal and only happens once.")
            raise
    
    def _setup_device(self, device):
        """Determine which device to use"""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def describe_object(self, image, bbox, context=""):
        """
        Generate a detailed description of an object in the image
        
        Args:
            image: Image as numpy array (BGR from OpenCV)
            bbox: Bounding box [x1, y1, x2, y2]
            context: Additional context (e.g., "in a busy city", "in a forest")
            
        Returns:
            String description of the object
        """
        # Crop the object from the image
        x1, y1, x2, y2 = bbox
        cropped = image[y1:y2, x1:x2]
        
        # Convert BGR to RGB
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cropped_rgb)
        
        # Create prompt
        prompt = f"""Describe this object in detail. Focus on:
- What type of object it is
- Distinctive visual features (color, shape, patterns)
- Any unique characteristics that would help identify it later
{f"Context: {context}" if context else ""}

Be concise but specific. Format: [object type] with [key features]"""
        
        # Prepare inputs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Process
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=[pil_image],
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate description
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False
            )
        
        # Decode
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        
        description = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]
        
        return description.strip()
    
    def batch_describe(self, image, detections, context=""):
        """
        Generate descriptions for multiple objects at once (more efficient)
        
        Args:
            image: Full image
            detections: List of detection dicts with 'bbox' keys
            context: Environmental context
            
        Returns:
            List of descriptions in same order as detections
        """
        descriptions = []
        
        for det in detections:
            try:
                desc = self.describe_object(image, det['bbox'], context)
                descriptions.append(desc)
            except Exception as e:
                print(f"⚠️ Warning: Failed to describe object: {e}")
                descriptions.append(f"{det.get('class', 'object')} (no description)")
        
        return descriptions


class SemanticTracker:
    """
    Combines YOLO detection with VLM descriptions for semantic tracking.
    This is the core of your "reasoning tracker" system.
    """
    
    def __init__(self, yolo_detector, vlm_descriptor, max_age=30):
        """
        Args:
            yolo_detector: YOLODetector instance from Phase 2
            vlm_descriptor: VisionLanguageDescriptor instance
            max_age: Frames to keep track alive without detection
        """
        self.detector = yolo_detector
        self.vlm = vlm_descriptor
        self.max_age = max_age
        
        # Track storage: {track_id: track_info}
        self.tracks = {}
        self.next_id = 1
        
        print("✅ Semantic Tracker initialized!")
    
    def update(self, frame, generate_descriptions=True):
        """
        Process a frame and update tracks
        
        Args:
            frame: Video frame (numpy array)
            generate_descriptions: Whether to generate VLM descriptions
                                  (disable for speed testing)
        
        Returns:
            List of active tracks with semantic information
        """
        # Get detections from YOLO
        detections = self.detector.detect(frame)
        
        # Generate semantic descriptions if enabled
        if generate_descriptions and detections:
            descriptions = self.vlm.batch_describe(frame, detections)
            
            # Add descriptions to detections
            for det, desc in zip(detections, descriptions):
                det['description'] = desc
        else:
            for det in detections:
                det['description'] = det.get('class', 'unknown')
        
        # Simple tracking: match detections to existing tracks
        # (Phase 4 will add sophisticated IoU-based matching)
        updated_tracks = self._simple_match(detections)
        
        # Update track ages
        self._update_ages()
        
        return updated_tracks
    
    def _simple_match(self, detections):
        """
        Simple nearest-neighbor matching for demonstration
        In Phase 4, we'll implement proper IoU-based matching
        """
        updated = []
        
        for det in detections:
            # For now, just create new tracks for each detection
            track_id = self.next_id
            self.next_id += 1
            
            track_info = {
                'id': track_id,
                'bbox': det['bbox'],
                'class': det['class'],
                'confidence': det['confidence'],
                'description': det['description'],
                'age': 0,
                'hits': 1
            }
            
            self.tracks[track_id] = track_info
            updated.append(track_info)
        
        return updated
    
    def _update_ages(self):
        """Remove old tracks"""
        to_remove = []
        for track_id, track in self.tracks.items():
            track['age'] += 1
            if track['age'] > self.max_age:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]
    
    def visualize(self, frame, tracks):
        """
        Draw tracks with semantic descriptions on frame
        
        Args:
            frame: Video frame
            tracks: List of track dictionaries
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            track_id = track['id']
            description = track['description']
            
            # Generate color based on track ID
            color = self._get_color(track_id)
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = f"ID:{track_id} | {description[:50]}"
            
            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated
    
    def _get_color(self, track_id):
        """Generate consistent color for track ID"""
        np.random.seed(track_id)
        return tuple(map(int, np.random.randint(0, 255, 3)))
    
    def export_tracks(self, output_path):
        """Export track information to JSON"""
        track_data = {
            str(k): {
                'id': v['id'],
                'description': v['description'],
                'class': v['class'],
                'last_seen': v['age']
            }
            for k, v in self.tracks.items()
        }
        
        with open(output_path, 'w') as f:
            json.dump(track_data, f, indent=2)
        
        print(f"✅ Tracks exported to {output_path}")


def demo_semantic_tracking():
    """
    Demonstration script showing how to use the semantic tracker
    """
    print("\n" + "="*60)
    print("🎬 SEMANTIC TRACKING DEMO")
    print("="*60)
    
    # Import YOLO from Phase 2
    from detection.yolo_detector import YOLODetector
    
    # Initialize components
    print("\n1️⃣ Loading YOLO detector...")
    yolo = YOLODetector(model_size='n')  # Use nano for speed
    
    print("\n2️⃣ Loading Vision-Language Model...")
    print("   (This may take a few minutes on first run)")
    vlm = VisionLanguageDescriptor(model_name="Qwen/Qwen2-VL-2B-Instruct")
    
    print("\n3️⃣ Creating Semantic Tracker...")
    tracker = SemanticTracker(yolo, vlm)
    
    print("\n4️⃣ Starting webcam tracking...")
    print("   Press 'q' to quit")
    print("   Press 's' to save current tracks")
    
    cap = cv2.VideoCapture(0)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 5 frames (to reduce VLM load)
        generate_desc = (frame_count % 5 == 0)
        
        # Update tracks
        tracks = tracker.update(frame, generate_descriptions=generate_desc)
        
        # Visualize
        annotated = tracker.visualize(frame, tracks)
        
        # Add info text
        cv2.putText(annotated, f"Frame: {frame_count} | Tracks: {len(tracks)}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated, "Press 'q' to quit, 's' to save",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        cv2.imshow('Semantic Tracking', annotated)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            tracker.export_tracks('outputs/tracks.json')
            cv2.imwrite('outputs/snapshot.jpg', annotated)
            print("💾 Saved snapshot and tracks!")
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n✅ Demo completed!")
    print(f"   Total frames processed: {frame_count}")
    print(f"   Active tracks: {len(tracker.tracks)}")


if __name__ == "__main__":
    # Make sure you're in the project root directory
    import sys
    sys.path.append('src')
    
    demo_semantic_tracking()
