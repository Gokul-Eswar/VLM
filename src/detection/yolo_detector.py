"""
YOLO Detector Module
This handles real-time object detection - the "eyes" of our system
"""

import sys
try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Error: 'ultralytics' module not found.")
    print("Please install dependencies using: pip install -r requirements.txt")
    sys.exit(1)

import cv2
import torch
import numpy as np
from pathlib import Path
import os

class YOLODetector:
    def __init__(self, model_size='n', device='auto'):
        """
        Initialize YOLO detector

        Args:
            model_size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
            device: 'auto', 'cpu', or 'cuda'
        """
        self.device = self._setup_device(device)
        self.model = self._load_model(model_size)
        print(f"✅ YOLO detector initialized on {self.device}")

    def _setup_device(self, device):
        """Determine which device to use"""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device

    def _load_model(self, size):
        """Load YOLO model"""
        model_name = f'yolov8{size}.pt'
        print(f"Loading {model_name}...")
        model = YOLO(model_name)
        # Ultralytics YOLO models handle device during inference
        return model

    def detect(self, frame, conf_threshold=0.5):
        """
        Detect objects in a single frame

        Args:
            frame: Image as numpy array (BGR format from OpenCV)
            conf_threshold: Confidence threshold (0-1)

        Returns:
            List of detections with bounding boxes and labels
        """
        # Pass device during inference
        results = self.model(frame, conf=conf_threshold, device=self.device, verbose=False)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = results.names[cls]

            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': conf,
                'class': label,
                'class_id': cls
            })

        return detections

    def detect_video(self, video_path, output_path=None, show=True):
        """
        Process entire video

        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            show: Display video while processing
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"❌ Error: Could not open video {video_path}")
            return

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")

        # Setup video writer if saving
        writer = None
        if output_path:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Saving output to: {output_path}")

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detect objects
            detections = self.detect(frame)

            # Draw bounding boxes
            annotated_frame = self._draw_detections(frame, detections)

            # Display FPS
            cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if writer:
                writer.write(annotated_frame)

            if show:
                # Disabled for headless environment compatibility
                try:
                    cv2.imshow('YOLO Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except cv2.error:
                    pass # Ignore imshow errors in headless

            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Processed {frame_count}/{total_frames} frames", end='\r')

        cap.release()
        if writer:
            writer.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass

        print(f"\n✅ Processed {frame_count} frames")

    def _draw_detections(self, frame, detections):
        """Draw bounding boxes on frame"""
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class']} {det['confidence']:.2f}"

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), (0, 255, 0), -1)

            # Draw label text
            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return annotated


# Test script
if __name__ == "__main__":
    # Initialize detector
    detector = YOLODetector(model_size='n')  # Start with nano for speed

    # Define paths
    input_video = "data/test_videos/city_traffic.mp4"
    output_video = "outputs/videos/city_traffic_detected.mp4"

    if os.path.exists(input_video):
        print(f"\nTesting on video: {input_video}")
        # Disable show=True to prevent cv2.imshow errors in headless
        detector.detect_video(input_video, output_path=output_video, show=False)
        print(f"Output saved to {output_video}")
    else:
        print(f"Video not found: {input_video}")
