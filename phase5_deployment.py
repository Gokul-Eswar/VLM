"""
Phase 5: Production Deployment with BentoML
This module wraps your tracking system in a production-ready API that can:
- Handle REST API requests
- Process video streams
- Run on edge devices or cloud
- Auto-scale based on load
- Monitor performance
"""

import bentoml
from bentoml.io import JSON, NumpyNdarray, Image as BentoImage, Multipart
import numpy as np
import cv2
from typing import List, Dict, Optional
import logging
from pathlib import Path
import base64
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security Constants
MAX_BATCH_SIZE = 32
MAX_IMAGE_SIZE = 4096
MAX_QUERY_LENGTH = 200


# ============================================================================
# Model Service Definition
# ============================================================================

@bentoml.service(
    name="spectrum_tracker",
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-tesla-t4",
        "memory": "8Gi",
    },
    traffic={
        "timeout": 300,
        "concurrency": 4,
    }
)
class SpectrumTrackerService:
    """
    Production-ready tracking service
    
    Features:
    - REST API endpoints
    - Video stream processing
    - Batch processing
    - Performance monitoring
    """
    
    def __init__(self):
        """Initialize the tracking system"""
        logger.info("🚀 Initializing Spectrum Tracker Service...")
        
        # Import your tracking components
        import sys
        sys.path.append('src')
        
        from detection.yolo_detector import YOLODetector
        from vlm.semantic_descriptor import VisionLanguageDescriptor
        from tracking.advanced_tracker import AdvancedSemanticTracker
        
        # Initialize components
        logger.info("Loading YOLO detector...")
        self.detector = YOLODetector(model_size='n', device='auto')
        
        logger.info("Loading Vision-Language Model...")
        self.vlm = VisionLanguageDescriptor(
            model_name="Qwen/Qwen2-VL-2B-Instruct",
            device='auto'
        )
        
        logger.info("Creating tracker...")
        self.tracker = AdvancedSemanticTracker(
            self.detector,
            self.vlm,
            max_age=30,
            min_hits=3,
            iou_threshold=0.3,
            description_interval=15
        )
        
        # Performance metrics
        self.metrics = {
            'total_requests': 0,
            'total_frames': 0,
            'total_time': 0,
            'avg_fps': 0
        }
        
        logger.info("✅ Service initialized successfully!")
    
    @bentoml.api
    def track_image(self, image: NumpyNdarray) -> JSON:
        """
        Track objects in a single image
        
        Args:
            image: numpy array (H, W, 3) in RGB format
            
        Returns:
            JSON with tracks and metadata
        """
        start_time = time.time()
        
        try:
            # Security check: Image dimensions
            if image.shape[0] > MAX_IMAGE_SIZE or image.shape[1] > MAX_IMAGE_SIZE:
                return {
                    'success': False,
                    'error': f'Image dimensions exceed maximum allowed size of {MAX_IMAGE_SIZE}px'
                }

            # Convert RGB to BGR for OpenCV
            if len(image.shape) == 3 and image.shape[2] == 3:
                frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                frame = image
            
            # Process frame
            tracks = self.tracker.update(frame)
            
            # Update metrics
            elapsed = time.time() - start_time
            self.metrics['total_requests'] += 1
            self.metrics['total_frames'] += 1
            self.metrics['total_time'] += elapsed
            self.metrics['avg_fps'] = self.metrics['total_frames'] / self.metrics['total_time']
            
            return {
                'success': True,
                'tracks': tracks,
                'frame_info': {
                    'frame_number': self.tracker.frame_count,
                    'active_tracks': len(tracks),
                    'processing_time': elapsed,
                    'fps': 1.0 / elapsed if elapsed > 0 else 0
                },
                'metadata': {
                    'service': 'spectrum_tracker',
                    'version': '1.0.0'
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
            return {
                'success': False,
                'error': 'An internal error occurred while processing the image'
            }
    
    @bentoml.api
    def track_batch(self, images: List[NumpyNdarray]) -> JSON:
        """
        Track objects in multiple images (batch processing)
        
        Args:
            images: List of numpy arrays
            
        Returns:
            JSON with batch results
        """
        # Security check: Batch size
        if len(images) > MAX_BATCH_SIZE:
            return {
                'success': False,
                'error': f'Batch size exceeds maximum allowed size of {MAX_BATCH_SIZE}'
            }

        start_time = time.time()
        results = []
        
        for idx, image in enumerate(images):
            result = self.track_image(image)
            result['batch_index'] = idx
            results.append(result)
        
        elapsed = time.time() - start_time
        
        return {
            'success': True,
            'results': results,
            'batch_info': {
                'batch_size': len(images),
                'total_time': elapsed,
                'avg_time_per_image': elapsed / len(images) if images else 0
            }
        }
    
    @bentoml.api
    def search_tracks(self, query: str) -> JSON:
        """
        Search for tracks by description
        
        Args:
            query: Natural language description (e.g., "person in red jacket")
            
        Returns:
            JSON with matching tracks
        """
        try:
            # Security check: Query length
            if len(query) > MAX_QUERY_LENGTH:
                return {
                    'success': False,
                    'error': f'Query length exceeds maximum allowed length of {MAX_QUERY_LENGTH} characters'
                }

            results = self.tracker.search_by_description(query)
            
            matches = []
            for track_id, similarity, description in results:
                matches.append({
                    'track_id': track_id,
                    'similarity': similarity,
                    'description': description
                })
            
            return {
                'success': True,
                'query': query,
                'matches': matches,
                'total_matches': len(matches)
            }
            
        except Exception as e:
            logger.error(f"Error searching tracks: {e}", exc_info=True)
            return {
                'success': False,
                'error': 'An internal error occurred while searching tracks'
            }
    
    @bentoml.api
    def reset_tracker(self) -> JSON:
        """
        Reset the tracker state (clear all tracks)
        
        Returns:
            JSON confirmation
        """
        try:
            from tracking.advanced_tracker import KalmanBoxTracker
            
            self.tracker.trackers = []
            KalmanBoxTracker.count = 0
            self.tracker.frame_count = 0
            
            return {
                'success': True,
                'message': 'Tracker reset successfully'
            }
            
        except Exception as e:
            logger.error(f"Error resetting tracker: {e}", exc_info=True)
            return {
                'success': False,
                'error': 'An internal error occurred while resetting the tracker'
            }
    
    @bentoml.api
    def get_metrics(self) -> JSON:
        """
        Get service performance metrics
        
        Returns:
            JSON with metrics
        """
        return {
            'success': True,
            'metrics': self.metrics,
            'tracker_stats': self.tracker.stats
        }


# ============================================================================
# Video Processing Client
# ============================================================================

class VideoStreamClient:
    """
    Client for processing video streams with the deployed service
    """
    
    def __init__(self, service_url: str = "http://localhost:3000"):
        """
        Args:
            service_url: URL of the deployed BentoML service
        """
        self.service_url = service_url
        import requests
        self.session = requests.Session()
    
    def process_video(self, video_path: str, output_path: Optional[str] = None,
                     show_preview: bool = True):
        """
        Process video file through the API
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            show_preview: Show preview while processing
        """
        cap = cv2.VideoCapture(video_path)
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to RGB for API
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Send to API
            try:
                response = self.session.post(
                    f"{self.service_url}/track_image",
                    json={'image': rgb_frame.tolist()}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result['success']:
                        tracks = result['tracks']
                        
                        # Draw tracks on frame
                        annotated = self._draw_tracks(frame, tracks)
                        
                        if writer:
                            writer.write(annotated)
                        
                        if show_preview:
                            cv2.imshow('Video Processing', annotated)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                
            except Exception as e:
                logger.error(f"API error: {e}")
            
            frame_count += 1
            
            # Progress
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_actual = frame_count / elapsed
                eta = (total_frames - frame_count) / fps_actual if fps_actual > 0 else 0
                logger.info(f"Progress: {frame_count}/{total_frames} ({fps_actual:.1f} fps, ETA: {eta:.0f}s)")
        
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Completed: {frame_count} frames in {elapsed:.1f}s ({frame_count/elapsed:.1f} fps)")
    
    def _draw_tracks(self, frame, tracks):
        """Draw tracks on frame"""
        annotated = frame.copy()
        
        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            track_id = track['id']
            description = track['description']
            
            # Color
            np.random.seed(track_id)
            color = tuple(map(int, np.random.randint(50, 255, 3)))
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"ID:{track_id} | {description[:30]}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated


# ============================================================================
# Docker and Deployment Configurations
# ============================================================================

def generate_dockerfile():
    """Generate optimized Dockerfile"""
    dockerfile = """# Spectrum Tracker - Production Dockerfile

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    wget \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:3000/healthz || exit 1

# Run service
CMD ["bentoml", "serve", "service:SpectrumTrackerService", "--port", "3000"]
"""
    
    with open('Dockerfile', 'w') as f:
        f.write(dockerfile)
    
    logger.info("✅ Generated Dockerfile")


def generate_docker_compose():
    """Generate docker-compose for local deployment"""
    compose = """version: '3.8'

services:
  spectrum-tracker:
    build: .
    ports:
      - "3000:3000"
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./outputs:/app/outputs
      - ./data:/app/data
    restart: unless-stopped

  # Optional: Redis for caching
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped

volumes:
  redis-data:
"""
    
    with open('docker-compose.yml', 'w') as f:
        f.write(compose)
    
    logger.info("✅ Generated docker-compose.yml")


def generate_kubernetes_config():
    """Generate Kubernetes deployment config"""
    k8s_config = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: spectrum-tracker
  labels:
    app: spectrum-tracker
spec:
  replicas: 2
  selector:
    matchLabels:
      app: spectrum-tracker
  template:
    metadata:
      labels:
        app: spectrum-tracker
    spec:
      containers:
      - name: spectrum-tracker
        image: spectrum-tracker:latest
        ports:
        - containerPort: 3000
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        livenessProbe:
          httpGet:
            path: /healthz
            port: 3000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /readyz
            port: 3000
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: spectrum-tracker-service
spec:
  selector:
    app: spectrum-tracker
  ports:
  - protocol: TCP
    port: 80
    targetPort: 3000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: spectrum-tracker-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: spectrum-tracker
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
"""
    
    with open('k8s-deployment.yaml', 'w') as f:
        f.write(k8s_config)
    
    logger.info("✅ Generated k8s-deployment.yaml")


# ============================================================================
# CLI for Deployment
# ============================================================================

def main():
    """Main deployment script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Spectrum Tracker Deployment')
    parser.add_argument('command', choices=['serve', 'build', 'deploy', 'client'],
                       help='Command to execute')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', default=3000, type=int, help='Port to bind to')
    parser.add_argument('--video', help='Video file for client mode')
    parser.add_argument('--output', help='Output video path')
    parser.add_argument('--url', default='http://localhost:3000', 
                       help='Service URL for client')
    
    args = parser.parse_args()
    
    if args.command == 'serve':
        # Start local service
        logger.info(f"🚀 Starting service on {args.host}:{args.port}")
        import bentoml
        bentoml.serve(
            "service:SpectrumTrackerService",
            host=args.host,
            port=args.port
        )
    
    elif args.command == 'build':
        # Generate deployment files
        logger.info("📦 Generating deployment files...")
        generate_dockerfile()
        generate_docker_compose()
        generate_kubernetes_config()
        logger.info("✅ Deployment files generated!")
        logger.info("\nNext steps:")
        logger.info("  Docker: docker-compose up --build")
        logger.info("  K8s:    kubectl apply -f k8s-deployment.yaml")
    
    elif args.command == 'deploy':
        # Deploy to production
        logger.info("🚀 Deploying to production...")
        logger.info("This would deploy to your cloud provider")
        logger.info("Configure with environment variables or config file")
    
    elif args.command == 'client':
        # Run client
        if not args.video:
            logger.error("❌ --video required for client mode")
            return
        
        logger.info(f"📹 Processing video: {args.video}")
        client = VideoStreamClient(args.url)
        client.process_video(args.video, args.output)


if __name__ == "__main__":
    main()
