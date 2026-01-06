"""
Phase 5: Edge Device Optimization
Optimized deployment for resource-constrained devices:
- NVIDIA Jetson (Orin, Xavier, Nano)
- Raspberry Pi 5 with AI HAT
- Intel Neural Compute Stick
- Mobile devices
"""

import torch
import numpy as np
import cv2
from pathlib import Path
import logging
import json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Model Quantization for Edge Devices
# ============================================================================

class ModelQuantizer:
    """
    Quantize models to reduce size and increase speed
    Supports: FP16, INT8, INT4
    """
    
    @staticmethod
    def quantize_yolo(model_path: str, quantization: str = 'fp16'):
        """
        Quantize YOLO model
        
        Args:
            model_path: Path to YOLO model
            quantization: 'fp16', 'int8', or 'dynamic'
            
        Returns:
            Path to quantized model
        """
        from ultralytics import YOLO
        
        logger.info(f"📦 Quantizing YOLO model to {quantization}...")
        
        model = YOLO(model_path)
        
        if quantization == 'fp16':
            # Export to FP16 (2x smaller, ~1.5x faster on GPU)
            output_path = model_path.replace('.pt', '_fp16.pt')
            model.export(format='engine', half=True)
            logger.info(f"✅ FP16 model saved")
            
        elif quantization == 'int8':
            # Export to INT8 (4x smaller, ~2x faster)
            output_path = model_path.replace('.pt', '_int8.pt')
            model.export(format='engine', int8=True)
            logger.info(f"✅ INT8 model saved")
            
        elif quantization == 'dynamic':
            # Dynamic quantization (PyTorch)
            output_path = model_path.replace('.pt', '_dynamic.pt')
            torch.quantization.quantize_dynamic(
                model.model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            torch.save(model.model.state_dict(), output_path)
            logger.info(f"✅ Dynamic quantized model saved")
        
        return output_path
    
    @staticmethod
    def quantize_vlm(model_name: str, quantization: str = '4bit'):
        """
        Quantize VLM for edge deployment
        
        Args:
            model_name: Hugging Face model name
            quantization: '4bit', '8bit', or 'fp16'
            
        Returns:
            Loaded quantized model
        """
        from transformers import Qwen2VLForConditionalGeneration, BitsAndBytesConfig
        
        logger.info(f"📦 Loading VLM with {quantization} quantization...")
        
        if quantization == '4bit':
            # 4-bit quantization (4x smaller, 3x faster)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            
        elif quantization == '8bit':
            # 8-bit quantization (2x smaller, 2x faster)
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            
        else:  # fp16
            quantization_config = None
        
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map='auto',
            torch_dtype=torch.float16 if quantization == 'fp16' else None
        )
        
        logger.info(f"✅ VLM loaded with {quantization} quantization")
        return model


# ============================================================================
# Edge-Optimized Tracker
# ============================================================================

class EdgeOptimizedTracker:
    """
    Lightweight tracker optimized for edge devices
    
    Features:
    - Minimal memory footprint
    - Configurable compute budget
    - Frame skipping for low-power modes
    - Adaptive quality based on resources
    """
    
    def __init__(self, config_path: str = 'configs/edge_config.json'):
        """
        Args:
            config_path: Path to edge configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components based on device profile
        self._setup_models()
        
        # Performance tracking
        self.frame_times = []
        self.power_mode = 'balanced'  # 'performance', 'balanced', 'power_save'
    
    def _load_config(self, config_path):
        """Load edge device configuration"""
        default_config = {
            'device_profile': 'jetson_orin',  # Options: jetson_orin, jetson_nano, rpi5, generic
            'target_fps': 15,
            'max_memory_mb': 4096,
            'yolo_size': 'n',
            'vlm_enabled': True,
            'vlm_interval': 30,
            'quantization': {
                'yolo': 'fp16',
                'vlm': '4bit'
            },
            'tracking': {
                'max_age': 30,
                'min_hits': 3,
                'iou_threshold': 0.3
            }
        }
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
        except FileNotFoundError:
            logger.warning(f"Config not found, using defaults")
            return default_config
    
    def _setup_models(self):
        """Initialize models based on device profile"""
        logger.info(f"🔧 Setting up for device: {self.config['device_profile']}")
        
        import sys
        sys.path.append('src')
        
        from detection.yolo_detector import YOLODetector
        
        # Setup YOLO
        self.detector = YOLODetector(
            model_size=self.config['yolo_size'],
            device='auto'
        )
        
        # Setup VLM (if enabled and resources available)
        self.vlm = None
        if self.config['vlm_enabled']:
            try:
                from vlm.semantic_descriptor import VisionLanguageDescriptor
                
                # Use quantized model for edge
                self.vlm = VisionLanguageDescriptor(
                    model_name="Qwen/Qwen2-VL-2B-Instruct",
                    device='auto'
                )
                logger.info("✅ VLM enabled")
            except Exception as e:
                logger.warning(f"⚠️ VLM disabled: {e}")
                self.config['vlm_enabled'] = False
        
        # Setup tracker
        from tracking.advanced_tracker import AdvancedSemanticTracker
        
        self.tracker = AdvancedSemanticTracker(
            self.detector,
            self.vlm if self.vlm else self._create_dummy_vlm(),
            max_age=self.config['tracking']['max_age'],
            min_hits=self.config['tracking']['min_hits'],
            iou_threshold=self.config['tracking']['iou_threshold'],
            description_interval=self.config['vlm_interval']
        )
        
        logger.info("✅ Edge tracker initialized")
    
    def _create_dummy_vlm(self):
        """Create dummy VLM when disabled"""
        class DummyVLM:
            def describe_object(self, frame, bbox, context=""):
                return "object"
        return DummyVLM()
    
    def process_frame(self, frame):
        """
        Process a single frame with adaptive quality
        
        Args:
            frame: Video frame (numpy array)
            
        Returns:
            Tracks and metadata
        """
        start_time = time.time()
        
        # Adaptive frame skipping based on power mode
        if self.power_mode == 'power_save':
            if self.tracker.frame_count % 3 != 0:  # Process every 3rd frame
                return self._predict_only()
        
        # Process frame
        tracks = self.tracker.update(frame)
        
        # Track performance
        elapsed = time.time() - start_time
        self.frame_times.append(elapsed)
        
        # Keep only last 30 frame times
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        
        # Adaptive power mode
        self._adjust_power_mode()
        
        return {
            'tracks': tracks,
            'frame_number': self.tracker.frame_count,
            'processing_time': elapsed,
            'power_mode': self.power_mode
        }
    
    def _predict_only(self):
        """Return predictions without detection (frame skip mode)"""
        predictions = []
        for tracker in self.tracker.trackers:
            tracker.predict()
            bbox = tracker.get_state()
            predictions.append({
                'id': tracker.id,
                'bbox': bbox.astype(int).tolist(),
                'predicted': True
            })
        return {'tracks': predictions, 'predicted_frame': True}
    
    def _adjust_power_mode(self):
        """Adjust power mode based on performance"""
        if not self.frame_times:
            return
        
        avg_time = sum(self.frame_times) / len(self.frame_times)
        target_time = 1.0 / self.config['target_fps']
        
        if avg_time > target_time * 1.5:
            # Too slow, switch to power save
            if self.power_mode != 'power_save':
                self.power_mode = 'power_save'
                logger.info("⚡ Switched to power_save mode")
        elif avg_time < target_time * 0.7:
            # Fast enough, can use performance mode
            if self.power_mode != 'performance':
                self.power_mode = 'performance'
                logger.info("🚀 Switched to performance mode")
        else:
            # Balanced
            if self.power_mode != 'balanced':
                self.power_mode = 'balanced'
                logger.info("⚖️ Switched to balanced mode")


# ============================================================================
# Jetson-Specific Optimizations
# ============================================================================

class JetsonOptimizer:
    """
    NVIDIA Jetson-specific optimizations
    Uses TensorRT and CUDA optimizations
    """
    
    @staticmethod
    def optimize_for_jetson(yolo_model_path: str):
        """
        Convert YOLO to TensorRT for maximum Jetson performance
        
        Args:
            yolo_model_path: Path to YOLO .pt file
            
        Returns:
            Path to TensorRT engine
        """
        from ultralytics import YOLO
        
        logger.info("🔧 Optimizing for Jetson with TensorRT...")
        
        model = YOLO(yolo_model_path)
        
        # Export to TensorRT
        engine_path = model_path.replace('.pt', '_jetson.engine')
        model.export(
            format='engine',
            half=True,  # FP16 for Jetson
            workspace=4,  # 4GB workspace
            device=0
        )
        
        logger.info(f"✅ TensorRT engine created: {engine_path}")
        return engine_path
    
    @staticmethod
    def set_power_mode(mode: str):
        """
        Set Jetson power mode
        
        Args:
            mode: 'max' (MAXN), 'balanced' (15W), 'low' (10W)
        """
        import subprocess
        
        mode_map = {
            'max': '0',
            'balanced': '1',
            'low': '2'
        }
        
        if mode in mode_map:
            try:
                subprocess.run([
                    'sudo', 'nvpmodel', '-m', mode_map[mode]
                ], check=True)
                logger.info(f"⚡ Jetson power mode set to: {mode}")
            except Exception as e:
                logger.warning(f"Could not set power mode: {e}")
    
    @staticmethod
    def enable_max_performance():
        """Enable maximum performance on Jetson"""
        import subprocess
        
        try:
            # Set CPU and GPU to max frequency
            subprocess.run(['sudo', 'jetson_clocks'], check=True)
            logger.info("🚀 Jetson clocks enabled (max performance)")
        except Exception as e:
            logger.warning(f"Could not enable jetson_clocks: {e}")


# ============================================================================
# Edge Deployment Script
# ============================================================================

def deploy_to_edge():
    """Main script for edge deployment"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy to Edge Device')
    parser.add_argument('--device', default='auto',
                       choices=['jetson_orin', 'jetson_nano', 'rpi5', 'auto'],
                       help='Target device')
    parser.add_argument('--optimize', action='store_true',
                       help='Run optimization (quantization, TensorRT)')
    parser.add_argument('--test', action='store_true',
                       help='Run performance test')
    parser.add_argument('--demo', action='store_true',
                       help='Run live demo')
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device == 'auto':
        device = detect_device()
        logger.info(f"🔍 Detected device: {device}")
    else:
        device = args.device
    
    # Optimization
    if args.optimize:
        logger.info("🔧 Running optimization...")
        
        if 'jetson' in device:
            # Jetson-specific optimization
            JetsonOptimizer.optimize_for_jetson('models/yolo/yolov8n.pt')
            JetsonOptimizer.enable_max_performance()
        
        # Quantize models
        quantizer = ModelQuantizer()
        quantizer.quantize_yolo('models/yolo/yolov8n.pt', 'fp16')
        
        logger.info("✅ Optimization complete")
    
    # Performance test
    if args.test:
        logger.info("🧪 Running performance test...")
        test_edge_performance(device)
    
    # Demo
    if args.demo:
        logger.info("🎬 Starting edge demo...")
        run_edge_demo(device)


def detect_device():
    """Auto-detect edge device type"""
    import subprocess
    import platform
    
    try:
        # Check for Jetson
        if Path('/etc/nv_tegra_release').exists():
            with open('/etc/nv_tegra_release', 'r') as f:
                content = f.read()
                if 'Orin' in content:
                    return 'jetson_orin'
                elif 'Xavier' in content:
                    return 'jetson_xavier'
                else:
                    return 'jetson_nano'
    except:
        pass
    
    # Check for Raspberry Pi
    if 'arm' in platform.machine().lower():
        return 'rpi5'
    
    return 'generic'


def test_edge_performance(device: str):
    """Test performance on edge device"""
    logger.info("📊 Performance Test")
    logger.info("=" * 60)
    
    # Create tracker
    config = {
        'device_profile': device,
        'yolo_size': 'n',
        'vlm_enabled': False,  # Disable for pure speed test
        'target_fps': 30
    }
    
    # Save temp config
    with open('configs/test_config.json', 'w') as f:
        json.dump(config, f)
    
    tracker = EdgeOptimizedTracker('configs/test_config.json')
    
    # Create test frames
    test_frames = [
        np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        for _ in range(100)
    ]
    
    logger.info("Running 100 frame test...")
    times = []
    
    for frame in test_frames:
        start = time.time()
        _ = tracker.process_frame(frame)
        times.append(time.time() - start)
    
    # Results
    avg_time = sum(times) / len(times)
    fps = 1.0 / avg_time
    
    logger.info(f"\n📊 Results:")
    logger.info(f"  Average time: {avg_time*1000:.1f}ms")
    logger.info(f"  FPS: {fps:.1f}")
    logger.info(f"  Min time: {min(times)*1000:.1f}ms")
    logger.info(f"  Max time: {max(times)*1000:.1f}ms")
    
    if fps >= 20:
        logger.info("  ✅ Excellent performance!")
    elif fps >= 10:
        logger.info("  ✅ Good performance")
    else:
        logger.info("  ⚠️ Consider optimization")


def run_edge_demo(device: str):
    """Run live demo on edge device"""
    config_path = f'configs/{device}_config.json'
    
    if not Path(config_path).exists():
        logger.warning(f"Config not found: {config_path}, using default")
        config_path = 'configs/edge_config.json'
    
    tracker = EdgeOptimizedTracker(config_path)
    
    logger.info("🎬 Starting edge demo...")
    logger.info("Press 'q' to quit, 'p' to toggle power mode")
    
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process
        result = tracker.process_frame(frame)
        
        # Visualize
        annotated = tracker.tracker.visualize(frame, result['tracks'])
        
        # Add info
        cv2.putText(annotated, f"Power: {result['power_mode']}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow('Edge Demo', annotated)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            # Cycle power mode
            modes = ['performance', 'balanced', 'power_save']
            idx = modes.index(tracker.power_mode)
            tracker.power_mode = modes[(idx + 1) % len(modes)]
            logger.info(f"⚡ Power mode: {tracker.power_mode}")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    deploy_to_edge()
