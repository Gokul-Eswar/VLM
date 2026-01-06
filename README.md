# Project Spectrum: Vision-Language Tracking System

A cutting-edge AI system for real-time object detection and tracking with semantic understanding.

## Features
- Real-time object detection using YOLO
- Semantic tracking with Vision-Language Models
- Multi-target tracking with occlusion handling
- Plug-and-play API deployment
- Edge and cloud deployment support

## Quick Start

1. Activate virtual environment:
```bash
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Run GPU test:
```bash
python tests/test_gpu.py
```

3. Test detection:
```bash
python src/detection/yolo_detector.py
```

## Project Structure
```
project-spectrum/
├── data/              # Test videos and datasets
├── models/            # Model weights
├── src/               # Source code
│   ├── detection/     # Object detection
│   ├── tracking/      # Tracking algorithms
│   ├── vlm/          # Vision-Language Model
│   └── deployment/    # API and deployment
├── tests/             # Test scripts
├── configs/           # Configuration files
└── outputs/           # Results and logs
```

## Documentation
See `BUILD_GUIDE.md` for detailed build instructions.

## License
MIT License
