# Tech Stack: Project Spectrum

## Core Language & Frameworks
- **Python 3.9+**: The primary programming language for the entire system.
- **PyTorch**: The foundational deep learning framework for model execution and optimization.

## Computer Vision & Tracking
- **Ultralytics YOLOv8**: Utilized for real-time object detection across various model sizes (n, s, m, l, x).
- **OpenCV**: Used for video capture, frame manipulation, and image processing.
- **Supervision**: Employs utilities for processing detections and drawing tracking results.

## Vision-Language Integration
- **Hugging Face Transformers**: For loading and running state-of-the-art VLMs like Qwen2-VL.
- **Accelerate**: For efficient model deployment across diverse hardware (CPU, GPU, Multi-GPU).

## Database & Storage
- **ChromaDB**: Open-source vector database for storing tracking results and semantic embeddings, enabling efficient similarity search for historical data.

## Deployment & Serving
- **BentoML**: The primary framework for model serving and packaging production-ready "Bentos".
- **FastAPI & Uvicorn**: Provides a high-performance REST API for system integration.
- **Docker & Kubernetes**: For containerized deployment and orchestration in cloud environments.

## Optimization & Edge
- **TensorRT**: For high-performance inference on NVIDIA Jetson and other NVIDIA GPUs.
- **Bitsandbytes**: For 4-bit and 8-bit quantization to reduce model footprint and increase speed.

## Data & Utilities
- **NumPy & Pandas**: For efficient numerical processing and data management.
- **Matplotlib**: Used for visualization and performance benchmarking.
- **PyYAML**: For managing configuration files.
