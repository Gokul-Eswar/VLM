"""
Project Spectrum Setup Script
Run this to automatically set up your development environment
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a shell command and handle errors"""
    print(f"\n{'='*60}")
    print(f"▶️  {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True)
        print(result.stdout)
        print(f"✅ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"Error: {e.stderr}")
        return False

def create_directory_structure():
    """Create all necessary directories"""
    print("\n📁 Creating project structure...")
    
    directories = [
        'data/test_videos',
        'data/datasets',
        'models/yolo',
        'models/vlm',
        'src/detection',
        'src/tracking',
        'src/vlm',
        'src/deployment',
        'src/utils',
        'notebooks',
        'tests',
        'configs',
        'outputs/videos',
        'outputs/logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created: {directory}")
    
    print("✅ Directory structure created!")

def create_requirements_file():
    """Create requirements.txt"""
    requirements = """# Core ML Framework
torch>=2.1.0
torchvision>=0.16.0
torchaudio>=2.1.0

# Computer Vision
opencv-python>=4.8.0
opencv-contrib-python>=4.8.0
ultralytics>=8.0.0
supervision>=0.16.0

# Vision-Language Model
transformers>=4.36.0
accelerate>=0.25.0
sentencepiece>=0.1.99
pillow>=10.0.0

# Deployment
bentoml>=1.2.0
fastapi>=0.104.0
uvicorn>=0.24.0

# Utilities
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
tqdm>=4.66.0
pyyaml>=6.0
requests>=2.31.0

# Development
jupyter>=1.0.0
ipykernel>=6.26.0
pytest>=7.4.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    print("✅ requirements.txt created!")

def create_gitignore():
    """Create .gitignore file"""
    gitignore = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Models
models/*.pt
models/*.pth
models/*.onnx

# Data
data/test_videos/*
data/datasets/*
!data/test_videos/.gitkeep
!data/datasets/.gitkeep

# Outputs
outputs/videos/*
outputs/logs/*
!outputs/videos/.gitkeep
!outputs/logs/.gitkeep

# Jupyter
.ipynb_checkpoints
*.ipynb

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore)
    
    # Create .gitkeep files
    for path in ['data/test_videos', 'data/datasets', 'outputs/videos', 'outputs/logs']:
        Path(f"{path}/.gitkeep").touch()
    
    print("✅ .gitignore created!")

def create_config_file():
    """Create default configuration"""
    config = """# Project Spectrum Configuration

# Detection Settings
detection:
  model_size: "n"  # Options: n, s, m, l, x
  confidence_threshold: 0.5
  device: "auto"  # Options: auto, cuda, cpu

# Tracking Settings
tracking:
  max_age: 30  # Maximum frames to keep track without detection
  min_hits: 3  # Minimum detections before track is confirmed
  iou_threshold: 0.3

# VLM Settings
vlm:
  model_name: "Qwen/Qwen2-VL-7B-Instruct"
  max_tokens: 256
  temperature: 0.7

# Deployment Settings
deployment:
  api_port: 8000
  max_workers: 4
  timeout: 30

# Video Processing
video:
  input_fps: 30
  output_fps: 30
  resolution: [1280, 720]  # Width, Height
  codec: "mp4v"
"""
    
    with open('configs/default.yaml', 'w') as f:
        f.write(config)
    
    print("✅ Configuration file created!")

def create_readme():
    """Create project README"""
    readme = """# Project Spectrum: Vision-Language Tracking System

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
venv\\Scripts\\activate     # Windows
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
"""
    
    with open('README.md', 'w') as f:
        f.write(readme)
    
    print("✅ README.md created!")

def main():
    """Main setup function"""
    print("\n" + "="*60)
    print("🚀 PROJECT SPECTRUM SETUP")
    print("="*60)
    
    # Check Python version
    py_version = sys.version_info
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 9):
        print(f"❌ Python 3.9+ required. You have {py_version.major}.{py_version.minor}")
        sys.exit(1)
    
    print(f"✅ Python {py_version.major}.{py_version.minor}.{py_version.micro} detected")
    
    # Create structure
    create_directory_structure()
    create_requirements_file()
    create_gitignore()
    create_config_file()
    create_readme()
    
    print("\n" + "="*60)
    print("📦 NEXT STEPS:")
    print("="*60)
    print("""
1. Create and activate virtual environment:
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\\Scripts\\activate     # Windows

2. Install dependencies:
   pip install --upgrade pip
   pip install -r requirements.txt

3. Test GPU setup:
   python tests/test_gpu.py

4. Start building!
   Follow the BUILD_GUIDE.md for detailed instructions
""")
    
    print("="*60)
    print("✅ SETUP COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
