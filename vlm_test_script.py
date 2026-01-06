"""
Test script for Vision-Language Model
Run this BEFORE the full semantic tracker to verify VLM works correctly
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')


def test_vlm_basic():
    """Test 1: Basic VLM loading and inference"""
    print("\n" + "="*60)
    print("TEST 1: Basic VLM Loading")
    print("="*60)
    
    try:
        from vlm.semantic_descriptor import VisionLanguageDescriptor
        
        print("\n📦 Loading VLM (this may take 2-5 minutes first time)...")
        print("   The model needs to download (~2GB)")
        
        vlm = VisionLanguageDescriptor(
            model_name="Qwen/Qwen2-VL-2B-Instruct",  # Smaller model for testing
            device='auto'
        )
        
        print("✅ VLM loaded successfully!")
        return vlm
        
    except Exception as e:
        print(f"❌ Failed to load VLM: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Make sure you have internet connection")
        print("   2. Check if you have enough disk space (~5GB needed)")
        print("   3. Try running: pip install --upgrade transformers")
        return None


def test_vlm_webcam(vlm):
    """Test 2: VLM with webcam image"""
    print("\n" + "="*60)
    print("TEST 2: VLM with Webcam")
    print("="*60)
    
    print("\n📷 Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return False
    
    print("✅ Webcam opened")
    print("\nInstructions:")
    print("  - Position an object in front of camera")
    print("  - Press SPACE to capture and describe")
    print("  - Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw crosshair to help framing
        h, w = frame.shape[:2]
        cv2.line(frame, (w//2, 0), (w//2, h), (0, 255, 0), 1)
        cv2.line(frame, (0, h//2), (w, h//2), (0, 255, 0), 1)
        
        # Draw instructions
        cv2.putText(frame, "Position object in center", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press SPACE to describe", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('VLM Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space bar
            print("\n🔍 Analyzing image...")
            
            # Use center region as bounding box
            bbox = [w//4, h//4, 3*w//4, 3*h//4]
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                         (0, 255, 0), 2)
            cv2.imshow('VLM Test', frame)
            
            try:
                description = vlm.describe_object(frame, bbox, context="webcam capture")
                print(f"\n📝 Description: {description}")
                print("\n✅ VLM inference successful!")
                
            except Exception as e:
                print(f"\n❌ VLM inference failed: {e}")
        
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return True


def test_vlm_sample_image(vlm):
    """Test 3: VLM with a sample image (if no webcam)"""
    print("\n" + "="*60)
    print("TEST 3: VLM with Sample Image")
    print("="*60)
    
    # Create a sample image with shapes
    print("\n🎨 Creating sample image...")
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Draw some shapes
    cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), -1)  # Blue square
    cv2.circle(img, (300, 100), 50, (0, 255, 0), -1)  # Green circle
    cv2.rectangle(img, (450, 50), (590, 150), (0, 0, 255), -1)  # Red rectangle
    
    # Add text
    cv2.putText(img, "Test Shapes", (200, 300),
               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    cv2.imshow('Sample Image', img)
    print("   Showing sample image...")
    
    # Describe different regions
    regions = {
        'Blue Square': [50, 50, 150, 150],
        'Green Circle': [250, 50, 350, 150],
        'Red Rectangle': [450, 50, 590, 150]
    }
    
    for name, bbox in regions.items():
        print(f"\n🔍 Analyzing {name}...")
        
        # Draw bbox
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.imshow('Sample Image', img)
        cv2.waitKey(500)
        
        try:
            description = vlm.describe_object(img, bbox, context="test image")
            print(f"   📝 {name}: {description}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    print("\n✅ Sample image test completed!")
    print("   Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_vlm_performance(vlm):
    """Test 4: Check VLM speed"""
    print("\n" + "="*60)
    print("TEST 4: VLM Performance")
    print("="*60)
    
    import time
    
    # Create test image
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    bbox = [100, 100, 400, 400]
    
    print("\n⏱️  Running 3 inference tests...")
    times = []
    
    for i in range(3):
        print(f"   Test {i+1}/3...", end=' ')
        start = time.time()
        
        try:
            _ = vlm.describe_object(img, bbox)
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"✅ {elapsed:.2f}s")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            return False
    
    avg_time = sum(times) / len(times)
    print(f"\n📊 Average inference time: {avg_time:.2f}s")
    
    if avg_time < 2.0:
        print("✅ Excellent performance!")
    elif avg_time < 5.0:
        print("✅ Good performance")
    else:
        print("⚠️  Slow performance - consider using smaller model or GPU")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🧪 VISION-LANGUAGE MODEL TEST SUITE")
    print("="*60)
    print("\nThis script will test if your VLM is working correctly")
    print("before running the full semantic tracker.\n")
    
    # Test 1: Load VLM
    vlm = test_vlm_basic()
    if vlm is None:
        print("\n❌ Cannot proceed - VLM failed to load")
        return
    
    # Test 2: Webcam test
    print("\n" + "="*60)
    input("Press ENTER to start webcam test (or Ctrl+C to skip)...")
    test_vlm_webcam(vlm)
    
    # Test 3: Sample image
    print("\n" + "="*60)
    input("Press ENTER for sample image test (or Ctrl+C to skip)...")
    test_vlm_sample_image(vlm)
    
    # Test 4: Performance
    print("\n" + "="*60)
    input("Press ENTER for performance test (or Ctrl+C to skip)...")
    test_vlm_performance(vlm)
    
    # Summary
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETED")
    print("="*60)
    print("\nYour VLM is working correctly!")
    print("You can now run the full semantic tracker:")
    print("   python src/vlm/semantic_descriptor.py")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
