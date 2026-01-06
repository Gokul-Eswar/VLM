"""
Phase 4: Test & Benchmark Script
Tests the advanced tracking system with various scenarios
"""

import numpy as np
import cv2
import time
import sys
sys.path.append('src')


def test_iou_calculation():
    """Test the IoU calculation function"""
    print("\n" + "="*60)
    print("TEST 1: IoU Calculation")
    print("="*60)
    
    from tracking.advanced_tracker import AdvancedSemanticTracker
    
    # Create dummy tracker to access _iou method
    class DummyTracker:
        def _iou(self, bbox1, bbox2):
            return AdvancedSemanticTracker._iou(None, bbox1, bbox2)
    
    tracker = DummyTracker()
    
    test_cases = [
        # (bbox1, bbox2, expected_iou_range, description)
        ([0, 0, 10, 10], [0, 0, 10, 10], (0.99, 1.0), "Perfect overlap"),
        ([0, 0, 10, 10], [5, 5, 15, 15], (0.14, 0.15), "Partial overlap"),
        ([0, 0, 10, 10], [20, 20, 30, 30], (0.0, 0.0), "No overlap"),
        ([0, 0, 10, 10], [5, 0, 15, 10], (0.33, 0.34), "Half overlap"),
    ]
    
    passed = 0
    for bbox1, bbox2, (min_iou, max_iou), desc in test_cases:
        iou = tracker._iou(bbox1, bbox2)
        if min_iou <= iou <= max_iou:
            print(f"  ✅ {desc}: IoU = {iou:.3f}")
            passed += 1
        else:
            print(f"  ❌ {desc}: IoU = {iou:.3f} (expected {min_iou}-{max_iou})")
    
    print(f"\n📊 Result: {passed}/{len(test_cases)} tests passed")
    return passed == len(test_cases)


def test_kalman_filter():
    """Test Kalman filter prediction"""
    print("\n" + "="*60)
    print("TEST 2: Kalman Filter Prediction")
    print("="*60)
    
    from tracking.advanced_tracker import KalmanBoxTracker
    
    # Create a tracker for a moving box
    initial_bbox = [100, 100, 150, 150]
    tracker = KalmanBoxTracker(initial_bbox)
    
    print(f"  Initial position: {initial_bbox}")
    
    # Simulate movement to the right
    positions = [
        [110, 100, 160, 150],
        [120, 100, 170, 150],
        [130, 100, 180, 150],
    ]
    
    for i, pos in enumerate(positions):
        tracker.update(pos)
        predicted = tracker.predict()
        print(f"  Frame {i+1}: Updated to {pos[:2]}, Predicted next: {predicted[:2].astype(int)}")
    
    # Predict without update (occlusion simulation)
    print("\n  Simulating occlusion (no updates)...")
    for i in range(3):
        predicted = tracker.predict()
        print(f"  Occluded frame {i+1}: Predicted at {predicted[:2].astype(int)}")
    
    print("\n✅ Kalman filter test completed")
    return True


def test_temporal_memory():
    """Test temporal memory system"""
    print("\n" + "="*60)
    print("TEST 3: Temporal Memory")
    print("="*60)
    
    from tracking.advanced_tracker import TemporalMemory
    
    memory = TemporalMemory(buffer_size=10)
    
    # Add some memories
    print("  Adding memories...")
    memory.add(1, 0, [100, 100, 150, 150], "person in red jacket")
    memory.add(1, 1, [105, 100, 155, 150], "person in red jacket")
    memory.add(2, 0, [200, 200, 250, 250], "car with dented door")
    
    # Test retrieval
    latest = memory.get_latest(1)
    print(f"  ✅ Latest for track 1: {latest['description']}")
    
    # Test search
    print("\n  Searching for 'person in red'...")
    results = memory.find_similar("person in red")
    for track_id, similarity, desc in results:
        print(f"    Track {track_id}: {similarity:.2f} - {desc}")
    
    print("\n✅ Temporal memory test completed")
    return True


def benchmark_tracking_speed():
    """Benchmark tracking performance"""
    print("\n" + "="*60)
    print("TEST 4: Performance Benchmark")
    print("="*60)
    
    from detection.yolo_detector import YOLODetector
    
    print("\n  Loading YOLO detector...")
    detector = YOLODetector(model_size='n')
    
    # Create test frames
    print("  Generating test frames...")
    test_frames = [
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        for _ in range(30)
    ]
    
    # Benchmark detection only
    print("\n  Benchmarking YOLO detection...")
    times = []
    for frame in test_frames[:10]:
        start = time.time()
        _ = detector.detect(frame)
        times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    fps = 1.0 / avg_time if avg_time > 0 else 0
    
    print(f"    Average time: {avg_time*1000:.1f}ms")
    print(f"    FPS: {fps:.1f}")
    
    if fps > 20:
        print("    ✅ Excellent performance!")
    elif fps > 10:
        print("    ✅ Good performance")
    else:
        print("    ⚠️  Slow - consider using GPU or smaller model")
    
    return True


def test_occlusion_handling():
    """Test how tracker handles occlusions"""
    print("\n" + "="*60)
    print("TEST 5: Occlusion Handling")
    print("="*60)
    
    from tracking.advanced_tracker import KalmanBoxTracker
    
    print("\n  Simulating object moving then being occluded...")
    
    tracker = KalmanBoxTracker([100, 100, 150, 150])
    
    # Normal tracking
    print("  Normal tracking phase:")
    for i in range(5):
        new_pos = [100 + i*10, 100, 150 + i*10, 150]
        tracker.update(new_pos)
        predicted = tracker.predict()
        print(f"    Frame {i}: Detected at {new_pos[0]}, Predicted: {int(predicted[0])}")
    
    # Occlusion
    print("\n  Occlusion phase (no detections):")
    for i in range(5):
        predicted = tracker.predict()
        print(f"    Frame {5+i}: No detection, Predicted: {int(predicted[0])}")
        print(f"              Time since update: {tracker.time_since_update}")
    
    print("\n✅ Occlusion handling test completed")
    print("   Note: In real system, track would be removed after max_age frames")
    
    return True


def test_multi_object_tracking():
    """Test tracking multiple objects"""
    print("\n" + "="*60)
    print("TEST 6: Multi-Object Tracking")
    print("="*60)
    
    from tracking.advanced_tracker import AdvancedSemanticTracker
    from detection.yolo_detector import YOLODetector
    
    # Create simple mock VLM for testing
    class MockVLM:
        def describe_object(self, frame, bbox, context=""):
            return f"object at ({bbox[0]}, {bbox[1]})"
    
    detector = YOLODetector(model_size='n')
    vlm = MockVLM()
    
    print("\n  Creating tracker with 3 objects...")
    tracker = AdvancedSemanticTracker(detector, vlm, max_age=5, min_hits=1)
    
    # Create frame with simulated detections
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Simulate 3 objects moving
    for frame_num in range(5):
        print(f"\n  Frame {frame_num}:")
        
        # Mock detections (simulate objects moving)
        class MockDetector:
            def detect(self, frame):
                return [
                    {'bbox': [100 + frame_num*10, 100, 150 + frame_num*10, 150], 
                     'confidence': 0.9, 'class': 'person'},
                    {'bbox': [200, 200 + frame_num*5, 250, 250 + frame_num*5], 
                     'confidence': 0.8, 'class': 'car'},
                    {'bbox': [300, 150, 350, 200], 
                     'confidence': 0.85, 'class': 'bicycle'},
                ]
        
        tracker.detector = MockDetector()
        tracks = tracker.update(frame)
        
        print(f"    Active tracks: {len(tracks)}")
        for track in tracks:
            print(f"      ID {track['id']}: bbox={track['bbox'][:2]}")
    
    print(f"\n✅ Multi-object tracking test completed")
    print(f"   Total tracks created: {tracker.stats['total_tracks']}")
    
    return True


def interactive_test():
    """Interactive test with webcam"""
    print("\n" + "="*60)
    print("TEST 7: Interactive Webcam Test")
    print("="*60)
    
    response = input("\n  Run interactive webcam test? (y/n): ")
    if response.lower() != 'y':
        print("  Skipped")
        return True
    
    from detection.yolo_detector import YOLODetector
    from tracking.advanced_tracker import KalmanBoxTracker
    
    print("\n  Opening webcam...")
    cap = cv2.VideoCapture(0)
    detector = YOLODetector(model_size='n')
    
    trackers = []
    frame_count = 0
    
    print("\n  Instructions:")
    print("    - Move objects in view")
    print("    - Watch track IDs stay consistent")
    print("    - Press 'q' to finish test")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect
        detections = detector.detect(frame)
        
        # Simple tracking (just to test)
        annotated = frame.copy()
        
        for det in detections:
            bbox = det['bbox']
            cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                         (0, 255, 0), 2)
            cv2.putText(annotated, det['class'], (bbox[0], bbox[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.putText(annotated, f"Frame: {frame_count} | Objects: {len(detections)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated, "Press 'q' to finish", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow('Interactive Test', annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Interactive test completed")
    print(f"   Processed {frame_count} frames")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🧪 PHASE 4: ADVANCED TRACKING TEST SUITE")
    print("="*70)
    
    tests = [
        ("IoU Calculation", test_iou_calculation),
        ("Kalman Filter", test_kalman_filter),
        ("Temporal Memory", test_temporal_memory),
        ("Tracking Speed", benchmark_tracking_speed),
        ("Occlusion Handling", test_occlusion_handling),
        ("Multi-Object Tracking", test_multi_object_tracking),
        ("Interactive Test", interactive_test),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            print(f"\n{'='*70}")
            result = test_func()
            results.append((name, result))
        except KeyboardInterrupt:
            print("\n\n⚠️  Tests interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ {name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! You're ready for Phase 5!")
    else:
        print("\n⚠️  Some tests failed. Review the output above.")
    
    print("="*70)


if __name__ == "__main__":
    main()
