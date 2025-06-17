#!/usr/bin/env python3
"""
AIDetect.py - 100% OFFLINE Multi-Engine Inference with Coordinates Output using Ultralytics
Usage: python AIDetect.py --image_path <path> --confidence <threshold>
Output: ClassName (x1,y1,x2,y2) hoặc "No detect"
"""

import os
import sys
import cv2
import numpy as np
import argparse
import concurrent.futures
from ultralytics import YOLO

class OfflineAIDetector:
    def __init__(self):
        """100% Offline AI Detector using Ultralytics - NO INTERNET NEEDED"""
        
        # All operations are LOCAL
        self.engine_paths = [
            "/home/jkl0909/TestCycletimeMeiko/Modelx/thuadong.engine",
            "/home/jkl0909/TestCycletimeMeiko/Modelx/khuyetmach.engine", 
            "/home/jkl0909/TestCycletimeMeiko/Modelx/nganmach.engine",
            "/home/jkl0909/TestCycletimeMeiko/Modelx/vetlom.engine",
            "/home/jkl0909/TestCycletimeMeiko/Modelx/xuoc/xuoc.engine"
        ]
        
        self.engine_names = ["ThuaDong", "KhuyetMach", "NganMach", "VetLom", "Xuoc"]
        
        # Use Ultralytics models instead of native TensorRT
        self.models = {}
        
        # LOCAL class mapping - NO INTERNET
        self.class_names = {
            0: "ThuaDong",
            1: "KhuyetMach", 
            2: "NganMach",
            3: "VetLom",
            4: "Xuoc"
        }
        
        # Load models using Ultralytics (handles TensorRT compatibility)
        self.load_models()
    
    def load_models(self):
        """Load models using Ultralytics YOLO wrapper - NO INTERNET"""
        
        for engine_path, engine_name in zip(self.engine_paths, self.engine_names):
            if os.path.exists(engine_path):
                try:
                    # ✅ Ultralytics handles TensorRT version compatibility automatically
                    model = YOLO(engine_path, task='obb')  # OBB for oriented bounding boxes
                    self.models[engine_name] = model
                except Exception as e:
                    # Try without task specification
                    try:
                        model = YOLO(engine_path)
                        self.models[engine_name] = model
                    except:
                        continue
        
        if not self.models:
            print("❌ No models loaded")
            sys.exit(1)
    
    def infer_single_model(self, model_name, image_path, conf_threshold):
        """Run inference on single model using Ultralytics - NO INTERNET"""
        
        try:
            model = self.models[model_name]
            
            # ✅ Ultralytics handles all preprocessing and inference automatically
            results = model(image_path, conf=conf_threshold, verbose=False)
            
            detections = []
            
            # Handle OBB (Oriented Bounding Box) results
            if hasattr(results[0], 'obb') and results[0].obb is not None:
                # Get corner coordinates directly from Ultralytics
                if len(results[0].obb.xyxy) > 0:
                    boxes = results[0].obb.xyxy  # Already in (x1, y1, x2, y2) format
                    confidences = results[0].obb.conf
                    classes = results[0].obb.cls
                    
                    for box, conf, cls in zip(boxes, confidences, classes):
                        if conf >= conf_threshold:
                            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                            detections.append({
                                'confidence': float(conf),
                                'class': int(cls),
                                'class_name': self.class_names.get(int(cls), f"Class_{int(cls)}"),
                                'coordinates': (x1, y1, x2, y2),
                                'source_model': model_name
                            })
            
            # Handle regular bounding box results
            elif hasattr(results[0], 'boxes') and results[0].boxes is not None:
                if len(results[0].boxes.xyxy) > 0:
                    boxes = results[0].boxes.xyxy  # Already in (x1, y1, x2, y2) format
                    confidences = results[0].boxes.conf
                    classes = results[0].boxes.cls
                    
                    for box, conf, cls in zip(boxes, confidences, classes):
                        if conf >= conf_threshold:
                            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                            detections.append({
                                'confidence': float(conf),
                                'class': int(cls),
                                'class_name': self.class_names.get(int(cls), f"Class_{int(cls)}"),
                                'coordinates': (x1, y1, x2, y2),
                                'source_model': model_name
                            })
            
            return detections
            
        except Exception as e:
            return []
    
    def detect_offline(self, image_path, confidence_threshold):
        """Main detection function using Ultralytics - 100% OFFLINE"""
        
        if not os.path.exists(image_path):
            return None
        
        try:
            all_detections = []
            
            # Parallel inference using Ultralytics models - NO INTERNET
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.models)) as executor:
                futures = [
                    executor.submit(self.infer_single_model, model_name, image_path, confidence_threshold)
                    for model_name in self.models.keys()
                ]
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        detections = future.result()
                        all_detections.extend(detections)
                    except:
                        continue
            
            if not all_detections:
                return None
            
            # Find best detection - LOCAL processing - NO INTERNET
            best = max(all_detections, key=lambda x: x['confidence'])
            return best
            
        except Exception as e:
            return None

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="100% OFFLINE AI Detector with Coordinates Output using Ultralytics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python AIDetect.py --image_path test.jpg --confidence 0.5
  Output: ThuaDong (120,45,380,290)
  
  python AIDetect.py --image_path test.jpg --confidence 0.9
  Output: No detect
        """
    )
    
    parser.add_argument(
        '--image_path', 
        type=str, 
        required=True,
        help='Path to input image file'
    )
    
    parser.add_argument(
        '--confidence', 
        type=float,
        required=True,
        help='Minimum confidence threshold (0.0 to 1.0)'
    )
    
    return parser.parse_args()

def main():
    """MAIN - 100% OFFLINE EXECUTION using Ultralytics"""
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Validate offline
        if not (0.0 <= args.confidence <= 1.0):
            print("No detect")
            sys.exit(0)
            
        if not os.path.exists(args.image_path):
            print("No detect")
            sys.exit(0)
        
        # Initialize OFFLINE detector using Ultralytics
        detector = OfflineAIDetector()
        
        # Run OFFLINE detection
        result = detector.detect_offline(args.image_path, args.confidence)
        
        # Output result with coordinates
        if result and result['confidence'] >= args.confidence:
            x1, y1, x2, y2 = result['coordinates']
            print(f"{result['class_name']} ({x1},{y1},{x2},{y2})")
        else:
            print("No detect")
        
    except KeyboardInterrupt:
        print("No detect")
        sys.exit(0)
    except Exception as e:
        print("No detect")
        sys.exit(0)

if __name__ == "__main__":
    main()