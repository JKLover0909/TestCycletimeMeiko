#!/usr/bin/env python3
"""
AIDetect.py - 100% OFFLINE Multi-Engine Inference with OBB Coordinates + Theta Output
Usage: python AIDetect.py --image_path <path> --confidence <threshold>
Output: ClassName (x1,y1,x2,y2,theta) hoặc "No detect"
"""

import os
import sys
import cv2
import numpy as np
import argparse
import concurrent.futures
import subprocess
from ultralytics import YOLO

class OfflineAIDetector:
    def __init__(self):
        """100% Offline AI Detector with OBB support - NO INTERNET NEEDED"""
        
        self.engine_paths = [
            "/home/jkl0909/TestCycletimeMeiko/models/thuadong.engine",
            "/home/jkl0909/TestCycletimeMeiko/models//khuyetmach.engine", 
            "/home/jkl0909/TestCycletimeMeiko/models//nganmach.engine",
            "/home/jkl0909/TestCycletimeMeiko/models//vetlom.engine",
            "/home/jkl0909/TestCycletimeMeiko/models//xuoc.engine"
        ]
        
        self.engine_names = ["Thuadong", "KhuyetMach", "NganMach", "VetLom", "Xuoc"]
        self.models = {}
        
        # LOCAL class mapping
        self.class_names = {
            0: "ThuaDong",
            1: "KhuyetMach", 
            2: "NganMach",
            3: "VetLom",
            4: "Xuoc"
        }
        
        # Load models
        self.load_models()
    
    def load_models(self):
        """Load models with OBB support"""
        
        for engine_path, engine_name in zip(self.engine_paths, self.engine_names):
            if os.path.exists(engine_path):
                try:
                    # Try OBB task first
                    model = YOLO(engine_path, task='obb')
                    self.models[engine_name] = {'model': model, 'type': 'obb'}
                    print(f"✅ Loaded {engine_name} as OBB model")
                except Exception as e:
                    # Fallback to regular detection
                    try:
                        model = YOLO(engine_path)
                        self.models[engine_name] = {'model': model, 'type': 'detect'}
                        print(f"✅ Loaded {engine_name} as detection model")
                    except Exception as e2:
                        print(f"❌ Failed to load {engine_name}: {e2}")
                        continue
        
        if not self.models:
            print("❌ No models loaded")
            sys.exit(1)
    
    def calculate_obb_theta(self, obb_points):
        """Calculate rotation angle from OBB 4 corner points"""
        
        try:
            # obb_points shape: [4, 2] - 4 corner points
            # Get the first edge vector (from point 0 to point 1)
            edge_vector = obb_points[1] - obb_points[0]
            
            # Calculate angle in radians
            theta_rad = np.arctan2(edge_vector[1], edge_vector[0])
            
            # Convert to degrees
            theta_deg = np.degrees(theta_rad)
            
            # Normalize to [-90, 90] range for standard OBB representation
            if theta_deg > 90:
                theta_deg -= 180
            elif theta_deg < -90:
                theta_deg += 180
            
            return theta_deg
            
        except Exception as e:
            print(f"⚠️ Error calculating theta: {e}")
            return 0.0
    
    def infer_single_model(self, model_name, image_path, conf_threshold):
        """Run inference with OBB and theta support"""
        
        try:
            model_info = self.models[model_name]
            model = model_info['model']
            model_type = model_info['type']
            
            results = model(image_path, conf=conf_threshold, verbose=False)
            detections = []
            
            # Handle OBB results (with theta)
            if model_type == 'obb' and hasattr(results[0], 'obb') and results[0].obb is not None:
                obb_data = results[0].obb
                
                if len(obb_data.xyxyxyxy) > 0:
                    # Get OBB corner points
                    obb_boxes = obb_data.xyxyxyxy.cpu().numpy()  # Shape: [N, 4, 2]
                    confidences = obb_data.conf.cpu().numpy()
                    classes = obb_data.cls.cpu().numpy()
                    
                    print(f"🔍 {model_name} OBB: Found {len(obb_boxes)} detections")
                    
                    for i, (obb_points, conf, cls) in enumerate(zip(obb_boxes, confidences, classes)):
                        if conf >= conf_threshold:
                            # Calculate bounding rectangle from 4 corner points
                            x_coords = obb_points[:, 0]
                            y_coords = obb_points[:, 1]
                            
                            x1, y1 = int(x_coords.min()), int(y_coords.min())
                            x2, y2 = int(x_coords.max()), int(y_coords.max())
                            
                            # Calculate rotation angle
                            theta = self.calculate_obb_theta(obb_points)
                            
                            class_name = self.class_names.get(int(cls), f"Class_{int(cls)}")
                            
                            detections.append({
                                'confidence': float(conf),
                                'class': int(cls),
                                'class_name': class_name,
                                'coordinates': (x1, y1, x2, y2),
                                'theta': theta,
                                'source_model': model_name,
                                'has_theta': True
                            })
                            
                            print(f"   📍 {class_name}: bbox=({x1},{y1},{x2},{y2}), θ={theta:.1f}°, conf={conf:.3f}")
            
            # Handle regular detection results (no theta)
            elif hasattr(results[0], 'boxes') and results[0].boxes is not None:
                boxes_data = results[0].boxes
                
                if len(boxes_data.xyxy) > 0:
                    boxes = boxes_data.xyxy.cpu().numpy()
                    confidences = boxes_data.conf.cpu().numpy()
                    classes = boxes_data.cls.cpu().numpy()
                    
                    print(f"🔍 {model_name} Detection: Found {len(boxes)} detections")
                    
                    for box, conf, cls in zip(boxes, confidences, classes):
                        if conf >= conf_threshold:
                            x1, y1, x2, y2 = box.astype(int)
                            class_name = self.class_names.get(int(cls), f"Class_{int(cls)}")
                            
                            detections.append({
                                'confidence': float(conf),
                                'class': int(cls),
                                'class_name': class_name,
                                'coordinates': (x1, y1, x2, y2),
                                'theta': 0.0,  # No rotation for regular detection
                                'source_model': model_name,
                                'has_theta': False
                            })
                            
                            print(f"   📍 {class_name}: bbox=({x1},{y1},{x2},{y2}), conf={conf:.3f}")
            
            return detections
            
        except Exception as e:
            print(f"❌ Error in {model_name}: {e}")
            return []
    
    def detect_offline(self, image_path, confidence_threshold, auto_segment=False):
        """Main detection function with optional auto-segmentation"""
        
        if not os.path.exists(image_path):
            return None
        
        try:
            all_detections = []
            
            # Parallel inference
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.models)) as executor:
                futures = [
                    executor.submit(self.infer_single_model, model_name, image_path, confidence_threshold)
                    for model_name in self.models.keys()
                ]
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        detections = future.result()
                        all_detections.extend(detections)
                    except Exception as e:
                        print(f"⚠️ Future error: {e}")
                        continue
            
            if not all_detections:
                return None
            
            # Find best detection
            best = max(all_detections, key=lambda x: x['confidence'])
            
            # Auto-call Calculator.py if requested
            if auto_segment and best:
                self.call_calculator(image_path, best)
            
            return best
            
        except Exception as e:
            print(f"❌ Detection error: {e}")
            return None
    
    def call_calculator(self, image_path, detection):
        """Automatically call Calculator.py with detection results"""
        
        try:
            x1, y1, x2, y2 = detection['coordinates']
            theta = detection['theta']
            class_name = detection['class_name']
            
            print(f"\n🧮 Auto-calling Calculator.py for {class_name}...")
            
            # Prepare command
            cmd = [
                "python", "Calculator.py",
                "--image_path", image_path,
                "--class_name", class_name,
                "--coordinates", f"{x1},{y1},{x2},{y2}",
                "--theta", f"{theta:.2f}"
            ]
            
            # Execute Calculator.py
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Calculator.py completed successfully")
                if result.stdout:
                    print(f"📄 Calculator output: {result.stdout.strip()}")
            else:
                print(f"⚠️ Calculator.py failed: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Error calling Calculator.py: {e}")

def parse_arguments():
    """Parse command line arguments"""
    
    parser = argparse.ArgumentParser(
        description="100% OFFLINE AI Detector with OBB + Theta Output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python AIDetect.py --image_path test.jpg --confidence 0.5
  Output: ThuaDong (120,45,380,290,15.5)
  
  python AIDetect.py --image_path test.jpg --confidence 0.5 --auto_segment
  Output: ThuaDong (120,45,380,290,15.5) + auto-runs Calculator.py
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
    
    parser.add_argument(
        '--auto_segment',
        action='store_true',
        help='Automatically run Calculator.py for segmentation'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output'
    )
    
    return parser.parse_args()

def main():
    """MAIN - 100% OFFLINE EXECUTION with OBB + Theta"""
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Validate inputs
        if not (0.0 <= args.confidence <= 1.0):
            print("No detect")
            sys.exit(0)
            
        if not os.path.exists(args.image_path):
            print("No detect")
            sys.exit(0)
        
        # Initialize detector
        detector = OfflineAIDetector()
        
        # Run detection
        result = detector.detect_offline(
            args.image_path, 
            args.confidence, 
            auto_segment=args.auto_segment
        )
        
        # Output result with coordinates and theta
        if result and result['confidence'] >= args.confidence:
            x1, y1, x2, y2 = result['coordinates']
            theta = result['theta']
            class_name = result['class_name']
            
            if args.debug:
                print(f"🎯 Detection Details:")
                print(f"   Class: {class_name}")
                print(f"   Confidence: {result['confidence']:.3f}")
                print(f"   Coordinates: ({x1}, {y1}, {x2}, {y2})")
                print(f"   Theta: {theta:.2f}°")
                print(f"   Source: {result['source_model']}")
                print(f"   Has Theta: {result['has_theta']}")
            
            # Output format: ClassName (x1,y1,x2,y2,theta)
            if result['has_theta']:
                print(f"{class_name} ({x1},{y1},{x2},{y2},{theta:.2f})")
            else:
                print(f"{class_name} ({x1},{y1},{x2},{y2},0.0)")
        else:
            print("No detect")
        
    except KeyboardInterrupt:
        print("No detect")
        sys.exit(0)
    except Exception as e:
        if args.debug if 'args' in locals() else False:
            print(f"❌ Error: {e}")
        print("No detect")
        sys.exit(0)

if __name__ == "__main__":
    main()