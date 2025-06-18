#!/usr/bin/env python3
"""
Infer.py - Complete Inference Pipeline
Integrates AI Detection → SAM Segmentation → Rotated BBox Calculation
"""

import subprocess
import json
import os
import sys
import argparse
import tempfile
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, List

class InferencePipeline:
    """Complete inference pipeline"""
    
    def __init__(self, temp_dir: str = None):
        """Initialize inference pipeline"""
        self.version = "3.0"
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="inference_")
        self.ensure_directories()
    
    def ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [
            "sam_segmentation_results",
            "calculation_results", 
            "inference_results",
            self.temp_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def run_ai_detection(self, image_path: str, confidence: float = 0.5) -> Dict[str, Any]:
        """Run AI detection step"""
        try:
            print(f"🤖 Step 1: Running AI Detection...")
            print(f"   Image: {image_path}")
            print(f"   Confidence: {confidence}")
            
            # Build command
            cmd = [
                "python", "AIDetect.py",
                "--image_path", image_path,
                "--confidence", str(confidence)
            ]
            
            print(f"🔍 Command: {' '.join(cmd)}")
            
            # Run detection
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                # Parse text output from AIDetect.py
                detection_data = self.parse_ai_detection_output(result.stdout)
                
                if detection_data:
                    print(f"✅ Detection successful: {len(detection_data)} objects found")
                    for i, det in enumerate(detection_data):
                        print(f"   {i+1}. {det['class_name']} (conf: {det['confidence']:.3f})")
                    return {"success": True, "detections": detection_data}
                else:
                    return {"success": False, "error": "No valid detections found"}
            else:
                print(f"❌ Detection failed: {result.stderr}")
                return {"success": False, "error": result.stderr}
                
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "AI Detection timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def parse_ai_detection_output(self, output: str) -> List[Dict[str, Any]]:
        """Parse text output from AIDetect.py - Updated for actual format"""
        try:
            detections = []
            lines = output.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                
                # Match format: "📍 ThuaDong: bbox=(217,241,292,319), θ=-89.6°, conf=0.934"
                if "📍" in line and "bbox=" in line:
                    try:
                        # Extract class name
                        class_name = line.split("📍")[1].split(":")[0].strip()
                        
                        # Extract bbox coordinates
                        bbox_match = line.split("bbox=(")[1].split(")")[0]
                        bbox_coords = [float(x.strip()) for x in bbox_match.split(",")[:4]]
                        
                        # Extract confidence
                        confidence = 0.5  # Default
                        if "conf=" in line:
                            conf_str = line.split("conf=")[1].strip()
                            confidence = float(conf_str.split()[0])
                        
                        # Extract angle
                        angle = 0
                        if "θ=" in line:
                            angle_str = line.split("θ=")[1].split("°")[0]
                            angle = float(angle_str)
                        
                        detection = {
                            "class_name": class_name,
                            "confidence": confidence,
                            "bbox": bbox_coords,
                            "coordinates": f"{int(bbox_coords[0])},{int(bbox_coords[1])},{int(bbox_coords[2])},{int(bbox_coords[3])}",
                            "angle": angle
                        }
                        
                        detections.append(detection)
                        print(f"✅ Successfully parsed: {class_name} at {detection['coordinates']}")
                        
                    except Exception as e:
                        print(f"⚠️ Error parsing line: {line}")
                        print(f"   Error: {e}")
            
            return detections
            
        except Exception as e:
            print(f"❌ Detection parsing failed: {e}")
            return []
    
    def run_segmentation(self, image_path: str, detection: Dict[str, Any]) -> Dict[str, Any]:
        """Run SAM segmentation step"""
        try:
            print(f"🎭 Step 2: Running SAM Segmentation...")
            print(f"   Class: {detection['class_name']}")
            print(f"   Coordinates: {detection['coordinates']}")
            print(f"   Angle: {detection['angle']:.2f}°")
            
            # Build command
            cmd = [
                "python", "Segment.py",
                "--image_path", image_path,
                "--class_name", detection['class_name'],
                "--coordinates", detection['coordinates'],
                "--theta", str(detection['angle'])
            ]
            
            print(f"🔍 Command: {' '.join(cmd)}")
            
            # Run segmentation
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Parse JSON output from Segment.py
                seg_data = self.parse_json_output(result.stdout)
                
                if seg_data and seg_data.get("success", False):
                    print(f"✅ Segmentation successful")
                    print(f"   Area: {seg_data['segmentation_results']['original_contour']['area']:.1f} pixels")
                    print(f"   Points: {seg_data['segmentation_results']['original_contour']['num_points']}")
                    
                    # Get JSON file path
                    json_file = seg_data['output_files']['json_file']
                    seg_data['json_file_path'] = json_file
                    
                    return {"success": True, "data": seg_data}
                else:
                    error_msg = seg_data.get("error", "Segmentation failed") if seg_data else "No JSON output"
                    return {"success": False, "error": error_msg}
            else:
                print(f"❌ Segmentation failed: {result.stderr}")
                return {"success": False, "error": result.stderr}
                
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Segmentation timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def run_calculation(self, segmentation_json: str, image_path: str) -> Dict[str, Any]:
        """Run rotated bbox calculation step"""
        try:
            print(f"🧮 Step 3: Running Rotated BBox Calculation...")
            print(f"   Segmentation data: {os.path.basename(segmentation_json)}")
            
            # Build command
            cmd = [
                "python", "Calculator.py",
                "--segmentation_data", segmentation_json,
                "--image_path", image_path
            ]
            
            print(f"🔍 Command: {' '.join(cmd)}")
            
            # Run calculation
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                # Parse JSON output from Calculator.py
                calc_data = self.parse_json_output(result.stdout)
                
                if calc_data and calc_data.get("success", False):
                    print(f"✅ Calculation successful")
                    bbox = calc_data['calculation_results']['rotated_bbox']
                    print(f"   Dimensions: {bbox['width']:.1f} x {bbox['height']:.1f} pixels")
                    print(f"   Area: {bbox['area']:.1f} pixels")
                    print(f"   Rotation: {bbox['rotation_angle']:.2f}°")
                    
                    return {"success": True, "data": calc_data}
                else:
                    error_msg = calc_data.get("error", "Calculation failed") if calc_data else "No JSON output"
                    return {"success": False, "error": error_msg}
            else:
                print(f"❌ Calculation failed: {result.stderr}")
                return {"success": False, "error": result.stderr}
                
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Calculation timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def parse_json_output(self, output: str) -> Optional[Dict[str, Any]]:
        """Parse JSON output from subprocess"""
        try:
            lines = output.strip().split('\n')
            json_start = -1
            json_end = -1
            
            # Find JSON markers
            for i, line in enumerate(lines):
                if "JSON_OUTPUT_START" in line:
                    json_start = i + 1
                elif "JSON_OUTPUT_END" in line:
                    json_end = i
                    break
            
            if json_start != -1 and json_end != -1:
                json_lines = lines[json_start:json_end]
                json_text = '\n'.join(json_lines)
                
                try:
                    return json.loads(json_text)
                except json.JSONDecodeError as e:
                    print(f"❌ JSON parse error: {e}")
                    print(f"JSON text: {json_text[:200]}...")
                    return None
            else:
                print("⚠️ No JSON markers found in output")
                return None
                
        except Exception as e:
            print(f"❌ JSON parsing error: {e}")
            return None
    
    def run_complete_inference(self, image_path: str, confidence: float = 0.5) -> Dict[str, Any]:
        """Run complete inference pipeline"""
        
        try:
            print(f"🚀 Complete Inference Pipeline v{self.version}")
            print("=" * 60)
            print(f"📂 Image: {image_path}")
            print(f"🎯 Confidence: {confidence}")
            print(f"🕒 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            # Validate input
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Step 1: AI Detection
            detection_result = self.run_ai_detection(image_path, confidence)
            if not detection_result.get("success", False):
                return {
                    "success": False,
                    "step": "detection",
                    "error": detection_result.get("error", "Detection failed")
                }
            
            detections = detection_result["detections"]
            if not detections:
                return {
                    "success": False,
                    "step": "detection", 
                    "error": "No objects detected"
                }
            
            # Process first detection (can be extended for multiple objects)
            detection = detections[0]
            print(f"\n📊 Processing: {detection['class_name']} (confidence: {detection['confidence']:.3f})")
            
            # Step 2: Segmentation
            segmentation_result = self.run_segmentation(image_path, detection)
            if not segmentation_result.get("success", False):
                return {
                    "success": False,
                    "step": "segmentation",
                    "error": segmentation_result.get("error", "Segmentation failed")
                }
            
            seg_data = segmentation_result["data"]
            segmentation_json = seg_data["json_file_path"]
            
            # Step 3: Calculation
            calculation_result = self.run_calculation(segmentation_json, image_path)
            if not calculation_result.get("success", False):
                return {
                    "success": False,
                    "step": "calculation",
                    "error": calculation_result.get("error", "Calculation failed")
                }
            
            calc_data = calculation_result["data"]
            
            # Compile final results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            final_results = {
                "success": True,
                "timestamp": timestamp,
                "pipeline_version": self.version,
                "input": {
                    "image_path": image_path,
                    "confidence_threshold": confidence
                },
                "results": {
                    "detection": {
                        "class_name": detection["class_name"],
                        "confidence": detection["confidence"],
                        "bbox": detection["bbox"],
                        "angle": detection["angle"]
                    },
                    "segmentation": {
                        "area": seg_data["segmentation_results"]["original_contour"]["area"],
                        "perimeter": seg_data["segmentation_results"]["original_contour"]["perimeter"],
                        "center": seg_data["segmentation_results"]["original_contour"]["center"],
                        "num_points": seg_data["segmentation_results"]["original_contour"]["num_points"]
                    },
                    "final_dimensions": {
                        "width": calc_data["calculation_results"]["rotated_bbox"]["width"],
                        "height": calc_data["calculation_results"]["rotated_bbox"]["height"],
                        "area": calc_data["calculation_results"]["rotated_bbox"]["area"],
                        "rotation_angle": calc_data["calculation_results"]["rotated_bbox"]["rotation_angle"]
                    }
                },
                "output_files": {
                    "segmentation_visualization": seg_data["output_files"]["visualization"],
                    "segmentation_data": seg_data["output_files"]["json_file"],
                    "calculation_visualization": calc_data["output_files"]["visualization"],
                    "calculation_data": calc_data["output_files"]["json_file"],
                    "final_results": ""  # Will be filled below
                }
            }
            
            # Save final results
            final_json = f"inference_results/inference_results_{detection['class_name']}_{timestamp}.json"
            final_results["output_files"]["final_results"] = final_json
            
            with open(final_json, 'w') as f:
                json.dump(final_results, f, indent=2)
            
            print(f"\n🎉 INFERENCE COMPLETED SUCCESSFULLY!")
            print(f"📊 Object: {detection['class_name']} (confidence: {detection['confidence']:.3f})")
            print(f"📐 Final Dimensions: {final_results['results']['final_dimensions']['width']:.1f} x {final_results['results']['final_dimensions']['height']:.1f} pixels")
            print(f"📏 Area: {final_results['results']['final_dimensions']['area']:.1f} pixels")
            print(f"🔄 Rotation: {final_results['results']['final_dimensions']['rotation_angle']:.2f}°")
            print(f"🕒 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            print(f"📁 Output Files:")
            for file_type, file_path in final_results["output_files"].items():
                if file_path:
                    print(f"   {file_type}: {file_path}")
            
            return final_results
            
        except Exception as e:
            error_result = {
                "success": False,
                "step": "pipeline",
                "error": str(e),
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            print(f"❌ Pipeline error: {e}")
            return error_result
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print(f"🧹 Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description="Complete Inference Pipeline v3.0")
    parser.add_argument("--image_path", required=True, help="Path to input image")
    parser.add_argument("--confidence", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--temp_dir", help="Temporary directory for processing")
    parser.add_argument("--keep_temp", action="store_true", help="Keep temporary files")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = InferencePipeline(args.temp_dir)
    
    try:
        # Run inference
        result = pipeline.run_complete_inference(args.image_path, args.confidence)
        
        if result.get("success", False):
            print("\n✅ PIPELINE COMPLETED SUCCESSFULLY")
            sys.exit(0)
        else:
            print(f"\n❌ PIPELINE FAILED at step: {result.get('step', 'unknown')}")
            print(f"Error: {result.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
    finally:
        # Cleanup unless requested to keep
        if not args.keep_temp:
            pipeline.cleanup()

if __name__ == "__main__":
    main()