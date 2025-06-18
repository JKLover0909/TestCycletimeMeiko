#!/usr/bin/env python3
"""
Calculator.py - Rotated BBox Calculator v6.0
Input: Segmentation JSON from Segment.py
Output: Rotated bounding box dimensions + Rotated image visualization
"""

import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Tuple

class RotatedBBoxCalculator:
    """Calculate rotated bounding box from segmentation data"""
    
    def __init__(self):
        self.version = "6.0"
    
    def calculate_bbox_from_points(self, points: List[List[float]]) -> Dict[str, Any]:
        """Calculate bounding box from contour points"""
        try:
            if not points or len(points) == 0:
                raise ValueError("No points provided")
            
            points_array = np.array(points)
            
            # Get min/max coordinates
            x_coords = points_array[:, 0]
            y_coords = points_array[:, 1]
            
            x_min = float(np.min(x_coords))
            x_max = float(np.max(x_coords))
            y_min = float(np.min(y_coords))
            y_max = float(np.max(y_coords))
            
            width = x_max - x_min
            height = y_max - y_min
            area = width * height
            
            return {
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "width": width,
                "height": height,
                "area": area,
                "center": {
                    "x": (x_min + x_max) / 2,
                    "y": (y_min + y_max) / 2
                }
            }
            
        except Exception as e:
            raise Exception(f"BBox calculation failed: {e}")
    
    def rotate_image(self, image: np.ndarray, angle_degrees: float, center: Tuple[float, float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Rotate image around center point and return rotation matrix"""
        try:
            h, w = image.shape[:2]
            
            # Use image center if no center provided
            if center is None:
                center = (w // 2, h // 2)
            
            # Use NEGATIVE angle to match contour rotation direction
            rotation_matrix = cv2.getRotationMatrix2D(center, -angle_degrees, 1.0)
            
            # Calculate new image dimensions after rotation
            cos_angle = np.abs(rotation_matrix[0, 0])
            sin_angle = np.abs(rotation_matrix[0, 1])
            
            new_w = int((h * sin_angle) + (w * cos_angle))
            new_h = int((h * cos_angle) + (w * sin_angle))
            
            # Adjust rotation matrix to account for translation
            rotation_matrix[0, 2] += (new_w / 2) - center[0]
            rotation_matrix[1, 2] += (new_h / 2) - center[1]
            
            # Perform rotation
            rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), 
                                         flags=cv2.INTER_LINEAR, 
                                         borderMode=cv2.BORDER_CONSTANT, 
                                         borderValue=(255, 255, 255))
            
            return rotated_image, rotation_matrix
            
        except Exception as e:
            print(f"❌ Image rotation error: {e}")
            return image, np.eye(2, 3)
    
    def transform_points_with_rotation(self, points: List[List[float]], 
                                     rotation_matrix: np.ndarray) -> List[List[float]]:
        """Transform points using rotation matrix"""
        try:
            if not points:
                return points
                
            points_array = np.array(points)
            
            # Add homogeneous coordinate (1s)
            ones = np.ones((points_array.shape[0], 1))
            points_homogeneous = np.hstack([points_array, ones])
            
            # Apply transformation
            transformed_points = rotation_matrix.dot(points_homogeneous.T).T
            
            return transformed_points.tolist()
            
        except Exception as e:
            print(f"❌ Points transformation error: {e}")
            return points
    
    def create_rotated_visualization(self, image_path: str, original_bbox: Dict, 
                               rotated_bbox: Dict, class_name: str, theta: float,
                               original_points: List, rotated_points: List,
                               output_path: str) -> Dict[str, Any]:
        """Create visualization with rotated image and overlays"""
        try:
            print(f"🎨 Creating rotated visualization...")
            print(f"   Image: {os.path.basename(image_path)}")
            print(f"   Rotation: {theta:.2f}° (rotating both image and contour)")
            print(f"   Output: {output_path}")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot load image: {image_path}")
            
            # Convert to RGB for matplotlib
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_height, original_width = image_rgb.shape[:2]
            original_center = (original_width // 2, original_height // 2)
            
            print(f"   Original image size: {original_width} x {original_height}")
            
            # ✅ Use ORIGINAL contour points (not rotated_points)
            # We will rotate both image and original contour together
            contour_to_use = original_points
            print(f"   Using original contour with {len(contour_to_use)} points")
            
            # Rotate the image
            rotated_image_rgb, image_rotation_matrix = self.rotate_image(image_rgb, theta, original_center)
            rotated_height, rotated_width = rotated_image_rgb.shape[:2]
            
            print(f"   Rotated image size: {rotated_width} x {rotated_height}")
            
            # ✅ Transform ORIGINAL contour points using the same rotation matrix
            transformed_contour_points = self.transform_points_with_rotation(contour_to_use, image_rotation_matrix)
            
            # Calculate final bbox from transformed original contour
            if transformed_contour_points:
                final_rotated_bbox = self.calculate_bbox_from_points(transformed_contour_points)
                final_rotated_bbox["rotation_angle"] = theta
            else:
                final_rotated_bbox = original_bbox
                final_rotated_bbox["rotation_angle"] = theta
            
            # Create visualization - Single panel showing final result
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            fig.suptitle(f'Rotated Analysis - {class_name}\nOriginal Contour + Image both rotated by θ = {theta:.2f}°', 
                        fontsize=16, weight='bold')
            
            # Show rotated image
            ax.imshow(rotated_image_rgb)
            ax.set_title(f"Rotated Image + Rotated Original Contour + Final BBox", fontsize=14, weight='bold')
            
            # ✅ Draw transformed original contour (this is the main contour)
            if transformed_contour_points:
                contour_array = np.array(transformed_contour_points)
                ax.plot(contour_array[:, 0], contour_array[:, 1], 'r-', linewidth=1, label='Rotated Original Contour')
                ax.fill(contour_array[:, 0], contour_array[:, 1], 'red', alpha=0.2)
                
                # ✅ Draw final bounding box around rotated contour
                rect = plt.Rectangle(
                    (final_rotated_bbox["x_min"], final_rotated_bbox["y_min"]), 
                    final_rotated_bbox["width"], final_rotated_bbox["height"],
                    fill=False, color='lime', linewidth=2, label='Final Rotated BBox'
                )
                ax.add_patch(rect)
                
                # Add dimensions text
                ax.text(final_rotated_bbox["x_min"], final_rotated_bbox["y_min"]-30, 
                        f'Width: {final_rotated_bbox["width"]:.1f}px\nHeight: {final_rotated_bbox["height"]:.1f}px\nArea: {final_rotated_bbox["area"]:.1f}px²\nRotation: {theta:.2f}°',
                        color='lime', fontsize=12, weight='bold',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.8))
                
                # Add center point
                ax.plot(final_rotated_bbox["center"]["x"], final_rotated_bbox["center"]["y"], 
                        'yo', markersize=8, label='BBox Center', markeredgecolor='black', markeredgewidth=1)
                
                # Add corner markers
                corners = [
                    (final_rotated_bbox["x_min"], final_rotated_bbox["y_min"]),
                    (final_rotated_bbox["x_max"], final_rotated_bbox["y_min"]),
                    (final_rotated_bbox["x_max"], final_rotated_bbox["y_max"]),
                    (final_rotated_bbox["x_min"], final_rotated_bbox["y_max"])
                ]
                
                for x, y in corners:
                    ax.plot(x, y, 'gs', markersize=6, markeredgecolor='black', markeredgewidth=1)
            
            ax.legend(fontsize=12, loc='upper right')
            ax.axis('off')
            
            # Save as PNG
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', format='png')
            plt.close()
            
            print(f"✅ Rotated visualization saved: {output_path}")
            print(f"   Final dimensions: {final_rotated_bbox['width']:.1f} x {final_rotated_bbox['height']:.1f} pixels")
            print(f"   Both image and original contour rotated by {theta:.2f}°")
            
            return {"success": True, "final_bbox": final_rotated_bbox}
            
        except Exception as e:
            print(f"❌ Visualization error: {e}")
            return {"success": False, "error": str(e), "final_bbox": original_bbox}

def run_calculation(segmentation_json: str, image_path: str) -> Dict[str, Any]:
    """Main calculation pipeline with rotated image output"""
    
    try:
        print(f"🧮 Rotated BBox Calculator v6.0")
        print("=" * 50)
        print(f"📂 Segmentation Data: {segmentation_json}")
        print(f"📸 Image: {image_path}")
        
        # Load segmentation data
        if not os.path.exists(segmentation_json):
            raise FileNotFoundError(f"Segmentation file not found: {segmentation_json}")
        
        with open(segmentation_json, 'r') as f:
            seg_data = json.load(f)
        
        if not seg_data.get("success", False):
            raise ValueError("Segmentation data indicates failure")
        
        # Extract data
        input_data = seg_data.get("input_data", {})
        seg_results = seg_data.get("segmentation_results", {})
        
        class_name = input_data.get("class_name", "Unknown")
        theta = input_data.get("theta", 0)
        
        print(f"🎯 Class: {class_name}")
        print(f"🔄 Theta: {theta:.2f}°")
        
        original_contour = seg_results.get("original_contour", {})
        rotated_contour = seg_results.get("rotated_contour")
        
        if not original_contour:
            raise ValueError("No original contour data found")
        
        original_points = original_contour.get("points", [])
        
        # Initialize calculator
        calculator = RotatedBBoxCalculator()
        
        # Calculate original bounding box
        print("📊 Calculating original bounding box...")
        original_bbox = calculator.calculate_bbox_from_points(original_points)
        
        # Calculate rotated bounding box
        if rotated_contour and theta != 0:
            print("📊 Calculating rotated bounding box...")
            rotated_points = rotated_contour.get("points", [])
            if not rotated_points:
                raise ValueError("No rotated contour points found")
            rotated_bbox = calculator.calculate_bbox_from_points(rotated_points)
        else:
            print("📊 No rotation applied, using original bbox")
            rotated_points = original_points
            rotated_bbox = original_bbox.copy()
            rotated_bbox["rotation_angle"] = 0
        
        # Add rotation info
        rotated_bbox["rotation_angle"] = theta
        
        # Create output files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "/home/jkl0909/TestCycletimeMeiko/calculation_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create rotated visualization (PNG format)
        vis_file = f"{output_dir}/rotated_analysis_{class_name}_{timestamp}.png"
        print("🎨 Creating rotated image visualization...")
        
        vis_result = calculator.create_rotated_visualization(
            image_path, original_bbox, rotated_bbox, class_name, theta,
            original_points, rotated_points, vis_file
        )
        
        # Get final bbox from visualization result
        final_rotated_bbox = vis_result.get("final_bbox", rotated_bbox)
        success = vis_result.get("success", False)
        
        print(f"📊 FINAL RESULTS:")
        print(f"   Original BBox: {original_bbox['width']:.1f} x {original_bbox['height']:.1f} pixels (area: {original_bbox['area']:.1f})")
        print(f"   Rotated BBox: {rotated_bbox['width']:.1f} x {rotated_bbox['height']:.1f} pixels (area: {rotated_bbox['area']:.1f})")
        print(f"   Final Rotated BBox: {final_rotated_bbox['width']:.1f} x {final_rotated_bbox['height']:.1f} pixels (area: {final_rotated_bbox['area']:.1f})")
        print(f"   Rotation: {theta:.2f}°")
        print(f"   Contour Area: {original_contour.get('area', 0):.1f} pixels")
        
        # Prepare results
        results = {
            "success": True,
            "timestamp": timestamp,
            "input_data": {
                "class_name": class_name,
                "theta": theta,
                "image_path": image_path,
                "segmentation_file": segmentation_json
            },
            "calculation_results": {
                "original_bbox": original_bbox,
                "rotated_bbox": rotated_bbox,
                "final_rotated_bbox": final_rotated_bbox,
                "contour_properties": {
                    "original_area": original_contour.get("area", 0),
                    "original_perimeter": original_contour.get("perimeter", 0),
                    "center": original_contour.get("center", {"x": 0, "y": 0})
                }
            },
            "output_files": {
                "json_file": "",
                "rotated_visualization": vis_file if success else ""
            }
        }
        
        # Save JSON results
        json_file = f"{output_dir}/calculation_results_{class_name}_{timestamp}.json"
        results["output_files"]["json_file"] = json_file
        
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Output files:")
        print(f"   📋 Results: {json_file}")
        if success:
            print(f"   🖼️  Rotated Visualization: {vis_file}")
        print()
        
        return results
        
    except Exception as e:
        error_data = {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        return error_data

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description="Rotated BBox Calculator v6.0")
    parser.add_argument("--segmentation_data", required=True, help="Path to segmentation JSON file")
    parser.add_argument("--image_path", required=True, help="Path to original image")
    
    args = parser.parse_args()
    
    try:
        # Run calculation
        result = run_calculation(args.segmentation_data, args.image_path)
        
        if result.get("success", False):
            print("🎉 CALCULATION COMPLETED SUCCESSFULLY!")
            calc_results = result["calculation_results"]
            print(f"📊 Class: {result['input_data']['class_name']}")
            print(f"📐 Final Dimensions: {calc_results['final_rotated_bbox']['width']:.1f} x {calc_results['final_rotated_bbox']['height']:.1f} pixels")
            print(f"📏 Area: {calc_results['final_rotated_bbox']['area']:.1f} pixels")
            print(f"🔄 Rotation: {calc_results['final_rotated_bbox'].get('rotation_angle', calc_results['rotated_bbox']['rotation_angle']):.2f}°")
            print(f"🖼️  Rotated Image: {result['output_files']['rotated_visualization']}")
        else:
            print("❌ CALCULATION FAILED")
            print(f"Error: {result.get('error', 'Unknown error')}")
            sys.exit(1)
        
        # Output JSON for pipeline
        print(f"\n📋 JSON_OUTPUT_START")
        print(json.dumps(result, indent=2))
        print(f"📋 JSON_OUTPUT_END")
        
    except KeyboardInterrupt:
        print("\n⏹️  Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        print(f"❌ Calculator error: {e}")
        print(f"\n📋 JSON_OUTPUT_START")
        print(json.dumps(error_result, indent=2))
        print(f"📋 JSON_OUTPUT_END")
        sys.exit(1)

if __name__ == "__main__":
    main()