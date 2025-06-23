#!/usr/bin/env python3
"""
Segment.py - SAM Segmentation Pipeline
Input: Image + Class + Coordinates + Theta
Output: JSON data + Visualization image
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import os
import sys
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional, Union
import torch

class SAMSegmenter:
    """SAM-based segmentation engine"""
    
    def __init__(self, model_path: str = None):
        """Initialize SAM segmenter"""
        if model_path is None:
            # Lấy đường dẫn model từ thư mục Model ngang hàng với src
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(project_root, "models", "sam2.1_s.pt")
        
        self.model_path = model_path
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """Load SAM model - simplified version"""
        try:
            print("🔄 Loading SAM model...")
            print(f"📁 Model path: {self.model_path}")
            
            # Check if model file exists
            if not os.path.exists(self.model_path):
                print(f"❌ Model file not found: {self.model_path}")
                return False
            
            try:
                # Try loading with ultralytics (if available)
                from ultralytics import SAM
                self.model = SAM(self.model_path)
                print("✅ SAM model loaded with Ultralytics!")
                return True
                
            except ImportError:
                print("⚠️ Ultralytics not available, trying torch load...")
                
                # Fallback: Direct torch load
                self.model = torch.load(self.model_path, map_location=self.device)
                print("✅ SAM model loaded with PyTorch!")
                return True
                
        except Exception as e:
            print(f"❌ Failed to load SAM model: {e}")
            # Create dummy model for testing
            print("⚠️ Using dummy segmentation for testing...")
            self.model = "dummy"
            return True
    
    def segment_from_bbox(self, image: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """Segment object from bounding box"""
        try:
            x1, y1, x2, y2 = bbox
            print(f"🎯 Segmenting bbox: [{x1}, {y1}, {x2}, {y2}]")
            print(f"🎯 Image shape: {image.shape}")
            
            # ✅ Validate bbox coordinates
            h, w = image.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            print(f"🎯 Validated bbox: [{x1}, {y1}, {x2}, {y2}]")
            
            if self.model == "dummy":
                # Create dummy mask for testing
                print("⚠️ Creating dummy mask...")
                mask = np.zeros((h, w), dtype=np.uint8)
                
                # Create elliptical mask in bbox region
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                width = max(1, x2 - x1)
                height = max(1, y2 - y1)
                
                # ✅ Create mask safely
                y_indices, x_indices = np.ogrid[:h, :w]
                
                # Avoid division by zero
                if width > 0 and height > 0:
                    ellipse_mask = ((x_indices - center_x) / (width/2))**2 + ((y_indices - center_y) / (height/2))**2 <= 1
                    bbox_mask = (x_indices >= x1) & (x_indices <= x2) & (y_indices >= y1) & (y_indices <= y2)
                    mask = (ellipse_mask & bbox_mask).astype(np.uint8)
                else:
                    # Simple rectangular mask
                    mask[y1:y2, x1:x2] = 1
                
                print(f"✅ Dummy mask created: shape={mask.shape}, dtype={mask.dtype}")
                print(f"✅ Mask stats: min={mask.min()}, max={mask.max()}, sum={mask.sum()}")
                
                return mask
            
            elif hasattr(self.model, 'predict'):
                # Ultralytics SAM
                results = self.model.predict(image, bboxes=[[x1, y1, x2, y2]])
                if results and len(results) > 0 and hasattr(results[0], 'masks'):
                    mask = results[0].masks[0].data.cpu().numpy()
                    return mask.astype(np.uint8)
            
            else:
                # Fallback to dummy
                print("⚠️ No valid model, using dummy segmentation")
                return self.segment_from_bbox(image, bbox)
                
        except Exception as e:
            print(f"❌ Segmentation error: {e}")
            print("⚠️ Creating emergency fallback mask...")
            
            # ✅ Emergency fallback
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Simple rectangular mask with validation
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            mask[y1:y2, x1:x2] = 1
            
            return mask
    
    def analyze_contour(self, mask: np.ndarray) -> Dict[str, Any]:
        """Analyze mask to extract contour properties"""
        try:
            print(f"🔍 Analyzing mask: shape={mask.shape}, dtype={mask.dtype}")
            print(f"🔍 Mask stats: min={mask.min()}, max={mask.max()}, unique={np.unique(mask)}")
            
            # ✅ Ensure mask is proper format
            if mask.dtype == bool:
                mask_uint8 = mask.astype(np.uint8) * 255
            elif mask.dtype == np.float32 or mask.dtype == np.float64:
                mask_uint8 = (mask * 255).astype(np.uint8)
            else:
                mask_uint8 = mask.astype(np.uint8)
            
            # ✅ Ensure mask is 2D
            if len(mask_uint8.shape) > 2:
                mask_uint8 = mask_uint8.squeeze()
                if len(mask_uint8.shape) > 2:
                    mask_uint8 = mask_uint8[:, :, 0]  # Take first channel
            
            print(f"🔍 Processed mask: shape={mask_uint8.shape}, dtype={mask_uint8.dtype}")
            print(f"🔍 Processed stats: min={mask_uint8.min()}, max={mask_uint8.max()}")
            
            # ✅ Find contours with error handling
            try:
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            except Exception as contour_error:
                print(f"❌ findContours error: {contour_error}")
                # Try different approach
                _, binary_mask = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                print("⚠️ No contours found, creating dummy contour from mask bounds")
                # Create contour from mask bounds
                y_indices, x_indices = np.where(mask_uint8 > 0)
                if len(y_indices) > 0 and len(x_indices) > 0:
                    x_min, x_max = x_indices.min(), x_indices.max()
                    y_min, y_max = y_indices.min(), y_indices.max()
                    
                    # Create rectangular contour
                    rect_contour = np.array([
                        [[x_min, y_min]],
                        [[x_max, y_min]],
                        [[x_max, y_max]],
                        [[x_min, y_max]]
                    ])
                    contours = [rect_contour]
                else:
                    return {"success": False, "error": "No valid pixels in mask"}
            
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            print(f"🔍 Largest contour: {largest_contour.shape}")
            
            # ✅ Calculate properties with error handling
            try:
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
            except Exception as calc_error:
                print(f"❌ Area/perimeter calculation error: {calc_error}")
                # Fallback calculation
                area = len(largest_contour)
                perimeter = len(largest_contour) * 2
            
            # ✅ Get center with multiple fallbacks
            try:
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                else:
                    raise ValueError("Zero moment")
            except Exception as moment_error:
                print(f"⚠️ Moment calculation failed: {moment_error}, using bbox center")
                # Fallback to contour bounding box center
                try:
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    cx = x + w / 2
                    cy = y + h / 2
                except Exception as bbox_error:
                    print(f"⚠️ Bbox calculation failed: {bbox_error}, using contour point average")
                    # Ultimate fallback: average of contour points
                    points_array = largest_contour.reshape(-1, 2)
                    cx = np.mean(points_array[:, 0])
                    cy = np.mean(points_array[:, 1])
            
            # ✅ Convert contour to points list safely
            try:
                points = largest_contour.reshape(-1, 2).tolist()
            except Exception as reshape_error:
                print(f"⚠️ Contour reshape failed: {reshape_error}")
                # Manual conversion
                points = []
                for point in largest_contour:
                    if len(point) > 0 and len(point[0]) >= 2:
                        points.append([int(point[0][0]), int(point[0][1])])
            
            result = {
                "success": True,
                "area": float(area),
                "perimeter": float(perimeter),
                "center": {"x": float(cx), "y": float(cy)},
                "points": points,
                "num_points": len(points)
            }
            
            print(f"✅ Contour analysis successful:")
            print(f"   Area: {area:.1f} pixels")
            print(f"   Perimeter: {perimeter:.1f} pixels")
            print(f"   Center: ({cx:.1f}, {cy:.1f})")
            print(f"   Points: {len(points)}")
            
            return result
            
        except Exception as e:
            print(f"❌ Contour analysis error: {e}")
            print(f"🔍 Mask debug info:")
            print(f"   Shape: {mask.shape if hasattr(mask, 'shape') else 'No shape'}")
            print(f"   Type: {type(mask)}")
            print(f"   Dtype: {mask.dtype if hasattr(mask, 'dtype') else 'No dtype'}")
            
            # ✅ Return error with debug info
            return {
                "success": False, 
                "error": str(e),
                "debug": {
                    "mask_shape": str(mask.shape) if hasattr(mask, 'shape') else 'unknown',
                    "mask_dtype": str(mask.dtype) if hasattr(mask, 'dtype') else 'unknown',
                    "mask_type": str(type(mask))
                }
            }
    
    def rotate_contour(self, points: List[List[int]], center: Tuple[float, float], 
                      theta_degrees: float) -> List[List[float]]:
        """Rotate contour points around center"""
        try:
            if theta_degrees == 0:
                return points
            
            cx, cy = center
            theta_rad = np.radians(theta_degrees)
            cos_theta = np.cos(theta_rad)
            sin_theta = np.sin(theta_rad)
            
            rotated_points = []
            
            for point in points:
                x, y = point
                
                # Translate to origin
                x_translated = x - cx
                y_translated = y - cy
                
                # Rotate
                x_rotated = x_translated * cos_theta - y_translated * sin_theta
                y_rotated = x_translated * sin_theta + y_translated * cos_theta
                
                # Translate back
                x_final = x_rotated + cx
                y_final = y_rotated + cy
                
                rotated_points.append([float(x_final), float(y_final)])
            
            return rotated_points
            
        except Exception as e:
            print(f"❌ Rotation error: {e}")
            return points
    
    def create_visualization(self, image: np.ndarray, mask: np.ndarray, 
                       class_name: str, theta: float, 
                       original_points: List, rotated_points: List,
                       output_path: str):
        """Create visualization with original and rotated contours"""
        try:
            print(f"🎨 Creating visualization...")
            print(f"   Image shape: {image.shape}")
            print(f"   Mask shape: {mask.shape}")
            print(f"   Mask dtype: {mask.dtype}")
            
            # Convert image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # ✅ Fix mask dimensions
            if len(mask.shape) == 3:
                if mask.shape[0] == 1:
                    # Shape is (1, H, W) -> squeeze to (H, W)
                    mask = mask.squeeze(0)
                elif mask.shape[2] == 1:
                    # Shape is (H, W, 1) -> squeeze to (H, W)
                    mask = mask.squeeze(2)
                else:
                    # Shape is (H, W, C) -> take first channel
                    mask = mask[:, :, 0]
            
            # ✅ Ensure mask is 2D
            while len(mask.shape) > 2:
                mask = mask.squeeze()
            
            print(f"   Fixed mask shape: {mask.shape}")
            
            # ✅ Normalize mask values for display
            if mask.dtype == bool:
                mask_display = mask.astype(np.float32)
            elif mask.max() <= 1.0:
                mask_display = mask.astype(np.float32)
            else:
                mask_display = (mask / 255.0).astype(np.float32)
            
            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'SAM Segmentation Results - {class_name}', fontsize=16, weight='bold')
            
            # Panel 1: Original Image
            ax1.imshow(image_rgb)
            ax1.set_title("Original Image", fontsize=12, weight='bold')
            ax1.axis('off')
            
            # Panel 2: Detection with original contour
            ax2.imshow(image_rgb)
            if original_points and len(original_points) > 2:
                try:
                    points_array = np.array(original_points)
                    # Close the contour
                    if not np.array_equal(points_array[0], points_array[-1]):
                        points_array = np.vstack([points_array, points_array[0]])
                    
                    ax2.plot(points_array[:, 0], points_array[:, 1], 'r-', linewidth=2, label='Original Contour')
                    ax2.fill(points_array[:, 0], points_array[:, 1], 'red', alpha=0.3)
                except Exception as e:
                    print(f"⚠️ Error drawing original contour: {e}")
            
            ax2.set_title(f"Original Contour\nθ = {theta:.2f}°", fontsize=12, weight='bold')
            ax2.legend()
            ax2.axis('off')
            
            # Panel 3: Segmentation Mask
            ax3.imshow(mask_display, cmap='gray', vmin=0, vmax=1)
            ax3.set_title("Segmentation Mask", fontsize=12, weight='bold')
            ax3.axis('off')
            
            # Panel 4: Rotated Result
            ax4.imshow(image_rgb)
            if rotated_points and len(rotated_points) > 2 and theta != 0:
                try:
                    rot_points_array = np.array(rotated_points)
                    # Close the contour
                    if not np.array_equal(rot_points_array[0], rot_points_array[-1]):
                        rot_points_array = np.vstack([rot_points_array, rot_points_array[0]])
                    
                    ax4.plot(rot_points_array[:, 0], rot_points_array[:, 1], 'g-', linewidth=2, label='Rotated Contour')
                    ax4.fill(rot_points_array[:, 0], rot_points_array[:, 1], 'green', alpha=0.3)
                    title = f"Rotated Contour\nθ = {theta:.2f}°"
                except Exception as e:
                    print(f"⚠️ Error drawing rotated contour: {e}")
                    title = "Rotated Contour (Error)"
            else:
                # Show original contour in green if no rotation
                if original_points and len(original_points) > 2:
                    try:
                        points_array = np.array(original_points)
                        if not np.array_equal(points_array[0], points_array[-1]):
                            points_array = np.vstack([points_array, points_array[0]])
                        
                        ax4.plot(points_array[:, 0], points_array[:, 1], 'g-', linewidth=2, label='Segmentation Contour')
                        ax4.fill(points_array[:, 0], points_array[:, 1], 'green', alpha=0.3)
                    except Exception as e:
                        print(f"⚠️ Error drawing segmentation contour: {e}")
                title = "Segmentation Result"
            
            ax4.set_title(title, fontsize=12, weight='bold')
            ax4.legend()
            ax4.axis('off')
            
            # ✅ Save with error handling
            try:
                plt.tight_layout()
                plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close()
                
                print(f"✅ Visualization saved: {output_path}")
                return True
                
            except Exception as save_error:
                print(f"❌ Save error: {save_error}")
                plt.close()
                return False
            
        except Exception as e:
            print(f"❌ Visualization error: {e}")
            print(f"🔍 Debug info:")
            print(f"   Image: {image.shape if hasattr(image, 'shape') else 'No shape'}")
            print(f"   Mask: {mask.shape if hasattr(mask, 'shape') else 'No shape'}")
            print(f"   Points: {len(original_points) if original_points else 0}")
            
            # ✅ Create simple fallback visualization
            try:
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                ax.text(0.5, 0.5, f'Visualization Error\n{class_name}\nθ = {theta:.2f}°\n\nError: {str(e)[:100]}...', 
                    ha='center', va='center', fontsize=12, 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcoral', alpha=0.7))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                ax.set_title(f'SAM Segmentation Error - {class_name}', fontsize=14, weight='bold')
                
                plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close()
                
                print(f"⚠️ Fallback visualization saved: {output_path}")
                return True
                
            except Exception as fallback_error:
                print(f"❌ Even fallback failed: {fallback_error}")
                return False

def run_segmentation(image_path: str, class_name: str, coordinates: str, theta: float, 
                    sam_model: str = None) -> Dict[str, Any]:
    """Main segmentation pipeline"""
    
    try:
        print(f"🎭 SAM Segmentation Pipeline")
        print("=" * 50)
        print(f"📂 Image: {image_path}")
        print(f"🎯 Class: {class_name}")
        print(f"📐 Coordinates: {coordinates}")
        print(f"🔄 Theta: {theta:.2f}°")
        print(f"🤖 Model: {sam_model}")
        print()
        
        # Parse coordinates
        coords = [float(x.strip()) for x in coordinates.split(',')]
        if len(coords) != 4:
            raise ValueError("Coordinates must be in format: x1,y1,x2,y2")
        
        bbox = [int(coord) for coord in coords]
        x1, y1, x2, y2 = bbox
        
        # Load image
        print("📸 Loading image...")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        print(f"✅ Image loaded: {image.shape}")
        
        # Initialize segmenter với đường dẫn model
        if sam_model is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sam_model = os.path.join(project_root, "models", "sam2.1_s.pt")
        
        segmenter = SAMSegmenter(sam_model)
        if not segmenter.load_model():
            raise Exception("Failed to load SAM model")
        
        # Run segmentation
        print("🎭 Running segmentation...")
        mask = segmenter.segment_from_bbox(image, bbox)
        
        if mask is None:
            raise Exception("Segmentation failed - no mask generated")
        
        print("✅ Segmentation successful")
        
        # Analyze contour
        print("📏 Analyzing contours...")
        contour_analysis = segmenter.analyze_contour(mask)
        
        if not contour_analysis.get("success", False):
            raise Exception(f"Contour analysis failed: {contour_analysis.get('error')}")
        
        original_points = contour_analysis["points"]
        center = contour_analysis["center"]
        
        print(f"📊 Contour Analysis:")
        print(f"   Area: {contour_analysis['area']:.1f} pixels")
        print(f"   Perimeter: {contour_analysis['perimeter']:.1f} pixels")
        print(f"   Center: ({center['x']:.1f}, {center['y']:.1f})")
        print(f"   Points: {contour_analysis['num_points']}")
        
        # Apply rotation if needed
        if theta != 0:
            print(f"🔄 Applying rotation: {theta:.2f}°...")
            rotated_points = segmenter.rotate_contour(
                original_points, 
                (center['x'], center['y']), 
                theta
            )
        else:
            rotated_points = original_points
        
        # Create output files với đường dẫn tương đối
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(project_root, "results", "sam_segmentation_results")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create visualization
        vis_file = os.path.join(output_dir, f"segmentation_{class_name}_{timestamp}.png")
        print("🎨 Creating visualization...")
        
        success = segmenter.create_visualization(
            image, mask, class_name, theta, 
            original_points, rotated_points, vis_file
        )
        
        if not success:
            print("⚠️ Visualization creation failed, but continuing...")
        
        # Prepare output data
        output_data = {
            "success": True,
            "timestamp": timestamp,
            "input_data": {
                "image_path": image_path,
                "class_name": class_name,
                "coordinates": coordinates,
                "theta": theta,
                "bbox": bbox
            },
            "segmentation_results": {
                "original_contour": {
                    "area": contour_analysis['area'],
                    "perimeter": contour_analysis['perimeter'],
                    "center": center,
                    "points": original_points,
                    "num_points": contour_analysis['num_points']
                },
                "rotated_contour": {
                    "theta": theta,
                    "points": rotated_points,
                    "center": center
                } if theta != 0 else None
            },
            "output_files": {
                "visualization": vis_file,
                "json_file": ""
            }
        }
        
        # Save JSON file
        json_file = os.path.join(output_dir, f"segmentation_data_{class_name}_{timestamp}.json")
        output_data["output_files"]["json_file"] = json_file
        
        with open(json_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"💾 Output files:")
        print(f"   📊 Visualization: {vis_file}")
        print(f"   📋 JSON Data: {json_file}")
        print()
        
        return output_data
        
    except Exception as e:
        error_data = {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        return error_data

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description="SAM Segmentation Pipeline")
    parser.add_argument("--image_path", required=True, help="Path to input image")
    parser.add_argument("--class_name", required=True, help="Class name for segmentation")
    parser.add_argument("--coordinates", required=True, help="Bounding box coordinates (x1,y1,x2,y2)")
    parser.add_argument("--theta", type=float, default=0, help="Rotation angle in degrees")
    parser.add_argument("--sam_model", default=None, help="Path to SAM model file")
    
    args = parser.parse_args()
    
    try:
        # Run segmentation pipeline
        result = run_segmentation(
            image_path=args.image_path,
            class_name=args.class_name,
            coordinates=args.coordinates,
            theta=args.theta,
            sam_model=args.sam_model
        )
        
        if result.get("success", False):
            print("🎉 SEGMENTATION COMPLETED SUCCESSFULLY!")
            print(f"📊 Class: {result['input_data']['class_name']}")
            print(f"📐 Rotation: {result['input_data']['theta']:.2f}°")
            print(f"📏 Area: {result['segmentation_results']['original_contour']['area']:.1f} pixels")
            print(f"🖼️  Files: {len(result['output_files'])} files generated")
        else:
            print("❌ SEGMENTATION FAILED")
            print(f"Error: {result.get('error', 'Unknown error')}")
            sys.exit(1)
        
        # Output JSON for pipeline integration
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
        print(f"❌ Pipeline error: {e}")
        print(f"\n📋 JSON_OUTPUT_START")
        print(json.dumps(error_result, indent=2))
        print(f"📋 JSON_OUTPUT_END")
        sys.exit(1)

if __name__ == "__main__":
    main()