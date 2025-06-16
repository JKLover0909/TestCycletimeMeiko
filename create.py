#!/usr/bin/env python3
"""
Script to create the folder structure for project_pcb_defects as specified.
Usage:
    python create_project_structure.py --root /path/to/project_pcb_defects
"""
import os
import argparse

# Define folder hierarchy
FOLDERS = [
    "data/raw",
    "data/processed",
    "model_repository/yolov11_defect1/1",
    "model_repository/yolov11_defect1/config",
    "model_repository/yolov11_defect2/1",
    "model_repository/yolov11_defect2/config",
    "model_repository/yolov11_defect3/1",
    "model_repository/yolov11_defect3/config",
    "model_repository/yolov11_defect4/1",
    "model_repository/yolov11_defect4/config",
    "model_repository/yolov11_defect5/1",
    "model_repository/yolov11_defect5/config",
    "model_repository/sam_segmenter/1",
    "model_repository/sam_segmenter/config",
    "src",
    "scripts",
]

# Define placeholder files to create (relative to root)
FILES = [
    "model_repository/yolov11_defect1/1/model.pt",
    "model_repository/yolov11_defect1/config/config.pbtxt",
    "model_repository/yolov11_defect2/config/config.pbtxt",
    "model_repository/yolov11_defect3/config/config.pbtxt",
    "model_repository/yolov11_defect4/config/config.pbtxt",
    "model_repository/yolov11_defect5/config/config.pbtxt",
    "model_repository/sam_segmenter/1/sam.pt",
    "model_repository/sam_segmenter/config/config.pbtxt",
    "docker-compose.yml",
    "requirements.txt",
    "README.md",
]

def create_structure(root):
    # Create directories
    for folder in FOLDERS:
        path = os.path.join(root, folder)
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")

    # Create placeholder files
    for file_rel in FILES:
        file_path = os.path.join(root, file_rel)
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write("# Placeholder for " + os.path.basename(file_path) + "\n")
            print(f"Created file: {file_path}")
        else:
            print(f"File already exists: {file_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create project directory structure')
    parser.add_argument('--root', type=str, default='project_pcb_defects',
                        help='Root directory for the project')
    args = parser.parse_args()
    create_structure(args.root)
