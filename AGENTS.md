# CLAUDE.md

Guidance for Claude Code (and similar agents) working in this repository.

## What this repo is

Đo cycle time + phát hiện lỗi PCB (Thừa đồng, Khuyết mạch, Ngắn mạch, Vết lõm, Xước) bằng YOLO (OBB) + TensorRT, chạy 100% offline. Pipeline: detect → segment → tính kích thước rotated bbox.

```
src/AIDetect.py    # multi-engine inference (TensorRT .engine), OBB + theta — offline, không cần internet
src/Segment.py     # segmentation theo class + tọa độ
src/Calculator.py  # tính rotated bbox từ segmentation JSON
src/Infer.py
results/           # output tính toán — KHÔNG commit (đã có trong .gitignore)
```

## Ràng buộc

- Đường dẫn model (`AIDetect.py`) hard-code tuyệt đối `/home/jkl0909/TestCycletimeMeiko/models/...` — khi sửa, giữ nguyên convention này hoặc hỏi lại nếu muốn đổi sang path tương đối (có thể ảnh hưởng tới cách script được gọi trên máy thật).
- `.gitignore` đã liệt kê `results/`, `data/`, `models/`, `bfg.jar` — `bfg.jar` (BFG Repo-Cleaner, 14.5MB) và `results/` từng bị track nhầm trước khi có rule, đã untrack ngày 2026-10-02.
- Model TensorRT `.engine` là build riêng theo GPU — không có sẵn trong môi trường agent, không chạy được `AIDetect.py` thật.

## Kiểm thử an toàn

```bash
python -c "import ast; ast.parse(open('src/Calculator.py').read())"  # syntax check, không cần GPU
```
Không chạy `AIDetect.py`/`Infer.py` thật (cần TensorRT engine + GPU).
