#!/bin/bash
# filepath: /root/Code/TestCycletimeMeiko/package_models.sh

# Thiết lập các đường dẫn
INPUT_DIR="/root/Code/TestCycletimeMeiko/ModelYolov12n_ONNX"
HANDLER_PATH="/root/Code/TestCycletimeMeiko/torchserve/handlers/onnx_handler.py"
OUTPUT_DIR="/root/Code/TestCycletimeMeiko/torchserve/model_store"
TEMP_DIR="/tmp/model_packaging"

# Tạo thư mục output và temp nếu chưa tồn tại
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TEMP_DIR"

# Xóa tất cả các file MAR cũ
rm -f "$OUTPUT_DIR"/*.mar

# Hiển thị thông tin
echo "Đường dẫn đầu vào: $INPUT_DIR"
echo "Đường dẫn handler: $HANDLER_PATH"
echo "Đường dẫn đầu ra: $OUTPUT_DIR"

# Kiểm tra các file trong thư mục đầu vào
echo "Files trong thư mục đầu vào:"
ls -la "$INPUT_DIR"/*.onnx 2>/dev/null || echo "Không tìm thấy file ONNX nào"

# Lặp qua tất cả các tệp .onnx trong thư mục đầu vào
for f in "$INPUT_DIR"/*.onnx; do
    if [ -f "$f" ]; then
        filename=$(basename "$f")
        model_name="${filename%.*}"
        
        echo "Đang đóng gói mô hình ONNX: $model_name"
        echo "Tên mô hình: $model_name"
        
        # Tạo thư mục tạm cho mô hình này
        model_temp_dir="$TEMP_DIR/$model_name"
        mkdir -p "$model_temp_dir"
        
        # Sao chép và đổi tên file ONNX thành model.onnx
        cp "$f" "$model_temp_dir/model.onnx"
        
        # Kiểm tra xem file tồn tại không
        echo "Kiểm tra file tạm: $model_temp_dir/model.onnx"
        ls -la "$model_temp_dir/model.onnx"
        
        # Chạy lệnh đóng gói với file đã đổi tên
        torch-model-archiver --model-name "$model_name" \
                             --version 1.0 \
                             --serialized-file "$model_temp_dir/model.onnx" \
                             --handler "$HANDLER_PATH" \
                             --export-path "$OUTPUT_DIR" \
                             --force
        
        if [ $? -eq 0 ]; then
            echo "Đã đóng gói mô hình ONNX $model_name thành công."
        else
            echo "Lỗi khi đóng gói mô hình ONNX $model_name"
        fi
        echo "-----------------------------------"
    fi
done

# Dọn dẹp thư mục tạm
rm -rf "$TEMP_DIR"

# Kiểm tra các file trong thư mục đầu ra
echo "Files trong thư mục đầu ra:"
ls -la "$OUTPUT_DIR"/*.mar 2>/dev/null || echo "Không tìm thấy file MAR nào"

echo "Tất cả các mô hình ONNX đã được đóng gói xong."
echo "Khởi động lại TorchServe với lệnh:"
echo "torchserve --start --model-store \"$OUTPUT_DIR\" --models all --ts-config config.properties"