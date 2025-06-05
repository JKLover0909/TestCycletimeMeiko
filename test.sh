#!/bin/bash
# filepath: /root/Code/TestCycletimeMeiko/predict.sh

# Đặt đường dẫn cho file output
OUTPUT_FILE="/root/Code/TestCycletimeMeiko/prediction_result.txt"

echo "Đang gửi ảnh đến model bamdinh..."
# Thực hiện lệnh curl và lưu kết quả vào file
curl -X POST http://localhost:8080/predictions/bamdinh -T /tmp/test.jpg > "$OUTPUT_FILE"

# Kiểm tra kết quả
if [ $? -eq 0 ]; then
    echo "Dự đoán thành công. Kết quả đã được lưu vào file: $OUTPUT_FILE"
    echo "Nội dung file:"
    cat "$OUTPUT_FILE"
else
    echo "Có lỗi xảy ra khi gửi yêu cầu đến server."
fi