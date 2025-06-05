@echo off
setlocal enabledelayedexpansion

:: Thiết lập các đường dẫn
set INPUT_DIR=/root/Code/TestCycletimeMeiko/ModelYolov12n_ONNX
set HANDLER_PATH=/root/Code/TestCycletimeMeiko/torchserve/handlers/onnx_handler.py
set OUTPUT_DIR=/root/Code/TestCycletimeMeiko/torchserve/model_store

:: Tạo thư mục output nếu chưa tồn tại
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

:: Hiển thị thông tin
echo Đường dẫn đầu vào: %INPUT_DIR%
echo Đường dẫn handler: %HANDLER_PATH%
echo Đường dẫn đầu ra: %OUTPUT_DIR%

:: Kiểm tra các file trong thư mục đầu vào
echo Files trong thư mục đầu vào:
dir "%INPUT_DIR%\*.onnx"

:: Lặp qua tất cả các tệp .onnx trong thư mục đầu vào
for %%f in ("%INPUT_DIR%\*.onnx") do (
    echo Đang đóng gói mô hình ONNX: %%~nf
    
    :: Lấy tên mô hình không có phần mở rộng
    set "model_name=%%~nf"
    
    echo Tên mô hình: !model_name!
    
    :: Chạy lệnh đóng gói
    torch-model-archiver --model-name !model_name! ^
                          --version 1.0 ^
                          --serialized-file "%%f" ^
                          --handler "%HANDLER_PATH%" ^
                          --export-path "%OUTPUT_DIR%"
    
    if %ERRORLEVEL% EQU 0 (
        echo Đã đóng gói mô hình ONNX !model_name! thành công.
    ) else (
        echo Lỗi khi đóng gói mô hình ONNX !model_name!
    )
    echo -----------------------------------
)

:: Kiểm tra các file trong thư mục đầu ra
echo Files trong thư mục đầu ra:
dir "%OUTPUT_DIR%\*.mar"

echo Tất cả các mô hình ONNX đã được đóng gói xong.
echo Khởi động TorchServe với lệnh:
echo torchserve --start --model-store "%OUTPUT_DIR%" --models all --ts-config config.properties

endlocal