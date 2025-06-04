@echo off
setlocal enabledelayedexpansion

:: Thiết lập các đường dẫn
set INPUT_DIR=C:\Users\1\MeikoAI\SonCode\TestCycletimeMeiko\ModelYolov12n_TS
set HANDLER_DIR=C:\Users\1\MeikoAI\SonCode\TestCycletimeMeiko\torchserve\handlers
set OUTPUT_DIR=C:\Users\1\MeikoAI\SonCode\TestCycletimeMeiko\torchserve\model_store

:: Tạo thư mục output nếu chưa tồn tại
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo Đang xóa các file .mar cũ...
del /Q "%OUTPUT_DIR%\*.mar"

:: Lặp qua tất cả các tệp .torchscript trong thư mục đầu vào
for %%f in ("%INPUT_DIR%\*.torchscript") do (
    echo Đang đóng gói mô hình: %%~nf
    
    :: Lấy tên mô hình không có phần mở rộng
    set "model_name=%%~nf"
    
    :: Chạy lệnh đóng gói với handler đúng
    torch-model-archiver --model-name !model_name! ^
                          --version 1.0 ^
                          --serialized-file "%%f" ^
                          --handler ^
                          --extra-files "%HANDLER_DIR%\yolo_handler.py" ^
                          --export-path "%OUTPUT_DIR%"
    
    if %ERRORLEVEL% EQU 0 (
        echo Đã đóng gói mô hình !model_name! thành công.
    ) else (
        echo Lỗi khi đóng gói mô hình !model_name!
    )
    echo -----------------------------------
)

echo Đã hoàn tất đóng gói lại các mô hình.
echo Khởi động TorchServe với lệnh:
echo conda activate ts_yolo12
echo cd "%~dp0..\torchserve"
echo torchserve --start --model-store model_store --models all --ts-config config.properties

endlocal