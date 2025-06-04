@echo off
setlocal enabledelayedexpansion

:: Thiết lập các đường dẫn
set INPUT_DIR=C:\Users\1\MeikoAI\SonCode\TestCycletimeMeiko\ModelYolov12n_TS
set HANDLER_PATH=C:\Users\1\MeikoAI\SonCode\TestCycletimeMeiko\torchserve\handlers\yolo_handler.py
set OUTPUT_DIR=C:\Users\1\MeikoAI\SonCode\TestCycletimeMeiko\torchserve\model_store

:: Tạo thư mục output nếu chưa tồn tại
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

:: Hiển thị thông tin
echo Đường dẫn đầu vào: %INPUT_DIR%
echo Đường dẫn handler: %HANDLER_PATH%
echo Đường dẫn đầu ra: %OUTPUT_DIR%

:: Kiểm tra các file trong thư mục đầu vào
echo Files trong thư mục đầu vào:
dir "%INPUT_DIR%\*.torchscript"

:: Lặp qua tất cả các tệp .torchscript trong thư mục đầu vào
for %%f in ("%INPUT_DIR%\*.torchscript") do (
    echo Đang đóng gói mô hình: %%~nf
    
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
        echo Đã đóng gói mô hình !model_name! thành công.
    ) else (
        echo Lỗi khi đóng gói mô hình !model_name!
    )
    echo -----------------------------------
)

:: Kiểm tra các file trong thư mục đầu ra
echo Files trong thư mục đầu ra:
dir "%OUTPUT_DIR%\*.mar"

echo Tất cả các mô hình đã được đóng gói xong.
echo Khởi động TorchServe với lệnh:
echo torchserve --start --model-store "%OUTPUT_DIR%" --models all --ts-config config.properties

endlocal