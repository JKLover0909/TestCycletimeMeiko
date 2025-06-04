# Script kiểm tra mô hình TorchScript
import torch
from PIL import Image
import torchvision.transforms as transforms
import os

def test_torchscript_model(model_path, test_image_path):
    # Tải mô hình
    model = torch.jit.load(model_path)
    model.eval()
    
    # Chuẩn bị ảnh đầu vào
    img = Image.open(test_image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(img).unsqueeze(0)
    
    # Chạy dự đoán
    with torch.no_grad():
        outputs = model(input_tensor)
        
    print(f"Mô hình {os.path.basename(model_path)} đã chạy thành công!")
    return outputs