from ts.torch_handler.base_handler import BaseHandler
import torch
import numpy as np
import onnxruntime as ort
import os

class YOLOv12OnnxHandler(BaseHandler):
    def initialize(self, ctx):
        self.device = 0 if torch.cuda.is_available() else -1  # GPU = 0, CPU = -1 for ONNX
        # Tạo ONNX Runtime session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 0 else ['CPUExecutionProvider']
        
        # Thay đổi đường dẫn tải mô hình
        import os
        model_path = os.path.join(ctx.system_properties.get("model_dir"), "model.onnx")
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Lấy thông tin về inputs
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.imgsz = 640

    def preprocess(self, data):
        import cv2, numpy as np, json, base64
        
        # Kiểm tra xem data có tồn tại không
        if data is None or len(data) == 0:
            raise ValueError("Không có dữ liệu đầu vào")
        
        # Lấy dữ liệu từ request
        input_data = data[0].get('body')
        if input_data is None:
            raise ValueError("Không tìm thấy dữ liệu trong request")
        
        # Xác định loại dữ liệu và xử lý phù hợp
        try:
            # Thử xử lý như binary data (từ curl -T)
            if isinstance(input_data, (bytes, bytearray)):
                nparr = np.frombuffer(input_data, np.uint8)
            else:
                # Thử xử lý như base64 string
                img_bin = base64.b64decode(input_data)
                nparr = np.frombuffer(img_bin, np.uint8)
                
            # Decode thành ảnh
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Kiểm tra xem ảnh có tồn tại không
            if img is None or img.size == 0:
                raise ValueError("Không thể đọc được ảnh từ dữ liệu đầu vào")
                
            # Xử lý ảnh
            img = cv2.resize(img, (self.imgsz, self.imgsz))
            img = img.transpose(2, 0, 1)  # HWC -> CHW
            img = np.ascontiguousarray(img) / 255.0  # normalize
            img = img.astype(np.float32)  # convert to float32
            return np.expand_dims(img, axis=0)  # add batch dimension
            
        except Exception as e:
            # Ghi log lỗi để debug
            import logging
            logging.error(f"Lỗi khi xử lý ảnh: {str(e)}")
            # Trả về lỗi rõ ràng
            raise ValueError(f"Lỗi khi xử lý ảnh: {str(e)}")

    def inference(self, inputs):
        # ONNX runtime requires input as dictionary
        ort_inputs = {self.input_name: inputs}
        outputs = self.session.run(self.output_names, ort_inputs)
        return outputs[0]  # Giả sử đầu ra đầu tiên là detection results

    def postprocess(self, preds):
        # Tùy dự án: NMS, format JSON
        # Chuyển từ numpy array sang list
        return [preds.tolist()]