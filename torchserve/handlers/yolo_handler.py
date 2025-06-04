from ts.torch_handler.base_handler import BaseHandler
import torch, torchvision
from ultralytics.yolo.utils import ops

class YOLOv12NanoHandler(BaseHandler):
    def initialize(self, ctx):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.jit.load(ctx.model_uri).to(self.device).eval()
        self.imgsz = 640

    def preprocess(self, data):
        import cv2, numpy as np, json, base64
        img_bin = base64.b64decode(data[0]['body'])
        nparr = np.frombuffer(img_bin, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (self.imgsz, self.imgsz))
        img = torch.from_numpy(img).permute(2,0,1).float()/255
        return img.unsqueeze(0).to(self.device)

    def inference(self, inputs):
        with torch.no_grad():
            pred = self.model(inputs)[0]        # tensor Nx7 (xyxy + conf + cls)
        return pred

    def postprocess(self, preds):
        # Tùy dự án: NMS, format JSON
        return [preds.cpu().tolist()]