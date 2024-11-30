import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import config
from load_weights import LoadYOLOWeights
from model import YOLOv3
from utils import cells_to_bboxes, plot_image

# Load weights
config_file_path = "./weights/yolov3.cfg"
weights_file_path = "./weights/yolov3.weights"

model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
weight_loader = LoadYOLOWeights(config_file_path, weights_file_path)
weight_loader.load(model)

# Switch to evaluation mode
model.eval()

# Preprocessing
image = Image.open("./data/PASCAL_VOC/images/000006.jpg").convert("RGB")
transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
])
input_tensor = transform(image).unsqueeze(0)

# Inference
with torch.no_grad():
    outputs = model(input_tensor)

    bboxes = cells_to_bboxes(outputs[2], np.array(config.ANCHORS[2]), 52, True)
    print(len(bboxes[0]))
    # plot_image(image, bboxes)



