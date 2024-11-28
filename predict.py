from model import YOLOv3  # Replace with your model implementation
from torchvision import transforms
from PIL import Image
import torch

# Define model
model = YOLOv3(num_classes=20)  # Adjust `num_classes` for your dataset

# Load weights
model.load_state_dict(torch.load("./weights/model_weights.pth"))


# Switch to evaluation mode
model.eval()

# Preprocessing
image = Image.open("./data/PASCAL_VOC/images/000006.jpg").convert("RGB")
transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
])
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Inference
with torch.no_grad():
    outputs = model(input_tensor)

print((outputs[2].shape))
print(torch.max(outputs[2]))
print(torch.min(outputs[2]))
