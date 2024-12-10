from PIL import Image
from torchvision import transforms

from models.yolov3 import YOLOv3
from utils.utils import *

model_type = str(input("Enter model type to train (yolov3 or resyolov3): "))

if model_type == "resyolov3":
    model = ResYOLOv3(num_classes=config.NUM_CLASSES).to("cpu")
elif model_type == "yolov3":
    model = YOLOv3(num_classes=config.NUM_CLASSES).to("cpu")
else:
    raise NotImplementedError

checkpoint_path = None
try:
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "Experiment_4.pth.tar")
    checkpoint = load_checkpoint(checkpoint_path, model)
except TypeError as e:
    raise TypeError(
        f"Failed to load checkpoint from '{checkpoint_path}'. Ensure the model architecture matches the checkpoint. "
        f"Error: {e}"
    )

# Switch to evaluation mode
model.eval()

# Preprocessing
idx = np.random.randint(1449)
print(idx)
image = Image.open(f"./NYUD/images/{idx}.png").convert("RGB")
hha = Image.open(f"./NYUD/hha/{idx}.png").convert("RGB")
transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
])
input_tensor = transform(image).unsqueeze(0).to("cpu")
hha_tensor = transform(hha).unsqueeze(0).to("cpu")

# Inference
with torch.no_grad():
    outputs = model(input_tensor, hha_tensor)
    bboxes_all_scales = []

    for i in range(len(outputs)):
        S = outputs[i].shape[2]
        bboxes = cells_to_bboxes(outputs[i], np.array(config.ANCHORS[i]), S, True)[0]
        bboxes_all_scales.extend(bboxes)

    nms_bboxes_all_scales = non_max_suppression(bboxes_all_scales, iou_threshold=1, threshold=0.3, box_format="corners")
    plot_image(image, nms_bboxes_all_scales)
