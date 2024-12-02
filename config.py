import os

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2

DATASET = 'NYUv2'
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

NUM_WORKERS = 0  # For some reason, num_workers > 0 causes an error on my machine
BATCH_SIZE = 8
IMAGE_SIZE = 416

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
CONF_THRESHOLD = 0.05
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]

PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL = False

CHECKPOINT_DIR = os.path.join(os.getcwd(), "checkpoints")
IMG_DIR = DATASET + "/images/"
LABEL_DIR = DATASET + "/labels/"
NYU_PATH = "./resources/nyu_depth_v2_labeled.mat"

# YOLOv3 specific
CONFIG_FILE_PATH = "./weights/yolov3.cfg"
WEIGHTS_FILE_PATH = "./weights/yolov3.weights"

'''
In YoLov3, there are three prediction scale, where each predication scale has three anchor boxes. This means that
when the entire image is divided into cells, each cell of the image has three anchor boxes. For example, the first 
prediction scale has the image divided into grid cells of dimensions 13x13, and each cell of this prediction scale has 
three anchor boxes.
Each tuple consists of the height and width of the respective anchor box. 
Each list grouping consists of the dimensions (height and width) of the anchor boxes for that scale prediction level. 

Note: The dimensions have been scaled to be relative to the image. 
'''
ANCHORS = [
    # Largest anchor boxes -> used for prediction on the coarsest grid, where predicting larger anchor boxes is easier.
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]

# # NYU_DATASET_ANCHORS
# ANCHORS = [
#     [[0.07, 0.07], [0.1, 0.18], [0.21, 0.11]],
#     [[0.1, 0.43], [0.21, 0.27], [0.19, 0.63]],
#     [[0.56, 0.28], [0.35, 0.51], [0.71, 0.55]]
# ]

scale = 1.1
train_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
        A.PadIfNeeded(
            min_height=int(IMAGE_SIZE * scale),
            min_width=int(IMAGE_SIZE * scale),
            border_mode=cv2.BORDER_CONSTANT,
            value=0
        ),
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.ColorJitter(brightness=0.6, contrast=0.6,
                      saturation=0.6, hue=0.6, p=0.4),
        A.HorizontalFlip(p=0.5),
        A.Blur(p=0.1),
        A.CLAHE(p=0.1),
        A.Posterize(p=0.1),
        A.ToGray(p=0.1),
        A.ChannelShuffle(p=0.05),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255, ),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(
        format="yolo", min_visibility=0.4, label_fields=[], ),
)

test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255, ),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(
        format="yolo", min_visibility=0.4, label_fields=[]),
)

NYU_TARGET_CATEGORIES = ['bathtub', 'bed', 'bookshelf', 'box', 'chair', 'counter', 'desk', 'door', 'dresser',
                         'garbage bin', 'lamp', 'monitor', 'night stand', 'pillow', 'sink', 'sofa', 'table',
                         'television', 'toilet']

NUM_CLASSES = NYU_TARGET_CATEGORIES.__len__()
