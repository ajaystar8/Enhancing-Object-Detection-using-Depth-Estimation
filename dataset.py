import os
import config

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from utils import (
    iou_width_height as iou
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


class YOLODataset(Dataset):
    def __init__(self, csv_file, img_dir, label_dir, anchors, image_size=416, S=None,
                 C=20, transform=None):
        if S is None:
            S = [13, 26, 52]
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.S = S
        self.C = C

        # anchors[i] -> [x, y] for 3 anchors in the ith scale.
        # Hence, anchors[0] + anchors[1] + anchors[2] -> will just put the coordinates of all anchors in one single list
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])

        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))

        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.roll(np.loadtxt(label_path, delimiter=" ", ndmin=2), 4, axis=1)

        # data augmentation
        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        # assumption: Each scale has 3 anchors and all scales have the same number of anchors.
        # 6 = object score + x + y + anchor height + anchor width + class label
        targets = [torch.zeros((self.num_anchors // 3, scale, scale, 6)) for scale in self.S]
        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)

            x, y, width, height, class_label = box

            # It is also assumed that each bounding box will be associated with only one class label.
            has_anchor = [False] * 3
            for anchor_index in anchor_indices:
                scale_idx = anchor_index // self.num_anchors_per_scale  # which scale
                S = self.S[scale_idx]

                anchor_on_scale = anchor_index % self.num_anchors_per_scale  # which anchor in this scale

                i, j = int(S * y), int(S * x)  # finds out the target cell of the image
                # choose a scale -> choose an anchor on this scale -> choose the cell in this scale
                # -> prob. of existence of an object
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i
                    width_cell, height_cell = width * S, height * S
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])

                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_index] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        return image, tuple(targets)

