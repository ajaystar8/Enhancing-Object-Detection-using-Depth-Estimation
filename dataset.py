import os

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader

import config
from utils import (
    cells_to_bboxes,
    iou_width_height,
    non_max_suppression as nms,
    plot_image
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
        # As there are three different prediction scales.
        self.num_anchors_per_scale = self.num_anchors // 3
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        """
        Steps involved:
        1) Load image and corresponding text file for labels.
        2) Create `target` (a regular Python list): 
            - For each scale, we have a 4D array: 
                - 1st dim: anchor box index
                - 2nd dim: scale dim 1
                - 3rd dim: scale dim 2
                - 4th dim: Target values (objectness_score, x, y, width, height, class_label)
        3) For every bounding box for an image: 
            - calculate IoU score for this box with all the anchor boxes. Argsort the anchor indices based on their IoU scores in descending order. 
            - For every anchor box: 
                - find the actual anchor and the scale it belongs to. 
                - calculate the cell coordinates of this box in this scale.
                - See if this anchor in this scale at this cell coordinates has been assigned to some other object before. If yes, skip this box. 
                - Else, mark the objectness score for this target position as 1. 
        """
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))

        label_path = os.path.join(
            self.label_dir, self.annotations.iloc[index, 1])
        # Label Format: [class_label, x, y, w, h] | Expected Format: [x, y, w, h, class_label]
        bboxes = np.roll(np.loadtxt(
            label_path, delimiter=" ", ndmin=2), 4, axis=1)

        # data augmentation
        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        # assumption: Each scale has 3 anchors and all scales have the same number of anchors.
        # 6 = object score + x + y + anchor height + anchor width + class label
        # For YoLov3, scale = [13, 26, 52]
        targets = [torch.zeros((self.num_anchors // 3, scale, scale, 6))
                   for scale in self.S]
        for box in bboxes:
            '''
            Since the grid cells are quite small, for easier IOU calculation, we can assume that anchor in a cell 
            and the ground truth bounding box's midpoint share the same midpoint. Hence, we can use iou_width_height.
            '''
            iou_anchors = iou_width_height(
                torch.tensor(box[2:4]), self.anchors)
            # So that we can get the index of the anchor box with the largest IOU with the target box.
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)

            x, y, width, height, class_label = box

            # It is also assumed that each bounding box will be associated with only one class label.
            has_anchor = [False] * 3
            for anchor_index in anchor_indices:
                scale_idx = anchor_index // self.num_anchors_per_scale  # which scale
                S = self.S[scale_idx]

                # which anchor in this scale
                anchor_on_scale = anchor_index % self.num_anchors_per_scale

                # finds out the target cell of the image in this scale
                i, j = int(S * y), int(S * x)
                # choose a scale -> choose an anchor on this scale -> choose the cell in this scale
                # -> prob. of existence of an object
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    # relative position of the object's center wrt the assigned grid cell
                    x_cell, y_cell = S * x - j, S * y - i
                    width_cell, height_cell = width * S, height * S
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell])

                    targets[scale_idx][anchor_on_scale,
                                       i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale,
                                       i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_index] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        return image, tuple(targets)


def test():
    anchors = config.ANCHORS

    transform = config.train_transforms

    dataset = YOLODataset(
        config.DATASET + '/train.csv',
        config.IMG_DIR,
        config.LABEL_DIR,
        S=[13, 26, 52],
        anchors=anchors,
        transform=transform,
    )
    S = [13, 26, 52]
    scaled_anchors = torch.tensor(anchors) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for x, y in loader:
        boxes = []

        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0]
        boxes = nms(boxes, iou_threshold=1,
                    threshold=0.7, box_format="midpoint")
        print(boxes)
        plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes)


if __name__ == "__main__":
    test()
