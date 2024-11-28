import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import (
    cells_to_bboxes,
    iou_width_height,
    non_max_suppression,
    plot_image
)
import config

class NYUYoloDataset(Dataset):
    def __init__(self, mat_file, anchors, image_size=416, S=None, C=40, transform=None, indices=None):
        """
        Args:
            mat_file (str): Path to the .mat file containing the dataset.
            anchors (list): List of anchors for each scale.
            image_size (int): Size to which images will be resized.
            S (list): List of scales (e.g., [13, 26, 52] for YOLOv3).
            C (int): Number of classes in the dataset.
            transform (callable, optional): Transformations to apply to the images and labels.
            indices (list, optional): List of indices specifying the subset of the dataset to use.
        """
        import h5py

        with h5py.File(mat_file, "r") as f:
            self.data = {key: np.array(f[key]) for key in f.keys()}
            self.images = np.rot90(self.data["images"], k=-1, axes=(2, 3))  # (N, 3, H, W), Rotate HxW axes
            self.instances = np.rot90(self.data["instances"], k=-1, axes=(1, 2)) # (N, H, W)
            self.labels = np.rot90(self.data["labels"], k=-1, axes=(1, 2)) # (N, H, W)
                # Use only the specified subset of data if indices are provided
        if indices is not None:
            self.images = self.images[indices]
            self.instances = self.instances[indices]
            self.labels = self.labels[indices]
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.ignore_iou_thresh = 0.5
        self.image_size = image_size
        self.S = S if S else [13, 26, 52]
        self.C = C
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """
        Returns:
            image: Tensor of shape [3, img_size, img_size]
            targets: List of tensors for each scale.
        """
        # Load image, instance map, and label map
        image = self.images[index]  # (C, H, W)
        instance_map = self.instances[index]  # (H, W)
        label_map = self.labels[index]  # (H, W)

        # Convert image to (H, W, C) and scale to [0, 255]
        image = np.transpose(image, (1, 2, 0)).astype(np.uint8)

        # Extract bounding boxes
        bboxes = self._get_bounding_boxes(instance_map, label_map)

        # Apply transformations
        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        # Prepare YOLO targets
        targets = [torch.zeros((self.num_anchors_per_scale, scale, scale, 6)) for scale in self.S]
        for box in bboxes:
            iou_anchors = iou_width_height(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True)
            x, y, width, height, class_label = box

            has_anchor = [False] * len(self.S)
            for anchor_index in anchor_indices:
                scale_idx = int(anchor_index // self.num_anchors_per_scale)
                S = self.S[scale_idx]
                anchor_on_scale = int(anchor_index % self.num_anchors_per_scale)
                i, j = int(S * y), int(S * x)

                if not has_anchor[scale_idx]:
                    x_cell, y_cell = S * x - j, S * y - i
                    width_cell, height_cell = width * S, height * S
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True
                elif not has_anchor[scale_idx] and iou_anchors[anchor_index] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        return image, tuple(targets)
    
    def _get_instance_masks(self, img_object_labels, img_instances):
        '''
        Extracts instance masks and labels from the object labels and instance maps.
        Args:
            imgObjectLabels (numpy.ndarray): 2D array (H, W) with object labels for each pixel.
            imgInstances (numpy.ndarray): 2D array (H, W) with instance IDs for each pixel.
        Returns:
            instanceMasks (numpy.ndarray): 3D array (H, W, N) with binary masks for each instance.
            instanceLabels (numpy.ndarray): 1D array (N,) with class labels for each instance.
        '''
        H, W = img_object_labels.shape

        # Find unique (label, instance_id) pairs
        pairs = np.unique(np.stack([img_object_labels.ravel(), img_instances.ravel()], axis=1), axis=0)
        pairs = pairs[np.sum(pairs, axis=1) != 0]  # Remove (0, 0) pairs (background)

        N = pairs.shape[0]
        instance_masks = np.zeros((H, W, N), dtype=bool)
        instance_labels = np.zeros(N, dtype=int)

        # Generate instance masks and labels
        for i, (label, instance_id) in enumerate(pairs):
            instance_masks[:, :, i] = (img_object_labels == label) & (img_instances == instance_id)
            instance_labels[i] = label

        return instance_masks, instance_labels

    def _get_bounding_boxes(self, instance_map, label_map):
        """
        Extracts bounding boxes and class labels using get_instance_masks.
        Args:
            label_map (numpy.ndarray): 2D array (H, W) with class labels for each pixel.
            instance_map (numpy.ndarray): 2D array (H, W) with instance IDs for each pixel.
        Returns:
            bboxes (list): List of bounding boxes in the format [x_center, y_center, width, height, class_label].
        """
        # Use get_instance_masks to extract binary masks and instance labels
        instance_masks, instance_labels = self._get_instance_masks(label_map, instance_map)

        bboxes = []
        for i in range(instance_masks.shape[-1]):
            mask = instance_masks[:, :, i]  # Binary mask for the current instance
            class_label = instance_labels[i] - 1  # Corresponding class label

            # Find the bounding box from the mask
            y, x = np.where(mask)  # Get mask coordinates
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()

            # Normalize bounding box coordinates
            H, W = mask.shape
            x_center = (x_min + x_max) / 2 / W
            y_center = (y_min + y_max) / 2 / H
            width = (x_max - x_min) / W
            height = (y_max - y_min) / H

            # If the bounding box does not meet the requirement of the YOLO model, skip it
            if x_center <= 0 or y_center <= 0 or width <= 0 or height <= 0 or x_center >= 1 or y_center >= 1 or width >= 1 or height >= 1:
                continue

            # Append the bounding box and class label
            bboxes.append([x_center, y_center, width, height, class_label])

        return bboxes
    

def test():
    anchors = config.ANCHORS
    dataset = NYUYoloDataset(
        mat_file=config.NYU_PATH,
        image_size=416,
        anchors=anchors,
        S=[13, 26, 52],
        C=config.NYU_LABELS.__len__(),
        transform=config.train_transforms,
    )

    S = [13, 26, 52]
    scaled_anchors = torch.tensor(anchors) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    index = 3
    loader = DataLoader(
        dataset=dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )
    for x, y in loader:
        boxes = []
        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0]
        boxes = non_max_suppression(boxes, iou_threshold=1,
                    threshold=0.7, box_format="midpoint")
        plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes)
        break
    
if __name__ == "__main__":
    test()
