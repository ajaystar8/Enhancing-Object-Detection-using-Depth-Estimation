import os.path
import pickle
from glob import glob

import h5py
import numpy as np
import torch
from PIL import Image
from pymatreader import read_mat
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import config
from utils.transforms import get_train_test_transforms_list
from utils.utils import (
    cells_to_bboxes,
    iou_width_height,
    non_max_suppression,
    plot_image,
    get_nyu_target_category_indices
)


class NYUYoloDataset(Dataset):
    def __init__(self, mat_file, hha_dir, anchors, image_size=416, S=None, C=40, apply_transforms=True, indices=None,
                 use_only_target_categories=False, generate_labels=False, train_mode="rgb"):
        """
        Args:
            mat_file (str): Path to the .mat file containing the dataset.
            anchors (list): List of anchors for each scale.
            image_size (int): Size to which images will be resized.
            S (list): List of scales (e.g., [13, 26, 52] for YOLOv3).
            C (int): Number of classes in the dataset.
            transform_list (callable, optional): Transformations to apply to the images and labels.
            indices (list, optional): List of indices specifying the subset of the dataset to use.
        """

        # Data loading
        if not os.path.exists('resources/sub_nyu_hha_mat.pkl'):
            with h5py.File(mat_file, "r") as f:
                self.data = {key: np.array(f[key]) for key in f.keys()}
                self.images = np.rot90(self.data["images"], k=-1, axes=(2, 3))  # (N, 3, H, W), Rotate HxW axes
                self.instances = np.rot90(self.data["instances"], k=-1, axes=(1, 2))  # (N, H, W)
                self.labels = np.rot90(self.data["labels"], k=-1, axes=(1, 2))  # (N, H, W)
                self.names = np.array(read_mat(mat_file, variable_names=['names'])["names"])
                hha_raw = self._get_hha_encoded_images(hha_dir)

                self.hha = np.rot90(hha_raw, k=-1, axes=(2, 3))
                sub_mat = {"images": self.data['images'], "instances": self.data['instances'],
                           "labels": self.data['labels'], "names": self.names, "hha": hha_raw}
                with open('../resources/sub_nyu_hha_mat.pkl', 'wb') as f1:
                    pickle.dump(sub_mat, f1, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('resources/sub_nyu_hha_mat.pkl', 'rb') as f:
                self.data = pickle.load(f)
                self.images = np.rot90(self.data["images"], k=-1, axes=(2, 3))  # (N, 3, H, W), Rotate HxW axes
                self.instances = np.rot90(self.data["instances"], k=-1, axes=(1, 2))  # (N, H, W)
                self.labels = np.rot90(self.data["labels"], k=-1, axes=(1, 2))  # (N, H, W)
                self.hha = np.rot90(self.data['hha'], k=-1, axes=(2, 3))
                self.names = self.data['names']

        if indices is not None:
            self.images = self.images[indices]
            self.hha = self.hha[indices]
            self.instances = self.instances[indices]
            self.labels = self.labels[indices]

        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3

        self.ignore_iou_thresh = 0.5
        self.image_size = image_size

        self.S = S if S else [13, 26, 52]
        self.C = C
        self.apply_transforms = apply_transforms
        self.use_only_target_categories = use_only_target_categories
        self.generate_labels = generate_labels
        self.train_mode = train_mode.strip().lower()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """
        Returns:
            image: Tensor of shape [3, img_size, img_size]
            targets: List of tensors for each scale.
        """
        # Load image, instance map, and label map
        image, depth = None, None
        if self.train_mode == "rgb":
            image = self.images[index]
        elif self.train_mode == "hha":
            image = self.hha[index]
        elif self.train_mode == "fusion":
            image = self.images[index]
            depth = self.hha[index]
        else:
            raise NotImplementedError

        instance_map = self.instances[index]  # (H, W)
        label_map = self.labels[index]  # (H, W)

        # Convert image to (H, W, C) and scale to [0, 255]
        image = np.transpose(image, (1, 2, 0)).astype(np.uint8)
        if depth is not None:
            depth = np.transpose(depth, (1, 2, 0)).astype(np.float32)

        # Extract bounding boxes
        bboxes = self._get_bounding_boxes(instance_map, label_map,
                                          get_nyu_target_category_indices()
                                          if self.use_only_target_categories else None)

        if self.generate_labels:
            # create label text files in YOLO format
            np_bboxes = np.array(bboxes, dtype=np.float32)
            # keep class label first
            if np_bboxes.size != 0:
                np_bboxes = np.roll(np_bboxes, 1, axis=1)
                np.savetxt(os.path.join("NYUD", "labels", f"{index}.txt"), np_bboxes, fmt='%d %f %f %f %f')
            else:
                np.savetxt(os.path.join("NYUD", "labels", f"{index}.txt"), np_bboxes)

        # Apply transformations
        if self.apply_transforms:
            if self.train_mode == "fusion":
                train_transforms, test_transforms = get_train_test_transforms_list({"depth": "image"})
            else:
                train_transforms, test_transforms = get_train_test_transforms_list()

            if self.train_mode == "fusion":
                augmentations = train_transforms(image=image, depth=depth, bboxes=bboxes)
                depth = augmentations["depth"]
            else:
                augmentations = train_transforms(image=image, bboxes=bboxes)
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

        if self.train_mode == "fusion":
            return torch.cat([image, depth], dim=0), tuple(targets)
        else:
            return image, tuple(targets)

    @staticmethod
    def _get_hha_encoded_images(hha_images_path: str):
        hha_image_paths = glob(os.path.join("../..", hha_images_path, "*.png"))
        hha_images = []
        for path in hha_image_paths:
            image = Image.open(path)
            image = np.array(image).transpose(2, 1, 0)
            hha_images.append(image)
        return np.array(hha_images)

    @staticmethod
    def _get_instance_masks(img_object_labels, img_instances):
        """
        Extracts instance masks and labels from the object labels and instance maps.
        Args:
            img_object_labels (numpy.ndarray): 2D array (H, W) with object labels for each pixel.
            img_instances (numpy.ndarray): 2D array (H, W) with instance IDs for each pixel.
        Returns:
            instanceMasks (numpy.ndarray): 3D array (H, W, N) with binary masks for each instance.
            instanceLabels (numpy.ndarray): 1D array (N,) with class labels for each instance.
        """
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

    def _get_bounding_boxes(self, instance_map, label_map, nyu_target_indices=None):
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
        # print(instance_masks.shape[-1])
        for i in range(instance_masks.shape[-1]):
            mask = instance_masks[:, :, i]  # Binary mask for the current instance
            class_label = instance_labels[i] - 1  # Corresponding class label

            # Skip the instance if the class label is not in the target indices
            if nyu_target_indices is not None and class_label not in nyu_target_indices:
                continue

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
            if (x_center <= 0 or y_center <= 0 or width <= 0 or height <= 0 or
                    x_center >= 1 or y_center >= 1 or width >= 1 or height >= 1):
                continue

            # Append the bounding box and class label
            # Convert the class label to the index in the target indices
            # print(nyu_target_indices.index(class_label), x_center, y_center, width, height)
            bboxes.append([x_center, y_center, width, height,
                           nyu_target_indices.index(class_label) if self.use_only_target_categories else class_label])

        return bboxes


def test():
    # anchors = config.ANCHORS
    train_mode = str(input("Enter train mode (rgb or hha or fusion): "))
    anchors = config.ANCHORS
    dataset = NYUYoloDataset(
        mat_file=config.NYU_PATH,
        hha_dir=config.HHA_IMAGE_DIR,
        image_size=416,
        anchors=anchors,
        S=[13, 26, 52],
        C=config.NYU_LABELS.__len__(),
        apply_transforms=True,
        use_only_target_categories=True,
        generate_labels=True,
        train_mode=train_mode
    )

    S = [13, 26, 52]
    scaled_anchors = torch.tensor(anchors) / (
            1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    # index = 3
    loader = DataLoader(
        dataset=dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )
    for x, y in tqdm(loader):
        image, depth = x[:, 0:3, :, :], x[:, 3:, :, :]
        boxes = []
        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0]
        boxes = non_max_suppression(boxes, iou_threshold=1,
                                    threshold=0.7, box_format="midpoint")
        plot_image(image[0].permute(1, 2, 0).to("cpu"), boxes)
        break


if __name__ == "__main__":
    test()
