import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

import config


def get_train_test_transforms_list(additional_targets=None):
    # irrespective of datatype of the pixel values
    train_transforms_list = [
        A.LongestMaxSize(max_size=int(config.IMAGE_SIZE * config.scale)),
        A.PadIfNeeded(
            min_height=int(config.IMAGE_SIZE * config.scale),
            min_width=int(config.IMAGE_SIZE * config.scale),
            border_mode=cv2.BORDER_CONSTANT,
            value=0
        ),
        A.RandomCrop(width=config.IMAGE_SIZE, height=config.IMAGE_SIZE),
        A.ColorJitter(brightness=0.6, contrast=0.6,
                      saturation=0.6, hue=0.6, p=0.4),
        A.HorizontalFlip(p=0.5),
        A.Blur(p=0.1),
    ]

    test_transforms_list = [
        A.LongestMaxSize(max_size=config.IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=config.IMAGE_SIZE, min_width=config.IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255, ),
        ToTensorV2(),
    ]

    if additional_targets is None:
        # These transformations can be applied only when pixel values are integers
        train_transforms_list.extend([
            A.CLAHE(p=0.1),
            A.Posterize(p=0.1),
            A.ToGray(p=0.1),
            A.ChannelShuffle(p=0.05),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255, ),
            ToTensorV2(),
        ])
        train_transforms = A.Compose(
            train_transforms_list,
            bbox_params=A.BboxParams(
                format="yolo", min_visibility=0.4, label_fields=[], ),
        )
        test_transforms = A.Compose(
            test_transforms_list,
            bbox_params=A.BboxParams(
                format="yolo", min_visibility=0.4, label_fields=[]),
        )
    else:
        # to ensure the correct order of transformations are being applied to the image
        # we are separating out these transformations
        train_transforms_list.extend([
            A.ToGray(p=0.1),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255, ),
            ToTensorV2(),
        ])
        train_transforms = A.Compose(
            train_transforms_list,
            bbox_params=A.BboxParams(
                format="yolo", min_visibility=0.4, label_fields=[], ),
            additional_targets=additional_targets,
        )
        test_transforms = A.Compose(
            test_transforms_list,
            bbox_params=A.BboxParams(
                format="yolo", min_visibility=0.4, label_fields=[]),
            additional_targets=additional_targets,
        )

    return train_transforms, test_transforms
