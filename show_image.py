import os

import torch.optim as optim

import config
from models.resyolov3 import ResYOLOv3
from models.resyolov3depth import ResYOLOv3Depth
from models.yolov3 import YOLOv3
from utils.utils import get_evaluation_bboxes, get_loaders_nyu, load_checkpoint, plot_image


def main():
    config.CONF_THRESHOLD = 0.6
    config.NMS_IOU_THRESH = 0.5

    model_type = str(input("Enter model type to train (yolov3 or resyolov3_add or resyolov3_depth): "))

    if model_type == "resyolov3_add":
        model = ResYOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    elif model_type == "resyolov3_depth":
        model = ResYOLOv3Depth(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    elif model_type == "yolov3":
        model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    else:
        raise NotImplementedError

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    train_mode = str(input("Enter train mode (rgb or hha or fusion): "))
    train_loader, test_loader, train_eval_loader = get_loaders_nyu(mat_file_path=config.NYU_PATH, train_mode=train_mode)

    ckpt_name = str(input("Enter the checkpoint file name (include .pth.tar): "))
    try:
        load_checkpoint(
            os.path.join("checkpoints", ckpt_name), model, optimizer, config.DEVICE
        )
    except TypeError as e:
        raise TypeError(
            f"Failed to load checkpoint from '{os.path.join(config.CHECKPOINT_DIR, ckpt_name)}'. "
            f"Ensure the model architecture matches the checkpoint. "
            f"Error: {e}"
        )

    pred_boxes, true_boxes = get_evaluation_bboxes(
        test_loader,
        model,
        iou_threshold=config.NMS_IOU_THRESH,
        anchors=config.ANCHORS,
        threshold=config.CONF_THRESHOLD,
    )

    # pred_boxes is in [train_idx, class_prediction, prob_score, x1, y1, x2, y2] format
    # we need to convert it to [class pred, confidence, x, y, width, height] format
    converted_boxes = {}
    for box in pred_boxes:
        # Extract values
        train_idx, class_pred, confidence, x1, y1, x2, y2 = box
        # Create a list entry for the idx if not already present
        if train_idx not in converted_boxes:
            converted_boxes[train_idx] = []
        # Append the converted box to the corresponding train_idx
        converted_boxes[train_idx].append([class_pred, confidence, x1, y1, x2, y2])

    # plot 5 images
    for i in range(5):
        if model_type == "yolov3":
            image = test_loader.dataset[i][0].permute(1, 2, 0).to("cpu").numpy()
        else:
            # Get the first three channels of the image
            image = test_loader.dataset[i][0][:3].permute(1, 2, 0).to("cpu").numpy()
        bboxes = converted_boxes[i]
        plot_image(image, bboxes)

if __name__ == "__main__":
    main()
