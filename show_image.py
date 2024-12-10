import os

import torch
import config
import torch.optim as optim
from models.yolov3 import YOLOv3
from utils.utils import get_evaluation_bboxes, get_loaders_nyu, load_checkpoint, plot_image


def main():
    config.CONF_THRESHOLD = 0.7
    config.NMS_IOU_THRESH = 0.5
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    train_loader, test_loader, train_eval_loader = get_loaders_nyu(mat_file_path=config.NYU_PATH, train_mode="rgb")
    load_checkpoint(
        os.path.join(".", "checkpoints", "rgb_loaded_weight_removed_low.pth.tar"), model, optimizer, config.DEVICE
    )

    # Scale anchors to each prediction scale
    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

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
        # Create a list entry for the train_idx if not already present
        if train_idx not in converted_boxes:
            converted_boxes[train_idx] = []
        # Append the converted box to the corresponding train_idx
        converted_boxes[train_idx].append([class_pred, confidence, x1, y1, x2, y2])

    # plot 5 images
    for i in range(5):
        image = test_loader.dataset[i][0].permute(1, 2, 0).to("cpu").numpy()
        bboxes = converted_boxes[i]
        plot_image(image, bboxes)

if __name__ == "__main__":
    main()