import torch
import torch.optim as optim
from tqdm import tqdm

import config
from load_weights import LoadYOLOWeights
from loss import YoloLoss
from model import YOLOv3
from utils import (
    mean_average_precision,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders_nyu
)


def train_fn(train_loader, model, optimizer, loss_fn, scaled_anchors):
    loop = tqdm(train_loader, leave=True)
    losses = []

    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE)
        )
        out = model(x)

        loss = (
                loss_fn(out[0], y0, scaled_anchors[0]) +
                loss_fn(out[1], y1, scaled_anchors[1]) +
                loss_fn(out[2], y2, scaled_anchors[2])
        )

        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)


def main():
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    weight_loader = LoadYOLOWeights(config.CONFIG_FILE_PATH, config.WEIGHTS_FILE_PATH)
    weight_loader.load(model)
    model.freeze_backbone_weights()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    loss_fn = YoloLoss()

    train_loader, test_loader, train_eval_loader = get_loaders_nyu(mat_file_path=config.NYU_PATH)
    print(config.DEVICE)

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
        )

    # Scale anchors to each prediction scale
    scaled_anchors = (
            torch.tensor(config.ANCHORS)
            * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    best_map_till_now = -1

    print("Training started!")
    save_checkpoint(model, optimizer, filename=f"initial_ckpt.pth.tar")
    for epoch in range(config.NUM_EPOCHS):
        print(f"--------[EPOCH-{epoch + 1}]-------------")
        train_fn(train_loader, model, optimizer, loss_fn, scaled_anchors)

        if best_map_till_now < 0:
            save_checkpoint(model, optimizer, filename=f"initial_ckpt.pth.tar")

        if epoch % 10 == 0 and epoch > 0:
            print("On Test loader:")
            check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
            # Run model on test set and convert outputs to bounding boxes relative to image
            pred_boxes, true_boxes = get_evaluation_bboxes(
                test_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )
            # Compute mean average precision 
            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
            if mapval > best_map_till_now:
                print(
                    "Model performance improved from {:.2f}% to {:.2f}!".format(best_map_till_now * 100, mapval * 100))
                best_map_till_now = mapval
                save_checkpoint(model, optimizer, filename=f"yolov3_rgb_ckpt.pth.tar")
            print(f"MAP: {mapval.item()}")


if __name__ == "__main__":
    main()
