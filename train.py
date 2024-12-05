import json
import os

import torch
import torch.optim as optim
from tqdm import tqdm

import config
from load_weights import LoadYOLOWeights
from loss import YoloLoss
from model import YOLOv3
from utils.utils import (
    mean_average_precision,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders_nyu,
    seed_everything
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

    return mean_loss


def main():
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)

    # Load pretrained weights
    weight_loader = LoadYOLOWeights(config.CONFIG_FILE_PATH, config.WEIGHTS_FILE_PATH)
    weight_loader.load(model)

    # Freeze weights
    # model.freeze_backbone_weights()

    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    loss_fn = YoloLoss()

    train_mode = str(input("Enter train mode (rgb or hha or fusion): "))
    train_loader, test_loader, train_eval_loader = get_loaders_nyu(mat_file_path=config.NYU_PATH, train_mode=train_mode)

    if config.LOAD_MODEL:
        load_checkpoint(
            os.path.join(config.CHECKPOINT_DIR, "rgb_1_resume_ckpt.pth.tar"), model, optimizer
        )

    # Scale anchors to each prediction scale
    scaled_anchors = (
            torch.tensor(config.ANCHORS)
            * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    best_map_till_now = -1
    training_history = {"loss": [], "class_acc": [], "obj_acc": [], "noobj_acc": [], "map": [],
                        "ap_per_class": [], "predicted_boxes": []}
    train_log_file_name = str(input("Enter filename to store training logs: "))
    model_ckpt_name = str(input("Enter model checkpoint name (include .pth.tar): "))

    print("Training started!")

    # Define the subfolder for saving the training history
    subfolder = os.path.join("training_logs")
    os.makedirs(subfolder, exist_ok=True)

    for epoch in range(config.NUM_EPOCHS):
        print(f"--------[EPOCH-{epoch + 1}]-------------")
        epoch_loss = train_fn(train_loader, model, optimizer, loss_fn, scaled_anchors)
        training_history["loss"].append({f"Epoch-{epoch + 1}": epoch_loss})
        scheduler.step()

        if best_map_till_now < 0:
            save_checkpoint(model, optimizer, filename=f"initial_ckpt.pth.tar")

        print("On Test loader:")
        class_acc, obj_acc, noobj_acc = check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
        training_history["class_acc"].append({f"Epoch-{epoch + 1}": class_acc.item()})
        training_history["obj_acc"].append({f"Epoch-{epoch + 1}": obj_acc.item()})
        training_history["noobj_acc"].append({f"Epoch-{epoch + 1}": noobj_acc.item()})

        if epoch >= 10 or config.LOAD_MODEL:
            # Run model on test set and convert outputs to bounding boxes relative to image
            pred_boxes, true_boxes = get_evaluation_bboxes(
                test_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
                device=config.DEVICE
            )
            training_history["predicted_boxes"].append({f"Epoch-{epoch + 1}": len(pred_boxes)})
            # Compute mean average precision
            mapval, ap_per_class = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
            training_history["map"].append({f"Epoch-{epoch + 1}": mapval})
            training_history["ap_per_class"].append({f"Epoch-{epoch + 1}": ap_per_class})
            if mapval > best_map_till_now:
                print(
                    "Model performance improved from {:.2f}% to {:.2f}%!".format(best_map_till_now * 100, mapval * 100))
                best_map_till_now = mapval
                save_checkpoint(model, optimizer, filename=model_ckpt_name)
            print(f"MAP: {mapval}")
        else:
            # Append None to history if no evaluation is done
            training_history["map"].append({f"Epoch-{epoch + 1}": None})
            training_history["ap_per_class"].append({f"Epoch-{epoch + 1}": None})

        # Save the training history to a file
        history_file_path = os.path.join(subfolder, train_log_file_name)
        with open(history_file_path, "w") as f:
            json.dump(training_history, f)


if __name__ == "__main__":
    config.LOAD_MODEL = True
    seed_everything()
    main()
