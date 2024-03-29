# -*- coding: utf-8 -*-
"""Train.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UVqHkIObxPsO0rfoqNIK93XkasTkOd8J
"""

import torch
import torch.optim as optim
import torch.nn as nn

from utils import (iou_width_height, intersection_over_union, non_max_suppression, mean_average_precision, get_evaluation_bboxes, cells_to_bboxes ,
                   save_checkpoint, load_checkpoint, check_class_accuracy, get_loaders, plot_couple_examples)
from loss import YOLOLoss
torch.backends.cudnn.benchmark = True

from model import YOLO
from tqdm import tqdm

def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
  losses=[]
  loader=tqdm(train_loader, leave=True)
  for i , (x, y) in enumerate(train_loader):
    x=x.to(config.DEVICE)
    y1, y2, y3=(y[0].to(config.DEVICE), y[1].to(config.DEVICE), y[2].to(config.DEVICE))
    with torch.cuda.amp.autocast():
      Y1, Y2, Y3=model.test(x)
      loss_1=loss_fn(y1, Y1, scaled_anchors[0]
      loss_2=loss_fn(y2, Y2, scaled_anchors[1])
      loss_3=loss_fn(y3, Y3, scaled_anchors[2])
      loss=loss_1+loss_2+loss_3
    losses.append(loss.item())
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    mean_loss = sum(losses) / len(losses)
    loop.set_postfix(loss=mean_loss)
  scheduler1=lr_scheduler.ReduceLROnPlateau()
  scheduler1.step()
  # scheduler2.step()

def main():
  model=YOLO(num_classes=1).to(config.DEVICE)
  optimizer=optim.Adam()
  loss_fn=YOLOLoss
  scaler=torch.cuda.amp.GradScaler()
  train_loader, _, _=get_loader("")

  if config.LOAD_MODEL:
    load_checkpoint(config.CHECKPOINT_FILE, model, optimizer, config.LEARNIG_RATE)
  scaled_anchors=torch.tensor(config.ANCHORS)*torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2).to(config.DEVICE)
  for epoch in range(config.EPOCH):
    train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)
    if epoch%5==0:
      pred_boxes, true_boxes = get_evaluation_bboxes(
                test_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )
      mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
      print(f"MAP: {mapval.item()}")
    model.train()

    if __name__ == "__main__":
    main()

# print(torch.tensor([[(1,2)], [(5,6)],[(11, 12)]]).shape)

