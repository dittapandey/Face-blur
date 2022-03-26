import config
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from matplotlib import image
import numpy as np
import random
import torch
import os
import cv2


image = cv2.imread('AdityaSign_11zon.jpg')
orig = image

def anonymize_face(image, factor=2.0):
    (h,w) = image.shape[:2]
    kW = int(w/factor)
    kH = int(h/factor)
    if kW%2 == 0:
        kW -=1
    if kH % 2 == 0:
        kH-=1
    return cv2.GaussianBlur(image, (kW,kH), 0)



def plot_image(image, boxes):
    cmap = plt.get_cmap("tab20b")
    
    im = np.array(image)
    height, width, _ = im.shape

    fig, ax = plt.subplots(1)
    ax.imshow(im)

    for box in boxes:
        assert len(box) ==6, "box should contain class_pred, confidence, x, y, width, height"
        class_pred = box[0] 
        box = box[2:]
        upper_left_x = box[0] - box[2]/2
        upper_left_y = box[1] - box[3]/2
        start_point = (int(upper_left_x*width), int(upper_left_y*height))
        end_point = (int((upper_left_x+box[2])*width),int((upper_left_y+box[3])*height))
        image = cv2.rectangle(image, start_point, end_point,(255,0,0),1)

        face = image[int(upper_left_y*height): int((upper_left_y+box[3])*height),int(upper_left_x*width): int((upper_left_x+box[2])*width)]
        face = anonymize_face(face)
        image[int(upper_left_y*height): int((upper_left_y+box[3])*height),int(upper_left_x*width): int((upper_left_x+box[2])*width)] = face
    return image


def real_time_data():
    vid = cv2.VideoCapture(0)
    while(True):
        ret, frame = vid.read()
        # code for getting the boxes from the model goes herer
        boxes = [[0.1, 0.2, 0.5, 0.6, 0.2, 0.4]]

        # the code will end here or somewhere
        
        frame = plot_image(frame, boxes)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    vid.release()
    cv2.destroyAllWindows()

# def video_data(filepath):



def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Save checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> load checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    # this following is in order to load the learning rat4e of the old checkpoint 
    # this will require rigorous debugging
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

        
boxes = [[0.1, 0.2, 0.5, 0.6, 0.2, 0.4]]

# plot_image(image, boxes)
real_time_data()

