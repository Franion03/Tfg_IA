import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
from ponit_selector import PointSelector
import cv2

def show_mask(masker, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = masker.shape[-2:]
    mask_image = masker.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

class Segmenter:
    def __init__(self, model_type, checkpoint):
        absolute_path = os.path.dirname(__file__)
        full_path = os.path.join(absolute_path, checkpoint)
        # initialize the model with the given type and checkpoint
        self.model = sam_model_registry[model_type](checkpoint=full_path)
        # device = "cuda"
        # self.model.to(device=device)
        # create a predictor object
        self.predictor = SamPredictor(self.model)
        self.points = []
        self.image = None
        self.input_label = np.array([1])

    def segment(self, image, prompts):
        # set the image for the predictor
        self.predictor.set_image(image)
        # get the masks from the given prompts
        masks, _, _ = self.predictor.predict(prompts)
        # return the masks
        return masks

    def segmentRoi(self, image):
        selector = PointSelector(image, "Selecciona los puntos")
        selector.show_and_wait()
        roi = selector.points
        # set the image for the predictor
        self.predictor.set_image(image)
        point = np.array(roi)
        self.input_label = np.array([1 for i in range(len(point))])
        # get the mask from the point prompt
        self.mask, self.scores, self.logits = self.predictor.predict(point_coords=point,point_labels=self.input_label,multimask_output=True,)
        # show the mask
        for i in range(len(self.mask)):
            fig = plt.figure(figsize=(10, 10))
            plt.imshow(image)
            show_mask(self.mask[i], plt.gca())
            show_points(point, self.input_label, plt.gca())
            plt.axis('off')
            plt.show()
        print ("¿Dime un número?")
        numero = input()
        numero = int(numero)
        return self.mask[numero]
    
    
