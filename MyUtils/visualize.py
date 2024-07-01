import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from torchvision.transforms import functional as F
import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import numpy as np
from fontTools import ttLib

def get_font_path(font_name):
    font_collection = ttLib.TTCollection(font_name)
    font_path = font_collection.fonts[0].path
    return font_path

def visualize(image, bboxes, keypoints, image_original=None, bboxes_original=None, keypoints_original=None, save_path=None, bld_score=None, kps_scores=None, obj_type='Building'):
    fontsize = 18
    sns.set_style("white")

    thickness = 1 + 2*int(image.copy().shape[0] > 700) + 2*int(image.copy().shape[1] > 700)
    
    for num, bbox in enumerate(bboxes):
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        image = cv2.rectangle(image.copy(), start_point, end_point, (0,255,0), thickness)
        if bld_score:
            score = round(bld_score[num], 2)
            image = cv2.putText(image.copy(), ' ' + obj_type + str(score), start_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    
    radius = 2 + 2*int(image.copy().shape[0] > 700) + 2*int(image.copy().shape[1] > 700)
#    cv2.imwrite('test.jpg', image)
    colors = {0: (0, 0, 255), 1: (255, 0, 0)}
    
    for kps in keypoints:
        for idx, kp in enumerate(kps):
            image = cv2.circle(image.copy(), tuple(kp[:2]), radius, colors[int(kp[2])], -1)
            if kps_scores is not None:
                score = round(kps_scores[idx], 2)
                image = cv2.putText(image.copy(), ' ' + str(score), tuple(kp[:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 1, cv2.LINE_AA)

    if image_original is None and keypoints_original is None:
        plt.figure(figsize=(15,15))
        plt.imshow(image)
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    else:
        for bbox in bboxes_original:
            start_point = (bbox[0], bbox[1])
            end_point = (bbox[2], bbox[3])
            image_original = cv2.rectangle(image_original.copy(), start_point, end_point, (0,255,0), thickness)
        
        for kps in keypoints_original:
            for idx, kp in enumerate(kps):
                image_original = cv2.circle(image_original.copy(), tuple(kp[:2]), radius, colors[int(kp[2])], -1)
#                image_original = cv2.putText(image_original, " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3, cv2.LINE_AA)

        f, ax = plt.subplots(1, 2, figsize=(20, 20))

        ax[0].imshow(image_original)
        ax[0].set_title('Original image', fontsize=fontsize)

        ax[1].imshow(image)
        ax[1].set_title('Transformed image', fontsize=fontsize)

        if save_path:
            cv2.imwrite(save_path[:-4] + '_tranformed.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(save_path, cv2.cvtColor(image_original, cv2.COLOR_RGB2BGR))


def visualise_tensor(image, final_target, path):
    image = image.detach().cpu().numpy()
    image = image.transpose(1,2,0) * 255
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    keypoints = final_target['keypoints'].detach().cpu().numpy()
    scores = final_target['keypoints_scores'].detach().cpu().numpy()
    boxes = final_target['boxes'].detach().cpu().numpy()

    keypoints = [list(map(int, keypoint)) for keypoint in keypoints]
    scores = [round(score, 2) for score in scores]
    boxes = [list(map(int, box)) for box in boxes]

    for num, box in enumerate(boxes):
        start_point = (box[0], box[1])
        end_point = (box[2], box[3])
        image = cv2.rectangle(image.copy(), start_point, end_point, (0,255,0), 1)
        image = cv2.putText(image.copy(), ' Building', start_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

    radius = 2 + 2*int(image.shape[0] > 700) + 2*int(image.shape[1] > 700)
    colors = {0: (255, 0, 0), 1: (0, 0, 255)} # FOR SOME REASON POINTS APPEARS BLUE
    
    for point, score in zip(keypoints, scores):
        image = cv2.circle(image.copy(), tuple(point[:2]), radius, colors[int(point[2])], -1)
        image = cv2.putText(image.copy(), ' ' + str(score), tuple(point[:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA)

    cv2.imwrite(path, image)

def visualize_masks(image: torch.Tensor, target: dict = None, inv_classes: dict = None, alpha: float = 0.7, obj_colors: dict = None, show: bool = False, save_path: str = None, mask_threshold: float = 0.5):
    '''
    Function to draw image with segmentation mask and boxes from model prediction output either from dataset. Also it can draw only image if only image given.
    Expects an image float tensor in [0.0, 1.0] range and target dict with "masks", "boxes" and "labels" fields. 
    inv_classes is a dict where keys is integers started from 0 and values is classes labels.
    obj_colors is a dict where keys is classes labels and values is tuples of int for colors in format (R, G, B).
    alpha is a transparacy value in range [0.0, 1.0] for masks on images where 0 is fully invisible.
    If show is given as True output image will be shown in notebook, otherwise save_path should be specified to write image on disk.
    mask_threshold is a value do determine is pixel of mask belongs to the object or not as mask's pixels is given as float in range [0.0, 1.0]
    '''
    font_path = "C:/Users/User/Petr/Net_4/Dataset/TimesNewRomanRegular.ttf"
    
    image = (image * 255).to(dtype=torch.uint8)
    if target:

        if len(target['masks'].shape) == 4:
            masks = target['masks'] > mask_threshold
            masks = masks.squeeze(1)
        else:
            masks = target['masks'].to(dtype=torch.bool)
        
        boxes = target['boxes']

        objects = [inv_classes[label] for label in target['labels'].detach().cpu().numpy()]

        if not len(masks):
            masked_boxed_image = image
        else:
            if obj_colors:
                masked_image = draw_segmentation_masks(image=image, masks=masks, alpha=alpha, colors=[obj_colors[obj] for obj in objects])
            else:
                masked_image = draw_segmentation_masks(image=image, masks=masks, alpha=alpha)
            masked_boxed_image = draw_bounding_boxes(masked_image, boxes, colors="red", width=2, labels=objects, font=font_path, font_size=75)

        masked_boxed_image = F.to_pil_image(masked_boxed_image)
    else:
         masked_boxed_image = F.to_pil_image(image)

    if show:
        plt.figure(figsize=(20,15))
        plt.imshow(np.asarray(masked_boxed_image))
    else:
        assert save_path is not None, 'Specify saving path!'
        cv2.imwrite(save_path, cv2.cvtColor(np.asarray(masked_boxed_image), cv2.COLOR_RGB2BGR))
        