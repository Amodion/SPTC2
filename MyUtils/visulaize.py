import matplotlib.pyplot as plt
import seaborn as sns
import cv2

def visualize(image, bboxes, keypoints, image_original=None, bboxes_original=None, keypoints_original=None, save_path=None, bld_score=None, kps_scores=None):
    fontsize = 18
    sns.set_style("white")

    thickness = 1 + 2*int(image.copy().shape[0] > 700) + 2*int(image.copy().shape[1] > 700)
    
    for num, bbox in enumerate(bboxes):
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        image = cv2.rectangle(image.copy(), start_point, end_point, (0,255,0), thickness)
        if bld_score:
            score = round(bld_score[num], 2)
            image = cv2.putText(image.copy(), ' Building' + str(score), start_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    
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