import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
import os
import json
import cv2
import copy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class NCornerDataset(Dataset):
    def __init__(self, root, transform=None, demo=False, N=9):                
        self.root = root
        self.transform = transform
        self.demo = demo # Use demo=True if you need transformed and original images (for example, for visualization purposes)
        self.imgs_files = sorted(os.listdir(os.path.join(root, "images")))
        self.N = N
        with open(os.path.join(root, 'annotation.json'), 'r') as file:
            annotations = json.load(file)
        self.annotations = sorted(annotations, key=self.sort_key)
        assert len(self.imgs_files)==len(self.annotations), f'Количество изображений ({len(self.imgs_files)}) не совпадает с количеством аннотаций ({self.annotations})!'

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs_files[idx])
        annotations_result = self.annotations[idx]['annotations'][0]['result']

        img_original = cv2.imread(img_path)
        img_h, img_w = img_original.shape[0], img_original.shape[1]
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

        keypoints_original, bboxes_original = self.export_annotations(annotations_result)
        
        kps = copy.deepcopy(keypoints_original)
        #print(kps)
        while len(keypoints_original[0]) < self.N:
            keypoints_original[0].append([0, 0, 0])      

        ###### UNCOMMENT FOR EXTRA KPS CYCLE COPY TRUE KPS ###########
        #for i in range(self.N):
        #    keypoints_original[0][i] = kps[0][i % len(kps[0])]

        bboxes_labels_original = ['Building']            

        if self.transform:   
            # Converting keypoints from [x,y,visibility]-format to [x, y]-format + Flattening nested list of keypoints            
            # For example, if we have the following list of keypoints for three objects (each object has two keypoints):
            # [[obj1_kp1, obj1_kp2], [obj2_kp1, obj2_kp2], [obj3_kp1, obj3_kp2]], where each keypoint is in [x, y]-format            
            # Then we need to convert it to the following list:
            # [obj1_kp1, obj1_kp2, obj2_kp1, obj2_kp2, obj3_kp1, obj3_kp2]
            keypoints_original_flattened = [el[0:2] for kp in keypoints_original for el in kp]
            try:
                transformed = self.transform(image=img_original, bboxes=bboxes_original, bboxes_labels=bboxes_labels_original, keypoints=keypoints_original_flattened)
            except Exception as e:
                print(f'Lines was:\n{lines}\n\n\nImage width and height: {img_w}, {img_h}\n\n\nImage path: {img_path}')
                raise e
                
            img = transformed['image']
            h, w = img.shape[0], img.shape[1]
            bboxes = transformed['bboxes']
    
            bboxes = [[max(2, bboxes[0][0]), max(2, bboxes[0][1]), min(w - 2, bboxes[0][2]), min(h - 2, bboxes[0][3])]]

            # Unflattening list transformed['keypoints']
            # For example, if we have the following list of keypoints for three objects (each object has two keypoints):
            # [obj1_kp1, obj1_kp2, obj2_kp1, obj2_kp2, obj3_kp1, obj3_kp2], where each keypoint is in [x, y]-format
            # Then we need to convert it to the following list:
            # [[obj1_kp1, obj1_kp2], [obj2_kp1, obj2_kp2], [obj3_kp1, obj3_kp2]]
            
            keypoints_transformed_unflattened = [[]]
            for kp in transformed['keypoints']:
                keypoints_transformed_unflattened[0].append(list(kp))
           
            # Converting transformed keypoints from [x, y]-format to [x,y,visibility]-format by appending original visibilities to transformed coordinates of keypoints
            keypoints = []
            for o_idx, obj in enumerate(keypoints_transformed_unflattened): # Iterating over objects
                obj_keypoints = []
                for k_idx, kp in enumerate(obj): # Iterating over keypoints in each object
                    # kp - coordinates of keypoint
                    # keypoints_original[o_idx][k_idx][2] - original visibility of keypoint
                    obj_keypoints.append(kp + [keypoints_original[o_idx][k_idx][2]])
                keypoints.append(obj_keypoints)
            while len(keypoints[0]) < self.N:
                keypoints[0].append([0, 0, 0]) 
        else:
            try:
                img, bboxes, keypoints = img_original, bboxes_original, keypoints_original        
            except Exception as e:
                print(img_path)
                raise e
        # Convert everything into a torch tensor
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)       
        target = {}
        target["boxes"] = bboxes
        target["labels"] = torch.as_tensor([1 for _ in bboxes], dtype=torch.int64) # all objects are buildings
        target["image_id"] = torch.tensor([idx])
        target["area"] = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        target["iscrowd"] = torch.zeros(len(bboxes), dtype=torch.int64)
        target["keypoints"] = torch.as_tensor(keypoints, dtype=torch.float32)        
        img = F.to_tensor(img)
        
        bboxes_original = torch.as_tensor(bboxes_original, dtype=torch.float32)
        target_original = {}
        target_original["boxes"] = bboxes_original
        target_original["labels"] = torch.as_tensor([1 for _ in bboxes_original], dtype=torch.int64)
        target_original["image_id"] = torch.tensor([idx])
        target_original["area"] = (bboxes_original[:, 3] - bboxes_original[:, 1]) * (bboxes_original[:, 2] - bboxes_original[:, 0])
        target_original["iscrowd"] = torch.zeros(len(bboxes_original), dtype=torch.int64)
        target_original["keypoints"] = torch.as_tensor(keypoints_original, dtype=torch.float32)        
        img_original = F.to_tensor(img_original)

        if self.demo:
            return img, target, img_original, target_original
        else:
            return img, target
    
    def __len__(self):
        return len(self.imgs_files)

    def export_annotations(self, result: list):
        keypoints = [[]]
        bbox = []
        for ann in result:
            if ann['type'] == 'keypointlabels':
                x = ann['value']['x'] / 100 * ann['original_width']
                y = ann['value']['y'] / 100 * ann['original_height']
                keypoints[0].append([x, y, 1])
            else:
                x, y = ann['value']['x'] / 100 * ann['original_width'], ann['value']['y'] / 100 * ann['original_height']
                w, h = ann['value']['width'] / 100 * ann['original_width'], ann['value']['height'] / 100 * ann['original_height']
                bbox.append([x, y, x + w, y + h])
        return keypoints, bbox

    def sort_key(self, d: dict):
        return d['data']['img'].split('-')[-1]
    
    @property
    def explore(self):
        nums_of_points = []
        max_points = 0
        buildings = {'One corner': 0, 'Two corner': 0, 'Three corner': 0, 'Many corner': 0}
        annotations_without_buildings = []

        for idx, annotation in enumerate(self.annotations):
            num_p = 0
            building_exist = False
            result = annotation['annotations'][0]['result']
            
            for res in result:
                if res['type']=='keypointlabels':
                    num_p += 1
                if res['type']=='rectanglelabels':
                    building_exist = True
            
            max_points = max(num_p, max_points)
            nums_of_points.append(num_p)
            if not building_exist:
                annotations_without_buildings.append(self.imgs_files[idx])

            if num_p == 1:
                buildings['One corner'] += 1
            elif num_p == 2:
                buildings['Two corner'] += 1
            elif num_p == 3:
                buildings['Three corner'] += 1
            else:
                buildings['Many corner'] += 1

        sns.set_theme()

        NPoints_hist = sns.histplot(nums_of_points, discrete=True, kde=False, stat='percent')
        NPoints_hist.set_title('Number of points on pictures')
        NPoints_hist.set_xlabel('Number of points per picture')
        NPoints_hist.set_ylabel('Percent of images')
        NPoints_hist.set(xticks=np.arange(1,max_points+1,1))
        for container in NPoints_hist.containers:
            NPoints_hist.bar_label(container, fontsize=10)

        plt.show()

        buildings_corners = sns.barplot(x=list(buildings.keys()), y=list(buildings.values()))
        buildings_corners.bar_label(buildings_corners.containers[0], fontsize=10)
        buildings_corners.set_title('Buildings with different number of visible corners')
        buildings_corners.set_ylabel('Number of Buildings')
        plt.show()

        print('Annotations without buildings:\n')
        print(*annotations_without_buildings)
































