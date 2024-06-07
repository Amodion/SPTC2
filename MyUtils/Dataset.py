import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from torchvision.io import read_image
from torchvision.ops import masks_to_boxes
import os
import json
import cv2
import copy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class NCornerDataset(Dataset):
    def __init__(self, root, N, transform=None, demo=False, corners=None):
        self.root = root
        self.transform = transform
        self.demo = demo # Use demo=True if you need transformed and original images (for example, for visualization purposes)
        self.N = N
        self.corners = corners
        with open(os.path.join(root, 'annotation.json'), 'r') as file:
            annotations = json.load(file)
        if self.corners is not None:
            annotations = self.choose_corners(annotations, self.corners)
            self.N = max(self.corners)
        self.annotations = annotations

    def __getitem__(self, idx):
        if not 'annotations' in self.annotations[idx]:
            raise Exception('Wrong convert. No annotations key')
            
        annotations_result = self.annotations[idx]['annotations'][0]['result'] # CHECK IF KEYS EXIST!!!!

        output = self.export_annotations(annotations_result, idx)

        if not all([out is not None for out in output]):
            raise Exception(f'Wrong convert. Result error.\nTask ID: {self.annotations[idx]["inner_id"]}')

        keypoints_original, bboxes_original = output

        img_path = self.get_image(self.annotations[idx])

        if img_path is None:
            raise Exception('Wrong convert. Image error')

        img_path = os.path.join(self.root, 'images', img_path)

        img_original = cv2.imread(img_path)
        img_h, img_w = img_original.shape[0], img_original.shape[1]
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

        kps = copy.deepcopy(keypoints_original)
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
                print(f'Image width and height: {img_w}, {img_h}\n\n\nImage path: {img_path}\n\nTask ID: {self.annotations[idx]["inner_id"]}\n{keypoints_original=}')
                raise e
                
            img = transformed['image']
            h, w = img.shape[0], img.shape[1]
            bboxes = transformed['bboxes']
    
            #bboxes = [[max(2, bboxes[0][0]), max(2, bboxes[0][1]), min(w - 2, bboxes[0][2]), min(h - 2, bboxes[0][3])]]

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

        assert len(target['keypoints']) != 0, f'No keypoints at image {self.get_image(self.annotations[idx])}, ID: {self.annotations[idx]["inner_id"]}'
        assert len(target_original['keypoints']) != 0, f'No keypoints at image {self.get_image(self.annotations[idx])}, ID: {self.annotations[idx]["inner_id"]}'
        if self.demo:
            return img, target, img_original, target_original
        else:
            return img, target
    
    def __len__(self):
        return len(self.annotations)

    def export_annotations(self, result: list, idx):
        keypoints = [[]]
        bbox = []
        for ann in result:
            if 'original_width' not in ann or 'original_height' not in ann:
                return None
            if ann['type'] == 'keypointlabels':
                x = ann['value']['x'] / 100 * ann['original_width']
                y = ann['value']['y'] / 100 * ann['original_height']
                keypoints[0].append([int(x), int(y), 1])
            else:
                x, y = ann['value']['x'] / 100 * ann['original_width'], ann['value']['y'] / 100 * ann['original_height']
                w, h = ann['value']['width'] / 100 * ann['original_width'], ann['value']['height'] / 100 * ann['original_height']
                bbox.append([int(x), int(y), int(x + w), int(y + h)])
        return keypoints, bbox

    def choose_corners(self, annotations, corners):
        return [annotation for annotation in annotations if (len(annotation['annotations'][0]['result']) - 1) in corners]

    def get_image(self, d: dict):
        if 'data' in d:
            return d['data']['img'].split('-')[-1]
    
    def sort_key(self, d: dict):
        return d['inner_id']
    
    @property
    def explore(self):
        nums_of_points = []
        max_points = 0
        buildings = {'One corner': 0, 'Two corner': 0, 'Three corner': 0, 'Many corner': 0}
        annotations_without_buildings = []
        annotations_with_extra_buildings = []
        annotations_with_errors = []

        print(f'Lenght of dataset is {self.__len__()}')

        for idx, annotation in enumerate(self.annotations):
            num_p = 0
            num_b = 0
            building_exist = False
            result = annotation['annotations'][0]['result']
            
            for res in result:
                if res['type']=='keypointlabels':
                    num_p += 1
                if res['type']=='rectanglelabels':
                    building_exist = True
                    num_b += 1

                for d in res:
                    if not isinstance(d, dict):
                        annotations_with_errors.append((self.get_image(annotation), annotation['inner_id']))
                    elif not ((0.0 <= d['value']['x'] <= 100.0) or (0.0 <= d['value']['y'] <= 100.0)):
                        annotations_with_errors.append((self.get_image(annotation), annotation['inner_id']))
            
            max_points = max(num_p, max_points)
            if num_p == 0:
                print(f'Zero points at id {annotation["inner_id"]}')
            nums_of_points.append(num_p)
            if num_b > 1:
                annotations_with_extra_buildings.append((self.get_image(annotation), annotation['inner_id']))
            if not building_exist:
                annotations_without_buildings.append((self.get_image(annotation), annotation['inner_id']))

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

        print('\nAnnotations with extra buiuldings:\n')
        print(*annotations_with_extra_buildings)

        print('\nAnnotations with errors:\n')
        print(*set(annotations_with_extra_buildings))


class PillarsDataset(Dataset):
    def __init__(self, root, transform=None, demo=False):
        self.root = root
        self.transform = transform
        self.demo = demo # Use demo=True if you need transformed and original images (for example, for visualization purposes)
        self.max_corners = None
        self.inner_id = None
        with open(os.path.join(root, 'annotation.json'), 'r') as file:
            annotations = json.load(file)
        self.annotations = annotations
        assert len(self.annotations) != 0, 'Annotations file empty!' 

    def __getitem__(self, idx):
        assert 'inner_id' in self.annotations[idx], f'No inner id at item {idx}'
        self.inner_id = self.annotations[idx]['inner_id']
        
        assert 'annotations' in self.annotations[idx], f'No annotations key. {self.inner_id=}'
        assert 'result' in self.annotations[idx]['annotations'][0], f'No result in annotations. {self.inner_id=}'
            
        result = self.annotations[idx]['annotations'][0]['result']

        output = self.export_annotations(result)

        if not all([out is not None for out in output]):
            raise Exception(f'Wrong convert. Result error.\nTask ID: {self.inner_id}')

        keypoints_original, bboxes_original = output

        img_path = self.get_image(self.annotations[idx])

        if img_path is None:
            raise Exception('Wrong convert. Image error')

        img_path = os.path.join(self.root, 'images', img_path)

        img_original = cv2.imread(img_path)
        img_h, img_w = img_original.shape[0], img_original.shape[1]
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

        kps = copy.deepcopy(keypoints_original)

        assert self.max_corners is not None, 'Max number of corners per pillar is unknown! Please, use dataset.explore first to find it.'
            
        for pillar in keypoints_original:
            while len(pillar) < self.max_corners:
               pillar.append([0, 0, 0])      

        ###### UNCOMMENT FOR EXTRA KPS CYCLE COPY TRUE KPS ###########
        #for i in range(self.N):
        #    keypoints_original[0][i] = kps[0][i % len(kps[0])]

        bboxes_labels_original = ['Pillar' for _ in bboxes_original]            

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
                print(f'Image width and height: {img_w}, {img_h}\n\n\nImage path: {img_path}\n\nTask ID: {self.inner_id}\n{keypoints_original=}\n{bboxes_original=}\nKps len = {len(keypoints_original)}\nBbox len = {len(bboxes_original)}\n{keypoints_original_flattened=}')
                raise e
                
            img = transformed['image']
            h, w = img.shape[0], img.shape[1]
            bboxes = transformed['bboxes']
    
           # bboxes = [[max(2, bboxes[0][0]), max(2, bboxes[0][1]), min(w - 2, bboxes[0][2]), min(h - 2, bboxes[0][3])]]

            # Unflattening list transformed['keypoints']
            # For example, if we have the following list of keypoints for three objects (each object has two keypoints):
            # [obj1_kp1, obj1_kp2, obj2_kp1, obj2_kp2, obj3_kp1, obj3_kp2], where each keypoint is in [x, y]-format
            # Then we need to convert it to the following list:
            # [[obj1_kp1, obj1_kp2], [obj2_kp1, obj2_kp2], [obj3_kp1, obj3_kp2]]
            
            keypoints_transformed_unflattened = []
            for idx in range(0, len(transformed['keypoints']) - self.max_corners + 1, self.max_corners):
                keypoints_transformed_unflattened.append(transformed['keypoints'][idx:idx+self.max_corners])
            
            #print(keypoints_transformed_unflattened)
           
            # Converting transformed keypoints from [x, y]-format to [x,y,visibility]-format by appending original visibilities to transformed coordinates of keypoints
            keypoints = []
            for o_idx, obj in enumerate(keypoints_transformed_unflattened): # Iterating over objects
                obj_keypoints = []
                for k_idx, kp in enumerate(obj): # Iterating over keypoints in each object
                    # kp - coordinates of keypoint
                    # keypoints_original[o_idx][k_idx][2] - original visibility of keypoint
                    obj_keypoints.append(list(kp) + [keypoints_original[o_idx][k_idx][2]])
                keypoints.append(obj_keypoints)
            
            for pillar in keypoints:
                while len(pillar) < self.max_corners:
                    pillar.append([0, 0, 0]) 
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

        assert len(target['keypoints']) != 0, f'No keypoints at image {self.get_image(self.annotations[idx])}, {self.inner_id=}'
        assert len(target_original['keypoints']) != 0, f'No keypoints at image {self.get_image(self.annotations[idx])}, {self.inner_id=}'
        if self.demo:
            return img, target, img_original, target_original
        else:
            return img, target
    
    def __len__(self):
        return len(self.annotations)

    def export_annotations(self, result: list):
        keypoints = []
        bbox = []
        pillars = [object for object in result if object['type']=='rectanglelabels']
        corners = [object for object in result if object['type']=='keypointlabels']

        for pillar in pillars:
            original_w = pillar['original_width']
            original_h = pillar['original_height']
            value = pillar['value']
            
            x, y = value['x'] / 100 * original_w, value['y'] / 100 * original_h
            w, h = value['width'] / 100 * original_w, value['height'] / 100 * original_h
            bbox.append([int(x), int(y), int(x + w), int(y + h)])
            
            id = pillar['id']
            pillar_corners = []
            for corner in corners:
                if corner['parentID'] == id:
                    value = corner['value']

                    x, y = value['x'] / 100 * original_w, value['y'] / 100 * original_h
                    pillar_corners.append([int(x), int(y), 1])

            assert len(pillar_corners) != 0, f'Wrong convert. No keypoints at task with {self.inner_id=}'
            keypoints.append(pillar_corners)

        return keypoints, bbox

    def get_image(self, d: dict):
        assert 'data' in d, f'No image data. {self.inner_id=}'
        assert 'image' in d['data'], f'No image data. {self.inner_id=}'
        return d['data']['image'].split('-')[-1]

    def get_max_corners(self):
        assert self.max_corners is not None, 'Max number of corners per pillar is unknown! Please, use dataset.explore first to find it.'
        return self.max_corners
    
    @property
    def explore(self):
        from collections import defaultdict
        
        nums_of_pillars = []
        corners_against_pillars = defaultdict(int)
        annotations_without_pillars = []
        annotations_with_cornerless_pillars = []
        annotations_with_errors = []

        print(f'Lenght of dataset is: {self.__len__()}')

        for idx, annotation in enumerate(self.annotations):

            assert 'inner_id' in annotation, f'No inner id at item {idx}'
            inner_id = annotation['inner_id']
        
            if 'annotations' not in annotation: 
                annotations_with_errors.append(inner_id)
                continue
            if 'result' not in annotation['annotations'][0]: 
                annotations_with_errors.append(inner_id)
                continue
                
            pillars_number = 0
            result = annotation['annotations'][0]['result']

            if len(result) == 0:
                annotations_with_errors.append(inner_id)
                continue
            
            pillars = [object for object in result if object['type']=='rectanglelabels']
            corners = [object for object in result if object['type']=='keypointlabels']

            num_pillars = len(pillars)

            if num_pillars == 0:
                annotations_without_pillars.append(inner_id)
            nums_of_pillars.append(num_pillars)

            for pillar in pillars:
                id = pillar['id']
                try:
                    pillar_corners = [corner for corner in corners if corner['parentID']==id]
                except:
                    annotations_with_errors.append(inner_id)
                num_corners = len(pillar_corners)
                    
                if num_corners == 0:
                    annotations_with_cornerless_pillars.append(inner_id)
                corners_against_pillars[num_corners] += 1

        sns.set_theme()

        self.max_corners = max(list(corners_against_pillars.keys()))

        print(f'Maximum number of corners per pillar is: {self.max_corners}')

        NPillars_hist = sns.histplot(nums_of_pillars, discrete=True, kde=False, stat='percent')
        NPillars_hist.set_title('Number of pillars on pictures')
        NPillars_hist.set_xlabel('Number of pillars per picture')
        NPillars_hist.set_ylabel('Percent of images')
        NPillars_hist.set(xticks=np.arange(1,max(nums_of_pillars)+1,1))
        for container in NPillars_hist.containers:
            NPillars_hist.bar_label(container, fontsize=10)

        plt.show()

        pillars_corners = sns.barplot(x=list(corners_against_pillars.keys()), y=list(corners_against_pillars.values()))
        pillars_corners.bar_label(pillars_corners.containers[0], fontsize=10)
        pillars_corners.set_title('Pillars with different number of visible corners')
        pillars_corners.set_ylabel('Number of Pillars')
        plt.show()

        if annotations_without_pillars:
            print('Annotations without pillars:\n')
            print(*annotations_without_pillars)

        if annotations_with_cornerless_pillars:
            print('\nAnnotations with cornerless pilars:\n')
            print(*annotations_with_cornerless_pillars)

        if annotations_with_errors:
            print('\nAnnotations with errors:\n')
            print(*set(annotations_with_errors))

class RoadsDataset(Dataset):
    def __init__(self, root, class_index=None, transform=None, demo=False, remove_single_class=None):
        self.root = root
        self.transform = transform
        self.demo = demo # Use demo=True if you need transformed and original images (for example, for visualization purposes)
        self.annotations = pd.read_json(self.root + 'annotations.json', orient='records')
        
        if remove_single_class:
            indxs = []
            for row in self.annotations.iterrows():
                labels = []
                for data in row[1].tag:
                    labels.extend(data['brushlabels'])
                if remove_single_class not in labels:
                    indxs.append(row[0])
            self.annotations = self.annotations.iloc[indxs]
            self.annotations.reset_index(drop=True, inplace=True)
            
        self.class_index = class_index
        assert self.class_index is not None, 'Class index is not specified!'
        assert len(self.annotations) != 0, 'Annotations file empty!'

    def __getitem__(self, idx):
        annotation = self.annotations.iloc[idx]
        image_path = annotation['image_path']
        #image_path = self.root + 'images/' +'-'.join(image_path.split('-')[1:])
        id = annotation['id']

        masks_original = [np.load(mask) for mask in annotation['masks']]
        masks_original = [np.where(mask > 0, 1, mask) for mask in masks_original]
        masks_original = np.array(masks_original)

        assert len(masks_original) != 0, f'No masks loaded for task {id}'

        objects_original = [mask.split('-')[-2] for mask in annotation['masks']]
        
        boxes_original = masks_to_boxes(torch.tensor(masks_original, dtype=torch.uint8)).detach().cpu().numpy()

        image_original = cv2.imread(image_path)
        image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image=image_original, bboxes=boxes_original, masks=masks_original, class_labels=objects_original)
            
            image = transformed['image']
            
            masks = np.array([mask for mask in transformed['masks'] if np.any(mask)]) # remove empty masks
            
            boxes = masks_to_boxes(torch.tensor(masks, dtype=torch.uint8)).detach().cpu().numpy()
                
            objects = [label for label, mask in zip(transformed['class_labels'], transformed['masks']) if np.any(mask)]
        else:
            image, boxes, masks, objects = image_original, boxes_original, masks_original, objects_original

        image = F.to_tensor(image)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        masks = torch.tensor(masks, dtype=torch.uint8)
        labels = torch.tensor([self.class_index[obj] for obj in objects], dtype=torch.int64)

        assert len(masks) != 0, f'No masks loaded for task {id}'

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["task_id"] = torch.tensor([id])
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros(len(boxes), dtype=torch.int64)
        target["masks"] = masks

        return image, target

    def __len__(self):
        return len(self.annotations)



























