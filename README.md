
# DESCRIPTION

## NET 1

Faster-RCNN architecture applyied for living buildings detection on imges taken from drone. Exmples shown below.

![[ship2-5-3.jpg_pred.jpg]]
![[ship2-5-5.jpg_pred.jpg]]

## NET 2

Keypoint-RCNN architecture applyied for buildings corners detection on images. Exaples shown below.
![[TEO_1_00167_Ship_crop_0.jpg]]

![[TEO_1_00168_Ship_crop_0.jpg]]
![[TEO_1_00404_Ship_crop_2.jpg]]![[TEO_1_00505_Ship_crop_4.jpg]]![[TEO_1_01114_Ship_crop_1.jpg]]

## NET 3

Work not finished. Idea was to detect power pillars and its ground point.

## NET 4

Mask-RCNN applyied for detection and segmentation of roads. Examples below.

![[rodionovo-6-0_2.jpg]]
![[Pasted image 20240829115806.png]]

Also, Yolov8 model was trained for the same task. Examples are shown in the following section


## Deploy

Python script was written for deployment models to specific working software. Scripts starts with dialog window with configurations possibilities as shown below

![[Пример_GUI.png]]

Results are as follows:

#### Building's corners

![[Пример_точки.png]]

![[Пример_точки_2.png]]

![[Пример_точки_3.png]]

#### Roads

Note: Red dots show countours of roads detected and segmented via Mask-RCNN. Blue ones via Yolov8.

![[Пример_дороги.png]]

![[Пример_дороги_3.png]]

![[Пример_дороги_2.png]]