
# DESCRIPTION

## NET 1

Faster-RCNN architecture applyied for living buildings detection on imges taken from drone. Exmples shown below.

![[Net_1.jpg]]
![[Net_1_1.jpg]]

## NET 2

Keypoint-RCNN architecture applyied for buildings corners detection on images. Exaples shown below.
![[Net_2.jpg]]

![[Net_2_1.jpg]]
![[Net_2_2.jpg]]![[Net_2_3.jpg]]![[Net_2_4.jpg]]

## NET 3

Work not finished. Idea was to detect power pillars and its ground point.

## NET 4

Mask-RCNN applyied for detection and segmentation of roads. Examples below.

![[Net_4.jpg]]
![[Net_4_1.png]]

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