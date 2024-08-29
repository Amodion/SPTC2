
# DESCRIPTION

## NET 1

Faster-RCNN architecture applyied for living buildings detection on imges taken from drone. Exmples shown below.
![Net_1](https://github.com/user-attachments/assets/c6e7dc88-96c0-4ab5-8331-9db1907b1950)

![Net_1_1](https://github.com/user-attachments/assets/fb183c85-aeda-4bcf-b423-c291ff8b3399)


## NET 2

Keypoint-RCNN architecture applyied for buildings corners detection on images. Exaples shown below.
![Net_2](https://github.com/user-attachments/assets/e134385b-d3ec-48e5-ae73-6b9c61d94c85)
![Net_2_1](https://github.com/user-attachments/assets/aa15f8b5-1227-40f8-8ccf-525da449b3e7)
![Net_2_2](https://github.com/user-attachments/assets/9fa62f3d-77b2-4dec-8c4b-a9fbfc7dada7)
![Net_2_3](https://github.com/user-attachments/assets/520438ee-2e8a-476c-9b62-6e5bfa532683)
![Net_2_4](https://github.com/user-attachments/assets/88645901-6236-4127-91e6-57f9ed7c4fb6)


## NET 3

Work not finished. Idea was to detect power pillars and its ground point.

## NET 4

Mask-RCNN applyied for detection and segmentation of roads. Examples below.

![Net_4](https://github.com/user-attachments/assets/50e6187e-dda2-40ae-898d-21eeab2dacd0)
![Net_4_1](https://github.com/user-attachments/assets/3e5cb703-c93f-470d-aa41-f28a5a5429e2)


Also, Yolov8 model was trained for the same task. Examples are shown in the following section


## Deploy

Python script was written for deployment models to specific working software. Scripts starts with dialog window with configurations possibilities as shown below

![Пример_GUI](https://github.com/user-attachments/assets/5e33e367-d893-4467-9b5c-0a0777d587ea)

Results are as follows:

#### Building's corners

![Пример_точки](https://github.com/user-attachments/assets/3dfcde90-68e6-4019-a498-69b730e714d3)
![Пример_точки_2](https://github.com/user-attachments/assets/7385e493-062f-4570-af98-e843e76025aa)
![Пример_точки_3](https://github.com/user-attachments/assets/0a492ce0-a8c0-47cf-bdd7-47cc06614402)

#### Roads

Note: Red dots show countours of roads detected and segmented via Mask-RCNN. Blue ones via Yolov8.
![Пример_дороги](https://github.com/user-attachments/assets/2c4a21a2-a20f-4cf7-9f7f-bd8ff4465c60)
![Пример_дороги_2](https://github.com/user-attachments/assets/66ed69b8-489d-459e-bb0c-078c9a76ff60)
![Пример_дороги_3](https://github.com/user-attachments/assets/fd2f3b99-fdc8-4b96-a6f6-c03458d7d0a5)


