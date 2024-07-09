from PySide2 import QtGui, QtCore, QtWidgets

#import urllib.request, tempfile
#temporary_file = tempfile.NamedTemporaryFile(delete=False)
#find_links_file_url = "https://raw.githubusercontent.com/agisoft-llc/metashape-scripts/master/misc/links.txt"
#urllib.request.urlretrieve(find_links_file_url, temporary_file.name)    

import Metashape
import torch
import pandas as pd
import time
import pathlib
import cv2
#from PIL import Image
import os
import numpy as np
from albumentations import RandomBrightnessContrast, InvertImg, Sharpen, Compose, Sequential
from torchvision.transforms import functional as F
from ultralytics import YOLO


class DetectObjectsDlg(QtWidgets.QDialog):

    def __init__(self, parent):

        self.group_label = 'Обнаруженные здания'
        self.kps_group_label = 'Обнаруженные точки'
        self.roads_group_label = 'Обнаруженные дороги'
        
        if len(Metashape.app.document.path) > 0:
            self.working_dir = str(pathlib.Path(Metashape.app.document.path).parent)
        else:
            self.working_dir = ""
            
        self.bottom = Metashape.Elevation.bottom
        
        self.major_version = float(".".join(Metashape.app.version.split('.')[:2]))
            
        self.current_image = ''
        
        self.ortho_path = self.working_dir + '/Orthomosaic/'
        self.roads_ortho_path = self.working_dir + '/Orthomosaic_roads/'
        self.do_export_ortho = False
        self.do_filter_points = True
        self.do_detect_buildings = True
        self.do_detect_roads = False
        self.yolo_roads = False
        self.use_path_mode = False
        self.model_path = 'C:/Users/User/Desktop' + '/NN_models/Building_detection_model.pth'
        self.model_kps_default_path = 'C:/Users/User/Desktop' + '/NN_models/Keypoints_detection_model.pth'
        self.model_kps_snow_path = 'C:/Users/User/Desktop' + '/NN_models/Keypoints_detection_model_snow.pth'
        self.model_mask_path = 'C:/Users/User/Desktop' + '/NN_models/Roads_detection_model.pth'
        self.model_yolo_mask_path = 'C:/Users/User/Desktop' + '/NN_models/YOLO_model.pt'
        
        self.buildings_path = 'C:/Users/User/Desktop/Buildings/'#self.working_dir + '/Buildings/'  C:\Users\User\Desktop\Teobox_Kedrovka_2023-06-06T11.31.53
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        #self.device = torch.device('cpu')
        
        self.use_snow_model = False
        self.do_visualize = False
        self.widgets_to_disable = []
        
        self.predicted_points = []
        self.predicted_points_scores = []
        self.roads_points = []
        
        self.chunk = Metashape.app.document.chunk
        self.ortho_crs = self.chunk.orthomosaic.crs
        self.num_keypoints = 17

        self.roads_threshold = 0.95
        self.mask_threshold = {1: 0.55, 2: 0.61, 3: 0.1} #{'Asphalt road': 1, 'Country road': 2, 'Water': 3}

        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle(f"Поиск точек на ортофотоплане (Metashape version: {self.major_version})")

        self.create_gui()

        self.exec()

    def stop(self):
        self.stopped = True

    def check_stopped(self):
        if self.stopped:
            self.extract_points()
            time_total = self.format_timedelta(time.time() - self.time_start)
            raise InterruptedError(f"Вы остановили процесс./nПоиск длился {time_total}")

    def create_gui(self):

        self.groupBoxGeneral = QtWidgets.QGroupBox("Настройки ортофотоплана")
        generalLayout = QtWidgets.QGridLayout()
        
        self.checkIfExportOtho = QtWidgets.QCheckBox("Экспортировать ортофотоплан")
        self.checkIfExportOtho.setToolTip("Экспорт ортофотоплана необходим для поиска зданий")
        self.checkIfExportOtho.setChecked(self.do_export_ortho)
        
        self.txtOrthoExp= QtWidgets.QLabel()
        self.txtOrthoExp.setText("Директория с ортофотопланом должна содержать изображения формата 'tiff' и соответсвующие мировые файлы. Укажите пустую папку для экспорта.")

        self.txtWorkingDir= QtWidgets.QLabel()
        self.txtWorkingDir.setText("Директория ортофотоплана:")
        self.edtWorkingDir= QtWidgets.QLineEdit()
        self.edtWorkingDir.setText(self.ortho_path)
        self.edtWorkingDir.setPlaceholderText("Путь к папке с ортофотопланом и мировыми файлами")
        self.edtWorkingDir.setToolTip("Путь к папке с ортофотопланом и мировыми файлами")
        self.btnWorkingDir = QtWidgets.QPushButton("...")
        self.btnWorkingDir.setFixedSize(25, 25)
        QtCore.QObject.connect(self.btnWorkingDir, QtCore.SIGNAL("clicked()"), lambda: self.choose_ortho_dir())
        
        self.groupBoxBuildingModelLoad = QtWidgets.QGroupBox("Настройки поиска зданий")
        LoadLayout = QtWidgets.QGridLayout()
        
        #self.checkIfDetectBuildings = QtWidgets.QCheckBox("Искать здания")
        #self.checkIfDetectBuildings.setChecked(self.do_detect_buildings)
        #self.txtDetectBuildings = QtWidgets.QLabel()
        #self.txtDetectBuildings.setText("Если отмечено, будет произведен поиск зданий.")
        
        self.txtModelLoadPath = QtWidgets.QLabel()
        self.txtModelLoadPath.setText("Путь к модели для поиска зданий:")
        self.edtModelLoadPath = QtWidgets.QLineEdit()
        self.edtModelLoadPath.setText(self.model_path)
        self.edtModelLoadPath.setPlaceholderText("Файл с моделью для поиска зданий")
        self.edtModelLoadPath.setToolTip("Файл с моделью для поиска зданий")
        self.btnModelLoadPath = QtWidgets.QPushButton("...")
        self.btnModelLoadPath.setFixedSize(25, 25)
        QtCore.QObject.connect(self.btnModelLoadPath, QtCore.SIGNAL("clicked()"), lambda: self.choose_building_model_load_path())

        self.widgets_to_disable.append(self.edtModelLoadPath)
        self.widgets_to_disable.append(self.btnModelLoadPath)

        self.groupBoxBuildingModelLoad.setLayout(LoadLayout)
        
        self.groupBoxKPSModelLoad = QtWidgets.QGroupBox("Настройка поиска точек")
        KPSLoadLayout = QtWidgets.QGridLayout()
        
        self.txtKPSModelLoadPath = QtWidgets.QLabel()
        self.txtKPSModelLoadPath.setText("Путь к модели для поиска точек:")
        self.edtKPSModelLoadPath = QtWidgets.QLineEdit()
        self.edtKPSModelLoadPath.setText(self.model_kps_default_path)
        self.edtKPSModelLoadPath.setPlaceholderText("Файл с моделью для поиска точек")
        self.edtKPSModelLoadPath.setToolTip("Файл с моделью для поиска точек")
        self.btnKPSModelLoadPath = QtWidgets.QPushButton("...")
        self.btnKPSModelLoadPath.setFixedSize(25, 25)
        QtCore.QObject.connect(self.btnKPSModelLoadPath, QtCore.SIGNAL("clicked()"), lambda: self.choose_points_model_load_path())

        self.widgets_to_disable.append(self.edtKPSModelLoadPath)
        self.widgets_to_disable.append(self.btnKPSModelLoadPath)
        
        self.txtKPSModelSnowLoadPath = QtWidgets.QLabel()
        self.txtKPSModelSnowLoadPath.setText("Путь к зимней модели для поиска точек:")
        self.edtKPSModelSnowLoadPath = QtWidgets.QLineEdit()
        self.edtKPSModelSnowLoadPath.setText(self.model_kps_snow_path)
        self.edtKPSModelSnowLoadPath.setPlaceholderText("Файл с зимней моделью для поиска точек")
        self.edtKPSModelSnowLoadPath.setToolTip("Файл с зимней моделью для поиска точек")
        self.btnKPSModelSnowLoadPath = QtWidgets.QPushButton("...")
        self.btnKPSModelSnowLoadPath.setFixedSize(25, 25)
        QtCore.QObject.connect(self.btnKPSModelSnowLoadPath, QtCore.SIGNAL("clicked()"), lambda: self.choose_points_snow_model_load_path())

        self.widgets_to_disable.append(self.edtKPSModelSnowLoadPath)
        self.widgets_to_disable.append(self.btnKPSModelSnowLoadPath)

        self.txtRoadsModelLoadPath = QtWidgets.QLabel()
        self.txtRoadsModelLoadPath.setText("Путь к модели для поиска точек дорог:")
        self.edtRoadsModelLoadPath = QtWidgets.QLineEdit()
        self.edtRoadsModelLoadPath.setText(self.model_mask_path)
        self.edtRoadsModelLoadPath.setPlaceholderText("Файл с зимней моделью для поиска точек")
        self.edtRoadsModelLoadPath.setToolTip("Файл с зимней моделью для поиска точек")
        self.btnRoadsModelLoadPath = QtWidgets.QPushButton("...")
        self.btnRoadsModelLoadPath.setFixedSize(25, 25)
        QtCore.QObject.connect(self.btnRoadsModelLoadPath, QtCore.SIGNAL("clicked()"), lambda: self.choose_roads_model_load_path())

        self.widgets_to_disable.append(self.edtRoadsModelLoadPath)
        self.widgets_to_disable.append(self.btnRoadsModelLoadPath)
        
        self.checkIfVisualize = QtWidgets.QCheckBox("Записывать предсказания на фото")
        self.checkIfVisualize.setToolTip("Записывать предсказания модели на фото зданий в отдельную папку")
        self.checkIfVisualize.setChecked(self.do_visualize)

        self.widgets_to_disable.append(self.checkIfVisualize)
        
        self.checkIfUseSnowModel = QtWidgets.QCheckBox("Использовать модель для зимы (BETA)")
        self.checkIfUseSnowModel.setToolTip("Использовать, если съемка проводилась в зимнее время")
        self.checkIfUseSnowModel.setChecked(self.use_snow_model)
        self.checkIfUseSnowModel.setEnabled(False)
        
        self.checkIfFilterPoints = QtWidgets.QCheckBox("Фильтровать точки")
        self.checkIfFilterPoints.setChecked(self.do_filter_points)
        self.widgets_to_disable.append(self.checkIfFilterPoints)
        
        self.fill= QtWidgets.QLabel()
        self.fill.setText("                                                            .                                                                             ")
        
        self.combo = QtWidgets.QComboBox()
        self.combo.addItem('Weighted')
        self.combo.addItem('Mean')
        self.combo.addItem('Max')
        self.widgets_to_disable.append(self.combo)

        self.combo_road = QtWidgets.QComboBox()
        self.combo_road.addItem('YOLOv8')
        self.combo_road.addItem('Mask_RCNN')
        self.widgets_to_disable.append(self.combo_road)
        
        self.filterKPSdistance = QtWidgets.QLineEdit()
        self.filterKPSdistance.setText('0.5')
        self.filterKPSdistance.setPlaceholderText("[0.1, 1]")
        self.filterKPSdistance.setToolTip("Расстояние фильтрации")
        self.widgets_to_disable.append(self.filterKPSdistance)
        
        
        
        self.KPSconfidanceText = QtWidgets.QLabel()
        self.KPSconfidanceText.setText("Уровень уверенности точек")
        
        self.comboText = QtWidgets.QLabel()
        self.comboText.setText("Метод фильтрации")
        
        self.KPSconfidance = QtWidgets.QLineEdit()
        self.KPSconfidance.setText('10.0')
        self.KPSconfidance.setToolTip("Уровень уверенности точек")
        self.widgets_to_disable.append(self.KPSconfidance)

        self.checkIfUsepathcMode = QtWidgets.QCheckBox("Использовать метод патчей")
        self.checkIfUsepathcMode.setChecked(self.use_path_mode)
        self.txtUsepathcMode = QtWidgets.QLabel()
        self.txtUsepathcMode.setText("Использовать метод патчей для поиска точек.")
        self.widgets_to_disable.append(self.checkIfUsepathcMode)

        self.checkIfDetectRoads = QtWidgets.QCheckBox("Искать дороги")
        self.checkIfDetectRoads.setChecked(self.do_detect_roads)
        self.txtDetectRoads = QtWidgets.QLabel()
        self.txtDetectRoads.setText("Произвести поиск дорог на ортофотоплане")
        self.widgets_to_disable.append(self.checkIfDetectRoads)

        self.checkIfDetectBuildings = QtWidgets.QCheckBox("Искать здания")
        self.checkIfDetectBuildings.setChecked(self.do_detect_buildings)
        self.txtDetectBuildings = QtWidgets.QLabel()
        self.txtDetectBuildings.setText("Произвести поиск углов зданий на ортофотоплане")
        self.widgets_to_disable.append(self.checkIfDetectBuildings)

        
        
        
        KPSLoadLayout.addWidget(self.txtModelLoadPath, 0, 0)
        KPSLoadLayout.addWidget(self.edtModelLoadPath, 0, 1)
        KPSLoadLayout.addWidget(self.btnModelLoadPath, 0, 2)
        
        
        KPSLoadLayout.addWidget(self.txtKPSModelLoadPath, 1, 0)
        KPSLoadLayout.addWidget(self.edtKPSModelLoadPath, 1, 1)
        KPSLoadLayout.addWidget(self.btnKPSModelLoadPath, 1, 2)
        
        KPSLoadLayout.addWidget(self.txtKPSModelSnowLoadPath, 2, 0)
        KPSLoadLayout.addWidget(self.edtKPSModelSnowLoadPath, 2, 1)
        KPSLoadLayout.addWidget(self.btnKPSModelSnowLoadPath, 2, 2)

        KPSLoadLayout.addWidget(self.txtRoadsModelLoadPath, 3, 0)
        KPSLoadLayout.addWidget(self.edtRoadsModelLoadPath, 3, 1)
        KPSLoadLayout.addWidget(self.btnRoadsModelLoadPath, 3, 2)
        
        KPSLoadLayout.addWidget(self.checkIfUseSnowModel, 4, 0)
        
        KPSLoadLayout.addWidget(self.checkIfVisualize, 4, 1)
        
        KPSLoadLayout.addWidget(self.checkIfFilterPoints, 5, 0)
        
        KPSLoadLayout.addWidget(self.filterKPSdistance, 5, 1)
        
        KPSLoadLayout.addWidget(self.comboText, 6, 0)
        
        KPSLoadLayout.addWidget(self.combo, 6, 1)
        
        KPSLoadLayout.addWidget(self.KPSconfidanceText, 7, 0)
        
        KPSLoadLayout.addWidget(self.KPSconfidance, 7, 1)

        KPSLoadLayout.addWidget(self.checkIfUsepathcMode, 9, 0)
        KPSLoadLayout.addWidget(self.combo_road, 9, 1)
        KPSLoadLayout.addWidget(self.checkIfDetectRoads, 8, 1)
        KPSLoadLayout.addWidget(self.checkIfDetectBuildings, 8, 0)
        

        self.groupBoxKPSModelLoad.setLayout(KPSLoadLayout)

        self.btnRun = QtWidgets.QPushButton(f"Пуск (Используется {self.device})")
        self.widgets_to_disable.append(self.btnRun)
        self.btnStop = QtWidgets.QPushButton("Стоп")
        self.btnStop.setEnabled(False)

        layout = QtWidgets.QGridLayout()
        row = 0

        layout.addWidget(self.groupBoxKPSModelLoad, row, 0, 1, 3)
        row += 1
        
        self.txtDetectionPBar = QtWidgets.QLabel()
        self.txtDetectionPBar.setText(f"Прогресс поиска зданий:")
        self.detectionPBar = QtWidgets.QProgressBar()
        self.detectionPBar.setTextVisible(True)
        
        
        layout.addWidget(self.txtDetectionPBar, row, 0)
        
        layout.addWidget(self.detectionPBar, row, 1, 1, 2)
        row += 1
        
        self.txtDetectionKPSPBar = QtWidgets.QLabel()
        self.txtDetectionKPSPBar.setText("Прогресс поиска точек:")
        self.detectionKPSPBar = QtWidgets.QProgressBar()
        self.detectionKPSPBar.setTextVisible(True)
        layout.addWidget(self.txtDetectionKPSPBar, row, 0)
        layout.addWidget(self.detectionKPSPBar, row, 1, 1, 2)
        row += 1

        layout.addWidget(self.btnRun, row, 1)
        layout.addWidget(self.btnStop, row, 2)
        row += 1

        self.setLayout(layout)
        
        self.resize(1000, 600)

        QtCore.QObject.connect(self.btnRun, QtCore.SIGNAL("clicked()"), lambda: self.process())
        QtCore.QObject.connect(self.btnStop, QtCore.SIGNAL("clicked()"), lambda: self.stop())

    def choose_building_model_load_path(self):
        working_dir = Metashape.app.getOpenFileName()
        self.edtModelLoadPath.setText(working_dir)
        self.model_path = self.edtModelLoadPath.text()

    def choose_points_snow_model_load_path(self):
        working_dir = Metashape.app.getOpenFileName()
        self.edtKPSModelSnowLoadPath.setText(working_dir)
        self.model_kps_snow_path = self.edtKPSModelLoadPath.text()
        
    def choose_points_model_load_path(self):
        working_dir = Metashape.app.getOpenFileName()
        self.edtKPSModelLoadPath.setText(working_dir)
        self.model_kps_default_path = self.edtKPSModelLoadPath.text()

    def choose_roads_model_load_path(self):
        working_dir = Metashape.app.getOpenFileName()
        self.edtRoadsModelLoadPath.setText(working_dir)
        self.model_mask_path = self.edtRoadsModelLoadPath.text()

    def choose_ortho_dir(self):
        working_dir = Metashape.app.getExistingDirectory()
        self.edtWorkingDir.setText(working_dir)
        self.ortho_path = self.edtWorkingDir.text()

    
    def draw_shape_point(self, coordinates: Metashape.Vector, label: str = '', roads=False):
        if len(coordinates) == 0:
            print('None in draw point')
            return None
        new_shape = self.chunk.shapes.addShape()
        new_shape.label = label
        if roads:
            group = self.roads_group
        else:
            group = self.target_group_kps
        new_shape.group = group
        new_shape.geometry = Metashape.Geometry.Point(coordinates)
        #new_shape.is_attached = True

    def draw_shape_point_1_6(self, coordinates: Metashape.Vector, label: str = '', roads=False):
        if len(coordinates) == 0:
            print('None in draw point')
            return None
        shape = self.chunk.shapes.addShape()
        shape.label = label
        shape.type = Metashape.Shape.Type.Point
        if roads:
            group = self.roads_group
        else:
            group = self.target_group_kps
        shape.group = group
        shape.vertices = [coordinates]
        #shape.is_attached = True
        shape.has_z = True
    
    def get_object_detection_model(self, num_classes):
        from torch import nn
        import torchvision
        from torchvision.models.detection import FasterRCNN
        from torchvision.models.detection.rpn import AnchorGenerator
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
        return model

    def load_model(self):   
        model = self.get_object_detection_model(num_classes=2)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        return model

    def image_transform_kps(self):
        if self.use_snow_model:
            return Compose([
                Sequential([
                RandomBrightnessContrast(brightness_limit=(0.0, 0.5), contrast_limit=0.0, brightness_by_max=True, always_apply=False, p=1.0),
                InvertImg(p=1)
                ], p=1)
            ])
        else:
            return Compose([
                Sequential([
                ], p=1)
            ])
    
    def eval_transform(self):
        from albumentations import Compose, ToFloat
        from albumentations.pytorch import ToTensorV2    
        return Compose([ToFloat(p=1.0), ToTensorV2(p=1.0)])
    
    @torch.no_grad()
    def predict_images(self, model) -> None:
        app = QtWidgets.QApplication.instance()
        import cv2
        from torchvision.utils import draw_bounding_boxes, save_image
        
        model_kps = self.get_model_kps_V2()
        
        classes = {'Building': 1}
        inv_classes = {value:key for key, value in classes.items()}    
        model.to(self.device)  
        cameras = self.chunk.cameras
        for num, cam in enumerate(cameras):
            im = cam.photo.path
            try:
                image = cv2.imdecode(np.fromfile(im, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except:
                print(f'Existing/cvt2 error at image: {im}')
            transform = self.eval_transform()        
            model.eval()       
            x = transform(image=image)       
            x = x['image']        
            x = x[:3, ...].to(self.device)        
            predictions = model([x, ])        
            pred = predictions[0]
            pred = self.cut_score_V2(predictions=pred, score_thresh=0.95)
            df = pd.DataFrame(data=pred['boxes'], columns=['xmin', 'ymin', 'xmax', 'ymax'])
            df['label'] = [inv_classes[label] + f': {round(score, 2)}' for label, score in zip(pred['labels'], pred['scores'])]
            
            tuples = list(df.itertuples())

            for idx, rect in enumerate(df.itertuples()):
                xmin = int(torch.FloatTensor.item(rect.xmin.detach().cpu())) - 50
                xmax = int(torch.FloatTensor.item(rect.xmax.detach().cpu())) + 50
                ymin = int(torch.FloatTensor.item(rect.ymin.detach().cpu())) - 50
                ymax = int(torch.FloatTensor.item(rect.ymax.detach().cpu())) + 50
            
                if xmin < 0:
                    xmin = 0
                if ymin < 0:
                    ymin = 0
                
                if xmax > image.shape[1] - 1:
                    xmax = image.shape[1] - 1
                if ymax > image.shape[0] - 1:
                    ymax = image.shape[0] - 1
            
                crop_image = image[ymin:ymax, xmin:xmax, :]
                
                points, scores = self.predict_points(model=model_kps, camera=cam, boundaries=[ymin, ymax, xmin, xmax], cropimage=crop_image, idx=idx)
                
                if points is None:
                    points, scores = [], []
                
                self.predicted_points.extend(points)
                self.predicted_points_scores.extend(scores)
                
                self.detectionKPSPBar.setValue((idx + 1) * 100 / len(tuples))
                Metashape.app.update()
                app.processEvents()
                self.check_stopped()
                
            self.detectionPBar.setValue((num + 1) * 100 / len(cameras))
            Metashape.app.update()
            app.processEvents()
            self.check_stopped()
            
        self.extract_points()
        
    @torch.no_grad()
    def predict_points(self, model, camera, boundaries, cropimage, idx):
        import torchvision
        from torchvision.transforms import functional as F
        
        img = cropimage
        img_orig = cropimage
        if img.shape[0] < 5 or img.shape[1] < 5:
            print(f'Shape drop at {camera.label}')
            return None, None
        transform = self.image_transform_kps()
        img = transform(image=img)['image']
        img = F.to_tensor(img)
        img = img.to(self.device)
        
        with torch.no_grad():
            model.to(self.device)
            model.eval()
            output = model([img,])[0]

        scores = output['scores'].detach().cpu().numpy()
        high_scores_idxs = np.where(scores > 0.9)[0].tolist() # Indexes of boxes with scores > 0.7
        post_nms_idxs = torchvision.ops.nms(output['boxes'][high_scores_idxs], output['scores'][high_scores_idxs], 0.3).cpu().numpy() # Indexes of boxes left after applying NMS (iou_threshold=0.3)

        if not len(output['keypoints_scores'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy()):
            good_kps_idxs = [[]]
        else:
            good_kps_idxs = np.where(output['keypoints_scores'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy()[0] > self.points_confidance) # for i in range(len(post_nms_idxs))
    
        keypoints = []
        
        for kps in output['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
            keypoints.append([list(map(int, kp)) for kp in kps[good_kps_idxs[0]]])
            
        if len(keypoints) == 0:
            return None, None
        
        keypoints_scores = output['keypoints_scores'].detach().cpu().numpy()
        kps_scores = keypoints_scores[0][good_kps_idxs[0]]
        
        bboxes = []
        for bbox in output['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
            bboxes.append(list(map(int, bbox.tolist())))
        bld_scores = output['scores'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy().tolist()
        label = camera.label + '_' + str(idx)
        
        if self.do_visualize:
            self.visualize(img_orig, bboxes, keypoints, label=label, bld_score=bld_scores, kps_scores=kps_scores)
        
        if self.major_version <= 1.8:
            surface = self.chunk.dense_cloud
        else:
            surface = self.chunk.point_cloud
        points = []
        DEM = self.chunk.elevation
        for num, point in enumerate(keypoints[0]):
            tile_x = boundaries[2] + point[0]
            tile_y = boundaries[0] + point[1]

            T = self.chunk.transform.matrix
            crs = self.chunk.crs

            p = surface.pickPoint(camera.center, camera.unproject(Metashape.Vector([tile_x, tile_y])))
            try:
                P = crs.project(T.mulp(p))
            except:
                print(f'P skipped at {camera.label} with boundaries {boundaries}')
                continue
            elevation = DEM.altitude(Metashape.Vector([P.x, P.y]))
            if abs(P.z - elevation) > 0.8:
                continue
            points.append(P)
             
        return points, kps_scores
    
    def extract_points(self):
        if self.do_use_patchmode and (not self.do_visualize):
            self.del_misc(pathlib.Path(self.working_dir + '/Patches/'))
        if self.do_detect_roads:
            self.extract_roads_points()
        if not len(self.predicted_points):
            print('No points detected')
            return None
        print(f'Extracting points. Number of detected points is: {len(self.predicted_points)}')
        predicted_points = self.predicted_points
        scores = self.predicted_points_scores
        if self.do_filter_points:
            time_filter = time.time()
            predicted_points, scores = self.filter_points(self.predicted_points, self.predicted_points_scores)
            print(f'Points filtered for {round(time.time() - time_filter, 2)} sec. Method: {self.combo.currentText()}')
            print(f'Number of points after filtering is: {len(predicted_points)}')
        else:
            print('No filter applyied')
        #print(len(predicted_points), len(scores))
        for point, score in zip(predicted_points, scores):
            score = round(float(score), 2)
            if self.major_version > 1.7:
                self.draw_shape_point(label=f'Point_{score}', coordinates=point)
            else:
                self.draw_shape_point_1_6(label=f'Point_{score}', coordinates=point)

    def extract_roads_points(self):
        #cycle over self.roads_points and calling draw function depending on version
        for point in self.roads_points:
            if self.major_version > 1.7:
                self.draw_shape_point(coordinates=point, roads=True)
            else:
                self.draw_shape_point_1_6(coordinates=point, roads=True)
    
    def cut_score_V2(self, predictions: dict, score_thresh = 0.5):
        new_pred = {'boxes': [], 'labels': [], 'scores': []}
        for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
            if score >= score_thresh:
                new_pred['boxes'].append(box)
                new_pred['labels'].append(torch.IntTensor.item(label))
                new_pred['scores'].append(torch.FloatTensor.item(score))
        return new_pred

    def get_model_kps(self):
        import torchvision
        from torchvision.ops import MultiScaleRoIAlign
        from torchvision.models.detection import KeypointRCNN
        from torchvision.models.detection.rpn import AnchorGenerator
    
        backbone = torchvision.models.mobilenet_v2(weights='DEFAULT').features
        backbone.out_channels = 1280
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512, 1024),), aspect_ratios=((0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0),))
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=7, sampling_ratio=2)
        keypoint_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],  output_size=14, sampling_ratio=2)
        model = KeypointRCNN(backbone, num_classes=2, num_keypoints=self.num_keypoints, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler, keypoint_roi_pool=keypoint_roi_pooler)
        
        state_dict = torch.load(self.model_kps_path, map_location=self.device)
        model.load_state_dict(state_dict)        
        
        return model

    def get_model_kps_V2(self):
        from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor
        from torchvision.models.detection.rpn import AnchorGenerator
        from torchvision.models.detection import keypointrcnn_resnet50_fpn

        model = keypointrcnn_resnet50_fpn(weights='DEFAULT')

        in_features = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
        model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(in_channels=in_features, num_keypoints=self.num_keypoints)

        model.name = 'keypointrcnn_resnet50_fpn'
        state_dict = torch.load(self.model_kps_path, map_location=self.device)
        model.load_state_dict(state_dict)        
        
        return model
    
    def visualize(self, image, bboxes, keypoints, image_original=None, bboxes_original=None, keypoints_original=None, label=None, bld_score=None, kps_scores=None):
        import seaborn as sns
        fontsize = 18
        sns.set_style("white")
    
        for num, bbox in enumerate(bboxes):
            start_point = (bbox[0], bbox[1])
            end_point = (bbox[2], bbox[3])
            image = cv2.rectangle(image.copy(), start_point, end_point, (0,255,0), 1)
            score = round(bld_score[num], 2)
            image = cv2.putText(image.copy(), ' Building' + str(score), start_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    
        radius = 2 + 2*int(image.shape[0] > 700) + 2*int(image.shape[1] > 700)
        colors = {0: (0, 0, 255), 1: (255, 0, 0)}
    
        for kps in keypoints:
            for idx, kp in enumerate(kps):
                image = cv2.circle(image.copy(), tuple(kp[:2]), radius, colors[int(kp[2])], -1)
                if kps_scores is not None:
                    score = round(kps_scores[idx], 2)
                    image = cv2.putText(image.copy(), ' ' + str(score), tuple(kp[:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 1, cv2.LINE_AA)
        if label:    
            cv2.imwrite(self.buildings_path + f'{label}.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        elif image_original is None and keypoints_original is None:
            plt.figure(figsize=(15,15))
            plt.imshow(image)
        else:
            for bbox in bboxes_original:
                start_point = (bbox[0], bbox[1])
                end_point = (bbox[2], bbox[3])
                image_original = cv2.rectangle(image_original.copy(), start_point, end_point, (0,255,0), 1)
        
            for kps in keypoints_original:
                for idx, kp in enumerate(kps):
                    image_original = cv2.circle(image_original.copy(), tuple(kp[:2]), radius, colors[int(kp[2])], -1)

            f, ax = plt.subplots(1, 2, figsize=(20, 20))

            ax[0].imshow(image_original)
            ax[0].set_title('Original image', fontsize=fontsize)

            ax[1].imshow(image)
            ax[1].set_title('Transformed image', fontsize=fontsize)

    def weighted_average(self, points):
        import numpy as np
        xyz = [point[:-1] for point in points]
        weights = [score[-1] for score in points]
        average = np.average(xyz, axis=0, weights=weights).tolist()
        weights = np.mean(weights).tolist()
        average.append(weights)
        return tuple(average)

    def mean_points(self, points):
        import numpy as np
        return tuple(np.mean(points, axis=0).tolist())

    def max_point(self, points):
        return max(points, key=lambda x: x[-1])

    def is_close(self, point1, point2):
        self.close_distance = float(self.filterKPSdistance.text())
        return (abs(point1[0] - point2[0]) < self.close_distance) and (abs(point1[1] - point2[1]) < self.close_distance)

    def filter_points(self, points: Metashape.Vector, scores):
        print('Start filtering points...')
        #print(len(points), len(scores))
        if not self.do_use_patchmode:
            points_tuple = [(point.x, point.y, point.z, score) for point, score in zip(points, scores)]
        else:
            points_tuple = [(point.x, point.y, score) for point, score in zip(points, scores)]
        final_points = set()
    
        for point in points_tuple:
            filtered_points = filter(lambda x: self.is_close(x, point), points_tuple)
            filtered_points = list(filtered_points)
            if len(filtered_points) >= 1:
                final_points.add(self.filter_method(list(filtered_points)))
        
        #print(len(final_points))
        filtered_vectors = ([Metashape.Vector(point[:-1]) for point in list(final_points)], [point[-1] for point in list(final_points)])
        return filtered_vectors
        
    def get_mask_model_v2(self, weights_path=None):
        from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, MaskRCNN_ResNet50_FPN_Weights
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
        
        model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        in_features_box = model.roi_heads.box_predictor.cls_score.in_features
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

        dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels

        model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features_box, num_classes=4)

        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels=in_features_mask, dim_reduced=dim_reduced, num_classes=4)

        if weights_path:
            state_dict = torch.load(weights_path)
            model.load_state_dict(state_dict)   

        model.name = 'maskrcnn_resnet50_fpn_v2'
        return model
    
    def detect_roads(self):
        self.export_ortho(width=2500, height=2500, roads=True)
        if self.yolo_roads:
            self.predict_roads_yolo()
        else:
            road_model = self.get_mask_model_v2(weights_path=self.model_mask_path)
            self.predict_roads(model=road_model)
            self.predict_roads_yolo()
        
    def detect_buildings(self):
        model = self.load_model()
        print('Model loaded')
        time_start = time.time()
        self.predict_images(model=model)

    def buildings_exists(self):
        return (self.group_label in [group.label for group in self.chunk.shapes.groups])

    def detect_buildings_Patch_mode(self):
        #Create folder for export ortho

        if self.create_building_layer:
            pathlib.Path(self.ortho_path).mkdir(parents=True, exist_ok=True)
            self.export_ortho()
            self.predict_buildings()
        else:
            print('Обнаружен слой со зданиями. Поиск проводится на основе этого слоя.')
            
        self.treat_buildings()

    def get_contours(self, target):
        if not len(target['masks']):
            return None, None
        masks = target['masks'] #> self.mask_threshold
        boxes = target['boxes']
    
        masks = masks.squeeze(1)

        total_contours = []
        total_boxes = []
    #masks = masks.detach().cpu().numpy()
        labels = set(target['labels'].detach().cpu().numpy())

        for label in labels:
            label_masks = target['labels'] == label

            label_boxes = boxes[label_masks]

            label_boxes = label_boxes.detach().cpu().numpy()

            label_masks = masks[label_masks] > self.mask_threshold[label]

            label_masks = label_masks.detach().cpu().numpy()

            total_mask = label_masks[0]

            for i in range(1, label_masks.shape[0]):
                total_mask = total_mask | label_masks[i]
        
            total_mask = total_mask.astype('uint8')
            contours, heir = cv2.findContours(total_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            areas = [cv2.contourArea(c) for c in contours]
            try:
                max_area = max(areas)
            except ValueError:
                print('areas is empty. skip.')
            valid_contours = [c for c in contours if cv2.contourArea(c) > max_area*0.5]

            total_contours.extend(valid_contours)

            total_boxes.extend(label_boxes)

        return tuple(total_contours), total_boxes
    
    def predict_roads_yolo(self):
        app = QtWidgets.QApplication.instance()
        self.txtDetectionPBar.setText(f"Прогресс поиска дорог (YOLOv8):")
        Metashape.app.update()
        app.processEvents()

        yolo_model = YOLO(self.model_yolo_mask_path)
        
        yolo_model.to(self.device) #????

        orthodata = [file for file in os.listdir(self.roads_ortho_path) if not file.endswith('.tfw')]

        for num, image_name in enumerate(orthodata):
            world_file_path = self.roads_ortho_path + image_name[:-4] + '.tfw'

            with open(world_file_path, "r") as file:
                matrix2x3 = list(map(float, file.readlines()))
                matrix2x3 = np.array(matrix2x3).reshape(3, 2).T

            res = yolo_model.predict(self.roads_ortho_path + image_name, conf=0.25, retina_masks=True, show_labels=False) # set specific cons value?
            points = []
            
            for prediction in res: # One prediction per image given in input. Therefore res will contain one prediction
                if prediction.masks is None:
                    continue
                cntrs = prediction.masks.xy
                for cntr in cntrs:
                    step = int(len(cntr) * 0.15)
                    step = step if step > 0 else 1
                    for point in cntr[::step]:
                        x, y = point
                        x, y = int(x), int(y)
                        point = (x, y)
                        x, y = matrix2x3 @ np.array([x, y, 1]).reshape(3, 1)
                        x, y = x[0], y[0]
                        p = Metashape.Vector([x, y])
                        p = Metashape.CoordinateSystem.transform(p, self.chunk.orthomosaic.crs, self.chunk.shapes.crs)
                        #print(f'\t{p}')
                        points.append(p)

            self.roads_points.extend(points)


            self.detectionPBar.setValue((num + 1) * 100 / len(orthodata))
            Metashape.app.update()
            app.processEvents()
            self.check_stopped()

        del res
        del yolo_model
    
        # extract points from common set to MS project
        self.extract_roads_points()
                        
    
    @torch.no_grad()
    def predict_roads(self, model):
        app = QtWidgets.QApplication.instance()
        self.txtDetectionPBar.setText(f"Прогресс поиска дорог (Mask-RCNN):")
        Metashape.app.update()
        app.processEvents()

        classes = {'Asphalt road': 1, 'Country road': 2, 'Water': 3}
        inv_classes = {v: k for k, v in classes.items()}
        
        model.to(self.device)

        orthodata = [file for file in os.listdir(self.roads_ortho_path) if not file.endswith('.tfw')]

        for num, image_name in enumerate(orthodata):
            world_file_path = self.roads_ortho_path + image_name[:-4] + '.tfw'

            with open(world_file_path, "r") as file:
                matrix2x3 = list(map(float, file.readlines()))
                matrix2x3 = np.array(matrix2x3).reshape(3, 2).T
            
            try:
                image = cv2.imdecode(np.fromfile(self.roads_ortho_path + image_name, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except:
                print(f'Existing/cv2 error at image: {image_name}')
                continue
            transform = self.eval_transform()        
            model.eval()  

            x = transform(image=image)       
            x = x['image']
            x = x[:3, ...].to(self.device) # RGBA -> RGB 
            predictions = model([x,])        
            out = predictions[0]

            scores_valid = out['scores'] > self.roads_threshold
            labels_valid = out['labels'] < 3

            target = {}
            target['masks'] = out['masks'][scores_valid * labels_valid]
            target['boxes'] = out['boxes'][scores_valid * labels_valid]
            target['labels'] = out['labels'][scores_valid * labels_valid]

            image_contours, boxes = self.get_contours(target) #tuple of contours or None

            image_contours = image_contours if image_contours is not None else []

            boxes = boxes if boxes is not None else []

            points = []

            eps = 20

            for contour, box in zip(image_contours, boxes):
                x_min, y_min, x_max, y_max = box
                c = np.squeeze(contour)
                step = int(len(c) * 0.15)
                for point in c[::step]:
                    x, y = point #slicing?
                    if (abs(x - x_min) <= eps or abs(x - x_max) <= eps) or (abs(y - y_min) <= eps or abs(y - y_max) <= eps):
                        continue
                    x, y = matrix2x3 @ np.array([x, y, 1]).reshape(3, 1)
                    x, y = x[0], y[0]
                    p = Metashape.Vector([x, y])
                    p = Metashape.CoordinateSystem.transform(p, self.chunk.orthomosaic.crs, self.chunk.shapes.crs)
                    #print(f'\t{p}')
                    points.append(p)

            self.roads_points.extend(points)


            self.detectionPBar.setValue((num + 1) * 100 / len(orthodata))
            Metashape.app.update()
            app.processEvents()
            self.check_stopped()
    
        # extract points from common set to MS project
        self.extract_roads_points()
            
    
    def predict_buildings(self):
        model = self.load_model()
        app = QtWidgets.QApplication.instance()
        
        classes = {'Building': 1}
        inv_classes = {value:key for key, value in classes.items()}    
        model.to(self.device)

        orthodata = [file for file in os.listdir(self.ortho_path) if not file.endswith('.tfw')]
        
        for num, image_name in enumerate(orthodata): #Iterate over orthophotos ignoring world files
            world_file_path = self.ortho_path + image_name[:-4] + '.tfw'
            try:
                image = cv2.imdecode(np.fromfile(self.ortho_path + image_name, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except:
                print(f'Existing/cv2 error at image: {image_name}')
                continue
            transform = self.eval_transform()        
            model.eval()       
            
            x = transform(image=image)       
            x = x['image']
            x = x[:3, ...].to(self.device) # RGBA -> RGB 
            predictions = model([x,])        
            pred = predictions[0]
            
            scores_valid = pred['scores'] > 0.95 # filter out bad predictions
            
            target = {}
            target['boxes'] = pred['boxes'][scores_valid]
            target['labels'] = pred['labels'][scores_valid]
            target['scores'] = pred['scores'][scores_valid]
            
            target = {key: value.detach().cpu().numpy() for key, value in target.items()}
            
            df = pd.DataFrame(data=target['boxes'], columns=['xmin', 'ymin', 'xmax', 'ymax'])
            df['label'] = [inv_classes[label.item()] + f': {round(score.item(), 2)}' for label, score in zip(target['labels'], target['scores'])]
            
            self.add_buildings(world_file_path=world_file_path, predictions=df)

            self.detectionPBar.setValue((num + 1) * 100 / len(orthodata))
            Metashape.app.update()
            app.processEvents()
            self.check_stopped()

    def treat_buildings(self):
        app = QtWidgets.QApplication.instance()

        kps_model = self.get_model_kps_V2()
        
        T = self.chunk.transform.matrix
        crs = self.chunk.crs
        shapes = list(filter(self.filter_polygons, self.chunk.shapes))
        patch_path = self.working_dir + '/Patches/'
        pathlib.Path(patch_path).mkdir(parents=True, exist_ok=True)
        
        for num, shape in enumerate(shapes):
            shape_points = []
            shape_points_scores = []
            self.get_images_from_shape(shape=shape, path=patch_path)
            for image in [image for image in os.listdir(patch_path + str(shape.key)) if (not image.endswith('.jgw')) and (not 'pred' in image)]:
                points, scores = self.predict_points_patch(model=kps_model, path=patch_path+str(shape.key)+'/'+image)
                if points is None:
                    continue
                shape_points.extend(points)
                shape_points_scores.extend(scores)
            
            #if self.do_filter_points:
            #    shape_points, shape_points_scores = self.filter_points(shape_points, shape_points_scores)

            self.predicted_points.extend(shape_points)
            self.predicted_points_scores.extend(shape_points_scores)

            self.detectionKPSPBar.setValue((num + 1) * 100 / len(shapes))
            Metashape.app.update()
            app.processEvents()
            self.check_stopped()

        self.extract_points()

    def predict_points_patch(self, model, path):
        world_file_path = path[:-4] + '.jgw'
        try:
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            print(f'Existing/cv2 error at image: {image_name}')
            return None, None

        image = F.to_tensor(image)
        image.to(self.device)

        model.eval()

        predictions = model([image])[0]

        scores_valid = predictions['scores'] > 0.95
        target = {}
        target['boxes'] = predictions['boxes'][scores_valid]
        target['scores'] = predictions['scores'][scores_valid]
        target['labels'] = predictions['labels'][scores_valid]
        target['keypoints'] = predictions['keypoints'][scores_valid]
        target['keypoints_scores'] = predictions['keypoints_scores'][scores_valid]

        final_target = {}
        final_target['boxes'] = target['boxes']
        final_target['scores'] = target['scores']
        final_target['labels'] = target['labels']

        valid_kps = target['keypoints_scores'][:] > self.points_confidance
        final_target['keypoints'] = target['keypoints'][valid_kps]
        final_target['keypoints_scores'] = target['keypoints_scores'][valid_kps]

        keypoints = final_target['keypoints'].detach().cpu().numpy()
        scores = final_target['keypoints_scores'].detach().cpu().numpy()

        keypoints = [list(keypoint) for keypoint in keypoints] # DO I HAVE TO CONVERT POINTS TO INT?
        scores = [round(score, 2) for score in scores]

        if self.do_visualize:
            save_path = path[:-4] + '_pred.jpg'
            self.visualise_patch(image, final_target, save_path)

        with open(world_file_path, "r") as file:
            matrix2x3 = list(map(float, file.readlines()))
        matrix2x3 = np.array(matrix2x3).reshape(3, 2).T

        points = []

        print(f'{len(keypoints)} keypoints found for image {path}:')

        for point, score in zip(keypoints, scores):
            x, y = point[:2]
            x, y = matrix2x3 @ np.array([x, y, 1]).reshape(3, 1)
            p = Metashape.Vector([x, y])
            p = Metashape.CoordinateSystem.transform(p, self.chunk.orthomosaic.crs, self.chunk.shapes.crs)
            print(f'\t{p}')
            points.append(p)

        return points, scores
    
    def visualise_patch(self, image, final_target, path):
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
        
    
    def export_shape_image(self, shape, label):
        coordinates = shape.geometry.coordinates[0]
        new_p = []
        for x, y in coordinates:
            t = Metashape.Vector([x, y])
            pair = Metashape.CoordinateSystem.transform(t, self.chunk.shapes.crs, self.chunk.orthomosaic.crs)
            pair = Metashape.Vector([round(val, 6) for val in pair])
            new_p.append(pair)
        s_min = new_p[3]
        s_max = new_p[1]
        box = Metashape.BBox()
        box.min = s_min
        box.max = s_max
        self.chunk.exportRaster(path=f"{label}.jpg", source_data=Metashape.OrthomosaicData, image_format=Metashape.ImageFormat.ImageFormatJPEG, save_alpha=True, white_background=True,
                                save_world=True,
                                split_in_blocks=False,
                                region=box)
    
    def get_images_from_shape(self, shape, path):
        cameras = self.get_overlaping_images_V2(shape)
        for camera in cameras:
            key = camera.key
            patch = Metashape.Orthomosaic.Patch()
            patch.image_keys = [key]
            self.chunk.orthomosaic.patches[shape] = patch
            try:
                self.chunk.orthomosaic.update()
            except KeyboardInterrupt:
                self.stop()
            #label = '_'.join(shape.label.split(':'))
            pathlib.Path(path + str(shape.key)).mkdir(parents=True, exist_ok=True)
            #if os.listdir(path + str(shape.key)):
            #    for file in os.listdir(path + str(shape.key)):
            #        os.remove(patch_path + str(shape.key) + '/' + file)
            label = path + str(shape.key) + '/' + str(camera.label)
            self.export_shape_image(shape, label)
            self.check_stopped()
    
    def filter_polygons(self, shape):
        if (shape.group.label == self.target_group.label and shape.geometry.type == Metashape.Geometry.PolygonType):
            return True
        else:
            return False
    
    def get_overlaping_images_V2(self, shape):
        '''
        Returns image keys which overlaps over the input shape.
        TODO: Figure out how to get specific boundaries for distance
        '''
        keys = []
        for camera in self.chunk.cameras:
            distance = self.calc_distance_points(self.get_shape_center(shape), camera.reference.location)
            if 30.0 < distance < 90.0:
                if not keys:
                    keys.append(camera.key)
                else:
                    if camera.key == keys[-1] + 1:
                        continue
                    else:
                        keys.append(camera.key)  
        cameras = [camera for camera in self.chunk.cameras if camera.key in keys]
        return cameras
    
    def draw_shape(self, label: str, corners: list):
        '''
        Создает на ортофотоплане фигуру полигон с именем 'label', в слое 'group' по координатам узлов в 'corners' (shapes.crs!) 
        '''
        if len(corners) == 0:
            print('None in draw point')
            return None
        if self.major_version > 1.7:
            new_shape = self.chunk.shapes.addShape()
            new_shape.label = label
            new_shape.group = self.target_group
            new_shape.geometry = Metashape.Geometry.Polygon(corners)
        else:   
            shape = self.chunk.shapes.addShape()
            shape.label = label
            shape.type = Metashape.Shape.Type.Polygon
            shape.group = self.target_group
            shape.vertices = corners
            #shape.has_z = True
    
    def add_buildings(self, world_file_path, predictions):
        '''
        TODO: Расширять ограничивающие прямоугольники путем добавления констант к координатам углов? Что бы область для поиска углов зданий была больше.
        '''
        with open(world_file_path, "r") as file:
            matrix2x3 = list(map(float, file.readlines()))
        matrix2x3 = np.array(matrix2x3).reshape(3, 2).T
        for row in predictions.itertuples():
            xmin, ymin, xmax, ymax, label = int(row.xmin), int(row.ymin), int(row.xmax), int(row.ymax), row.label
            corners = []
            for x, y in [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]:
                x, y = matrix2x3 @ np.array([x, y, 1]).reshape(3, 1)
                x, y = x[0], y[0]
                p = Metashape.Vector([x, y])
                p = Metashape.CoordinateSystem.transform(p, self.chunk.orthomosaic.crs, self.chunk.shapes.crs)
                corners.append([p.x, p.y])
            self.draw_shape(label=label, corners=corners)
        
    def export_ortho(self, width=5000, height=5000, roads=False):
        if not self.ortho_path or not self.roads_ortho_path:
            raise InterruptedError("Specify path to export orthomosaic!")
        
        if roads:
            path = self.roads_ortho_path
        else:
            path = self.ortho_path
        
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        if len(os.listdir(path)) > 0:
            print('Orthomosaic already exported. Proceed to predictions.')
        else:
            self.chunk.exportRaster(path=path + '/ortho.tif', source_data=Metashape.OrthomosaicData, image_format=Metashape.ImageFormat.ImageFormatTIFF, save_alpha=True, white_background=True,
                                    save_world=True,
                                    split_in_blocks=True, block_width=width, block_height=height,)
         
    def get_shape_center(self, shape):
        coordinates = shape.geometry.coordinates[0]
        lu_point, ru_point, rd_point, ld_point, *_ = coordinates
        x_center = lu_point[0] + self.calc_distance_points(lu_point, ru_point) / 2.0
        y_center = lu_point[1] - self.calc_distance_points(lu_point, ld_point) / 2.0
        return Metashape.Vector([x_center, y_center])
    
    def calc_distance_points(self, point1: Metashape.Vector, point2: Metashape.Vector):
        import math
        x1, y1, *_ = point1
        x2, y2, *_ = point2
        return math.sqrt((x2-x1)**2 + (y2-y1)**2)
    
    def format_timedelta(self, td):
        minutes, seconds = divmod(td, 60)
        hours, minutes = divmod(minutes, 60)
        return '{:02.0f}:{:02.0f}:{:02.0f}'.format(hours, minutes, seconds)

    def show_results_dialog(self):
        time_total = self.format_timedelta(self.results_time_total)
        message = f"Поиск завершен за {time_total}"
        print(message)
        Metashape.app.messageBox(message)

    def del_misc(self, path):
        for sub in path.iterdir():
            if sub.is_dir():
               self.del_misc(sub)
            else:
                sub.unlink()
        path.rmdir()
            
    def process(self):    
        try:
            self.stopped = False

            if not self.buildings_exists():
                self.target_group = self.chunk.shapes.addGroup()
                self.target_group.label = self.group_label
                self.target_group.enabled = False
                self.create_building_layer = True
            else:
                all_groups = {group.label: group for group in self.chunk.shapes.groups}
                self.target_group = all_groups[self.group_label]
                self.create_building_layer = False
            
            self.target_group_kps = self.chunk.shapes.addGroup()
            self.target_group_kps.label = self.kps_group_label
            self.target_group_kps.enabled = False

            self.roads_group = self.chunk.shapes.addGroup()
            self.roads_group.label = self.roads_group_label
            self.roads_group.enabled = False

            for widget in self.widgets_to_disable:
                widget.setEnabled(False)
            self.btnStop.setEnabled(True)
            
            self.do_export_ortho = self.checkIfExportOtho.isChecked()
            self.do_filter_points = self.checkIfFilterPoints.isChecked()
            self.do_detect_buildings = self.checkIfDetectBuildings.isChecked()
            self.use_snow_model = self.checkIfUseSnowModel.isChecked()
            self.do_visualize = self.checkIfVisualize.isChecked()
            self.do_use_patchmode = self.checkIfUsepathcMode.isChecked()
            self.do_detect_roads = self.checkIfDetectRoads.isChecked()
            
            self.filter_methods = {'Weighted': self.weighted_average, 'Mean': self.mean_points, 'Max': self.max_point}
            
            self.filter_method = self.filter_methods[self.combo.currentText()]
            
            self.points_confidance = float(self.KPSconfidance.text())
            
            if self.do_visualize and not self.do_use_patchmode:
                pathlib.Path(self.buildings_path).mkdir(parents=True, exist_ok=True)
            
            self.time_start = time.time()
            
            if self.use_snow_model:
                self.model_kps_path = self.model_kps_snow_path
            else:
                self.model_kps_path = self.model_kps_default_path

            if self.do_detect_buildings:
                if self.do_use_patchmode:
                    self.detect_buildings_Patch_mode()
                else:
                    self.detect_buildings()
            if self.do_detect_roads:
                if self.combo_road.currentText() == 'YOLOv8':
                    self.yolo_roads = True
                else:
                    self.yolo_roads = False
                self.detect_roads()
                
            self.results_time_total = time.time() - self.time_start

            self.show_results_dialog()
            
            for widget in self.widgets_to_disable:
                widget.setEnabled(True)
            self.btnStop.setEnabled(False)
        except:
            for widget in self.widgets_to_disable:
                widget.setEnabled(True)
            self.btnStop.setEnabled(False)
            if self.stopped:
                time_total = self.format_timedelta(time.time() - self.time_start)
                Metashape.app.messageBox(f"Процесс был остановлен.\nПоиск длился {time_total}")
            else:
                self.extract_points() # extract points in case of an error.
                Metashape.app.messageBox("Что-то пошло не так.\n"
                                             "Пожалуйста, проверте консоль.")
                raise
        print("Script finished.")
        return True

def start_script():
    chunk = Metashape.app.document.chunk

    if chunk is None or chunk.orthomosaic is None:
        raise Exception("No active chunks!")
    
    app = QtWidgets.QApplication.instance()
    parent = app.activeWindow()
    dlg = DetectObjectsDlg(parent)

label = "Scripts/Detect buildings"
Metashape.app.addMenuItem(label, start_script)
print("To execute this script press {}".format(label))