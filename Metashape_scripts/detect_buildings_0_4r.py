from PySide2 import QtGui, QtCore, QtWidgets

import urllib.request, tempfile
temporary_file = tempfile.NamedTemporaryFile(delete=False)
find_links_file_url = "https://raw.githubusercontent.com/agisoft-llc/metashape-scripts/master/misc/links.txt"
urllib.request.urlretrieve(find_links_file_url, temporary_file.name)    

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


class DetectObjectsDlg(QtWidgets.QDialog):

    def __init__(self, parent):

        self.group_label = 'Обнаруженные здания'
        self.kps_group_label = 'Обнаруженные точки'
        
        if len(Metashape.app.document.path) > 0:
            self.working_dir = str(pathlib.Path(Metashape.app.document.path).parent)
        else:
            self.working_dir = ""
            
        self.bottom = Metashape.Elevation.bottom
        
        self.major_version = float(".".join(Metashape.app.version.split('.')[:2]))
            
        self.current_image = ''
        
        self.ortho_path = self.working_dir + '/Orthomosaic/'
        self.do_export_ortho = False
        self.do_filter_points = True
        self.do_detect_buildings = True
        self.model_path = self.working_dir + '/NN_models/Building_detection_model.pth'
        self.model_kps_default_path = self.working_dir + '/NN_models/Keypoints_detection_model.pth'
        self.model_kps_snow_path = self.working_dir + '/NN_models/Keypoints_detection_model_snow.pth'
        
        self.buildings_path = 'C:/Users/User/Desktop/Buildings/'#self.working_dir + '/Buildings/'  C:\Users\User\Desktop\Teobox_Kedrovka_2023-06-06T11.31.53
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        #self.device = torch.device('cpu')
        
        self.use_snow_model = False
        self.do_visualize = False
        
        self.predicted_points = []
        self.predicted_points_scores = []
        
        self.chunk = Metashape.app.document.chunk
        self.ortho_crs = self.chunk.orthomosaic.crs
        self.num_keypoints = 3

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
        
        self.checkIfDetectBuildings = QtWidgets.QCheckBox("Искать здания")
        self.checkIfDetectBuildings.setChecked(self.do_detect_buildings)
        self.txtDetectBuildings = QtWidgets.QLabel()
        self.txtDetectBuildings.setText("Если отмечено, будет произведен поиск зданий.")
        
        self.txtModelLoadPath = QtWidgets.QLabel()
        self.txtModelLoadPath.setText("Путь к модели для поиска зданий:")
        self.edtModelLoadPath = QtWidgets.QLineEdit()
        self.edtModelLoadPath.setText(self.model_path)
        self.edtModelLoadPath.setPlaceholderText("Файл с моделью для поиска зданий")
        self.edtModelLoadPath.setToolTip("Файл с моделью для поиска зданий")
        self.btnModelLoadPath = QtWidgets.QPushButton("...")
        self.btnModelLoadPath.setFixedSize(25, 25)
        QtCore.QObject.connect(self.btnModelLoadPath, QtCore.SIGNAL("clicked()"), lambda: self.choose_building_model_load_path())

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
        
        self.txtKPSModelSnowLoadPath = QtWidgets.QLabel()
        self.txtKPSModelSnowLoadPath.setText("Путь к зимней модели для поиска точек:")
        self.edtKPSModelSnowLoadPath = QtWidgets.QLineEdit()
        self.edtKPSModelSnowLoadPath.setText(self.model_kps_snow_path)
        self.edtKPSModelSnowLoadPath.setPlaceholderText("Файл с зимней моделью для поиска точек")
        self.edtKPSModelSnowLoadPath.setToolTip("Файл с зимней моделью для поиска точек")
        self.btnKPSModelSnowLoadPath = QtWidgets.QPushButton("...")
        self.btnKPSModelSnowLoadPath.setFixedSize(25, 25)
        QtCore.QObject.connect(self.btnKPSModelSnowLoadPath, QtCore.SIGNAL("clicked()"), lambda: self.choose_points_snow_model_load_path())
        
        self.checkIfVisualize = QtWidgets.QCheckBox("Записывать предсказания на фото")
        self.checkIfVisualize.setToolTip("Записывать предсказания модели на фото зданий в отдельную папку")
        self.checkIfVisualize.setChecked(self.do_visualize)
        
        self.checkIfUseSnowModel = QtWidgets.QCheckBox("Использовать модель для зимы (BETA)")
        self.checkIfUseSnowModel.setToolTip("Использовать, если съемка проводилась в зимнее время")
        self.checkIfUseSnowModel.setChecked(self.use_snow_model)
        self.checkIfUseSnowModel.setEnabled(False)
        
        self.checkIfFilterPoints = QtWidgets.QCheckBox("Фильтровать точки")
        self.checkIfFilterPoints.setChecked(self.do_filter_points)
        
        self.fill= QtWidgets.QLabel()
        self.fill.setText("                                                            .                                                                             ")
        
        self.combo = QtWidgets.QComboBox()
        self.combo.addItem('Weighted')
        self.combo.addItem('Mean')
        self.combo.addItem('Max')
        
        self.filterKPSdistance = QtWidgets.QLineEdit()
        self.filterKPSdistance.setText('0.5')
        self.filterKPSdistance.setPlaceholderText("[0.1, 1]")
        self.filterKPSdistance.setToolTip("Расстояние фильтрации")
        
        
        
        self.KPSconfidanceText = QtWidgets.QLabel()
        self.KPSconfidanceText.setText("Уровень уверенности точек")
        
        self.comboText = QtWidgets.QLabel()
        self.comboText.setText("Метод фильтрации")
        
        self.KPSconfidance = QtWidgets.QLineEdit()
        self.KPSconfidance.setText('10.0')
        self.KPSconfidance.setToolTip("Уровень уверенности точек")
        
        
        KPSLoadLayout.addWidget(self.txtModelLoadPath, 0, 0)
        KPSLoadLayout.addWidget(self.edtModelLoadPath, 0, 1)
        KPSLoadLayout.addWidget(self.btnModelLoadPath, 0, 2)
        
        
        KPSLoadLayout.addWidget(self.txtKPSModelLoadPath, 1, 0)
        KPSLoadLayout.addWidget(self.edtKPSModelLoadPath, 1, 1)
        KPSLoadLayout.addWidget(self.btnKPSModelLoadPath, 1, 2)
        
        KPSLoadLayout.addWidget(self.txtKPSModelSnowLoadPath, 2, 0)
        KPSLoadLayout.addWidget(self.edtKPSModelSnowLoadPath, 2, 1)
        KPSLoadLayout.addWidget(self.btnKPSModelSnowLoadPath, 2, 2)
        
        KPSLoadLayout.addWidget(self.checkIfUseSnowModel, 3, 0)
        
        KPSLoadLayout.addWidget(self.checkIfVisualize, 3, 1)
        
        KPSLoadLayout.addWidget(self.checkIfFilterPoints, 4, 0)
        
        KPSLoadLayout.addWidget(self.filterKPSdistance, 4, 1)
        
        KPSLoadLayout.addWidget(self.comboText, 5, 0)
        
        KPSLoadLayout.addWidget(self.combo, 5, 1)
        
        KPSLoadLayout.addWidget(self.KPSconfidanceText, 6, 0)
        
        KPSLoadLayout.addWidget(self.KPSconfidance, 6, 1)
        

        self.groupBoxKPSModelLoad.setLayout(KPSLoadLayout)

        self.btnRun = QtWidgets.QPushButton(f"Пуск (Используется {self.device})")
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

    def choose_ortho_dir(self):
        working_dir = Metashape.app.getExistingDirectory()
        self.edtWorkingDir.setText(working_dir)
        self.ortho_path = self.edtWorkingDir.text()

    
    def draw_shape_point(self, label: str, coordinates: Metashape.Vector):
        if len(coordinates) == 0:
            print('None in draw point')
            return None
        new_shape = self.chunk.shapes.addShape()
        new_shape.label = label
        new_shape.group = self.target_group_kps
        new_shape.geometry = Metashape.Geometry.Point(coordinates)

    def draw_shape_point_1_6(self, label: str, coordinates: Metashape.Vector):
        if len(coordinates) == 0:
            print('None in draw point')
            return None
        shape = self.chunk.shapes.addShape()
        shape.label = label
        shape.type = Metashape.Shape.Type.Point
        shape.group = self.target_group_kps
        shape.vertices = [coordinates]
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
        predicted_points = self.predicted_points
        scores = self.predicted_points_scores
        if self.do_filter_points:
            time_filter = time.time()
            predicted_points, scores = self.filter_points(self.predicted_points, self.predicted_points_scores)
            print(f'Points filtered for {round(time.time() - time_filter, 2)} sec. Method: {self.combo.currentText()}')
        else:
            print('No filter applyied')
        print(len(predicted_points), len(scores))
        for point, score in zip(predicted_points, scores):
            score = round(float(score), 2)
            if self.major_version > 1.7:
                self.draw_shape_point(label=f'Point_{score}', coordinates=point)
            else:
                self.draw_shape_point_1_6(label=f'Point_{score}', coordinates=point)

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
        xyz = [[point[0], point[1], point[2]] for point in points]
        weights = [score[3] for score in points]
        average = np.average(xyz, axis=0, weights=weights).tolist()
        weights = np.mean(weights).tolist()
        average.append(weights)
        return tuple(average)

    def mean_points(self, points):
        import numpy as np
        return tuple(np.mean(points, axis=0).tolist())

    def max_point(self, points):
        return max(points, key=lambda x: x[3])

    def is_close(self, point1, point2):
        self.close_distance = float(self.filterKPSdistance.text())
        return (abs(point1[0] - point2[0]) < self.close_distance) and (abs(point1[1] - point2[1]) < self.close_distance)

    def filter_points(self, points: Metashape.Vector, scores):
    
        points_tuple = [(point.x, point.y, point.z, score) for point, score in zip(points, scores)]
        final_points = set()
    
        for point in points_tuple:
            filtered_points = filter(lambda x: self.is_close(x, point), points_tuple)
            filtered_points = list(filtered_points)
            if len(filtered_points) >= 1:
                final_points.add(self.filter_method(list(filtered_points)))
        
        filtered_vectors = ([Metashape.Vector([point[0], point[1], point[2]]) for point in list(final_points)], [point[3] for point in list(final_points)])
        return filtered_vectors

    
    def detect_buildings(self):
        model = self.load_model()
        print('Model loaded')
        time_start = time.time()
        self.predict_images(model=model)

    def detect_buildings_Patch_mode(self):
        #Create folder for export ortho
        pathlib.Path(self.ortho_path).mkdir(parents=True, exist_ok=True)
        
        self.export_ortho()

        self.predict_buildings()

        self.treat_buildings()

    def predict_buildings(self):
        model = self.load_model()
        app = QtWidgets.QApplication.instance()
        
        classes = {'Building': 1}
        inv_classes = {value:key for key, value in classes.items()}    
        model.to(self.device)
        
        for num, image_name in enumerate([file for file in os.listdir(self.ortho_path) if not file.endswith('.tfw')]): #Iterate over orthophotos ignoring world files
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
            x.to(self.device)        
            predictions = model([x,])        
            pred = predictions[0]
            
            scores_valid = pred['scores'] > 0.95 # filter out bad predictions
            
            target = {}
            target['boxes'] = pred['boxes'][scores_valid]
            target['labels'] = pred['labels'][scores_valid]
            target['scores'] = pred['scores'][scores_valid]
            target = {key: value.to(torch.device('cpu')) for key, value in target.items()}
            
            df = pd.DataFrame(data=pred['boxes'], columns=['xmin', 'ymin', 'xmax', 'ymax'])
            df['label'] = [inv_classes[label.item()] + f': {round(score.item(), 2)}' for label, score in zip(target['labels'], target['scores'])]
            
            self.add_buildings(world_file_path=world_file_path, predictions=df)

    def draw_shape(self, label: str, corners: list):
        '''
        Создает на ортофотоплане фигуру полигон с именем 'label', в слое 'group' по координатам узлов в 'corners' (shapes.crs!) 
        '''
        if len(coordinates) == 0:
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
            shape.vertices = [coordinates]
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
                p = Metashape.Vector([x, y])
                p = Metashape.CoordinateSystem.transform(p, self.chunk.orthomosaic.crs, self.chunk.shapes.crs)
                corners.append([p.x, p.y])
            self.draw_shape(label=label, corners=corners)
    
    def treat_buildings(self):
        pass
        
    def export_ortho(self):
        if not self.ortho_path > 0:
            raise InterruptedError("Specify path to export orthomosaic!")     
        self.chunk.exportRaster(path=self.ortho_path + '/ortho.tif', source_data=Metashape.OrthomosaicData, image_format=Metashape.ImageFormat.ImageFormatJPEG, save_alpha=True, white_background=True,
                                save_world=True,
                                split_in_blocks=True, block_width=5000, block_height=5000,)
         
    def format_timedelta(self, td):
        minutes, seconds = divmod(td, 60)
        hours, minutes = divmod(minutes, 60)
        return '{:02.0f}:{:02.0f}:{:02.0f}'.format(hours, minutes, seconds)

    def show_results_dialog(self):
        time_total = self.format_timedelta(self.results_time_total)
        message = f"Поиск завершен за {time_total}"
        print(message)
        Metashape.app.messageBox(message)
            
    def process(self):    
        try:
            self.stopped = False

            self.target_group = self.chunk.shapes.addGroup()
            self.target_group.label = self.group_label
            self.target_group.enabled = False
            
            self.target_group_kps = self.chunk.shapes.addGroup()
            self.target_group_kps.label = self.kps_group_label
            self.target_group_kps.enabled = False

            self.btnRun.setEnabled(False)
            self.combo.setEnabled(False)
            self.btnStop.setEnabled(True)
            
            self.do_export_ortho = self.checkIfExportOtho.isChecked()
            self.do_filter_points = self.checkIfFilterPoints.isChecked()
            self.do_detect_buildings = self.checkIfDetectBuildings.isChecked()
            self.use_snow_model = self.checkIfUseSnowModel.isChecked()
            self.do_visualize = self.checkIfVisualize.isChecked()
            
            self.filter_methods = {'Weighted': self.weighted_average, 'Mean': self.mean_points, 'Max': self.max_point}
            
            self.filter_method = self.filter_methods[self.combo.currentText()]
            
            self.points_confidance = float(self.KPSconfidance.text())
            
            if self.do_visualize:
                pathlib.Path(self.buildings_path).mkdir(parents=True, exist_ok=True)
            
            self.time_start = time.time()
            
            if self.use_snow_model:
                self.model_kps_path = self.model_kps_snow_path
            else:
                self.model_kps_path = self.model_kps_default_path
            
            if self.do_detect_buildings:
                self.detect_buildings()
                

            self.results_time_total = time.time() - self.time_start

            self.show_results_dialog()
            
            self.btnRun.setEnabled(True)
            self.combo.setEnabled(True)
            self.btnStop.setEnabled(False)
        except:
            self.btnRun.setEnabled(True)
            self.combo.setEnabled(True)
            self.btnStop.setEnabled(False)
            if self.stopped:
                time_total = self.format_timedelta(time.time() - self.time_start)
                Metashape.app.messageBox(f"Процесс был остановлен.\nПоиск длился {time_total}")
            else:
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