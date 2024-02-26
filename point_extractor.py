import Metashape
import pandas as pd
import time
import pathlib
#from PIL import Image
import os
import numpy as np
from pathlib import Path



class PointExtractor():
    
    def __init__(self):
        self.chunk = Metashape.app.document.chunk
        
        if self.chunk is None or self.chunk.orthomosaic is None:
            raise Exception("No active chunks!")
        
        self.cameras = self.chunk.cameras
        self.T = self.chunk.transform.matrix
        self.crs = self.chunk.crs
        
        if len(Metashape.app.document.path) > 0:
            self.working_dir = str(pathlib.Path(Metashape.app.document.path).parent)
        else:
            raise ValueError('Can`t find working dir!')
            
        self.save_dir = self.working_dir + '/Extracted_Points/'
        self.DEM = self.chunk.elevation

    def get_points_to_extract(self):
        points = [shape for shape in self.chunk.shapes if shape.geometry.type == Metashape.Geometry.PointType]
        return points

    def extract_point(self, point, camera, image):
        try:
            x = point.geometry.coordinates[0][0]
            y = point.geometry.coordinates[0][1]
            z = self.DEM.altitude(Metashape.Vector([x, y]))
        except:
            return
        P = self.crs.unproject(self.T.inv().mulp(Metashape.Vector([x, y, z])))
        pixels = camera.project(P)
        if pixels is None:
            pixels = Metashape.Vector([-1, -1])
        pixels = [int(pixels.x), int(pixels.y)]
        if pixels[0] > image.shape[0] or pixels[1] > image.shape[0]:
            pixels = Metashape.Vector([-1, -1])
        if all([coord > 0 for coord in pixels]):
            with open(self.save_dir + 'log.txt', 'a') as file:
                file.write(f'\t{x=}, {y=}, {z=}\n\t{P=}\n\t{pixels=}\n')
            self.points.append(pixels)

    def visualize(self, camera, image):
        import seaborn as sns

        fontsize = 18
        sns.set_style("white")
        radius = 2 + 2*int(image.shape[0] > 700) + 2*int(image.shape[1] > 700)
        colors = {0: (0, 0, 255), 1: (255, 0, 0)}
        
        print(f'Starting to vis camera: {camera.label}')
        for kps in self.points:
            if kps[0] > image.shape[0] or kps[1] > image.shape[1]:
                continue
            image = cv2.circle(image.copy(), tuple(kps), radius, colors[1], -1)
                    
        cv2.imwrite(self.save_dir + f'{camera.label}_extr.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    def create_annotation_file(self, camera, image):
        width, height = image.shape[1], image.shape[0]
        with open(self.save_dir + f'annotations/{camera.label}_extr.txt', 'w') as annotation:
            for point in self.points:
                x_c, y_c = [a/b for a, b in zip(point, [width, height])]
                w, h = [30/a for a in [width, height]]
                annotation.write(f'0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n')
            
    def process(self):
        pathlib.Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.save_dir + 'annotations/').mkdir(parents=True, exist_ok=True)

        points = self.get_points_to_extract()

        with open(self.save_dir + 'log.txt', 'w') as file:
            file.write('Logging...\n\n')
        
        for camera in self.cameras:
            image = Path(camera.photo.path)
            if not image.is_file():
                continue
            
            image = cv2.imdecode(np.fromfile(camera.photo.path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.points = []
            
            with open(self.save_dir + 'log.txt', 'a') as file:
                file.write(f'Camera: {camera.label}\n') 
            
            for point in points:
                self.extract_point(point, camera, image)
            
            self.visualize(camera, image)
            self.create_annotation_file(camera, image)
            
            with open(self.save_dir + 'log.txt', 'a') as file:
                file.write(f'\n\n\n')
        
def start_script():
    try:
        extractor = PointExtractor()
        extractor.process()
    except:
        Metashape.app.messageBox("Что-то пошло не так.\nПожалуйста, проверте консоль.")
        raise
        

label = "Scripts/Extract points"
Metashape.app.addMenuItem(label, start_script)
print("To execute this script press {}".format(label))