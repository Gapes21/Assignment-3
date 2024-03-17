from darknet import load_meta, load_net, detect
import numpy as np
import cv2
import os
import time

class Detector:
    def __init__(self):
        pass
    def getDetections(self, frame):
        pass
    def getDetectionsSORT(self, frame):
        pass
        
class DetectorYOLO(Detector):
    def __init__(self, confPath:os.PathLike="cfg/yolov2.cfg", weightsPath:os.PathLike="yolov2.weights", metaPath:os.PathLike="cfg/coco.data", outputDir:os.PathLike="yolo_output", tempDir:os.PathLike=".yolo_temp"):
        self.confPath = confPath.encode()
        self.weightsPath = weightsPath.encode()
        self.metaPath = metaPath.encode()
        self.outputDir = outputDir
        self.tempDir = tempDir

        self.net = load_net(self.confPath, self.weightsPath, 0)
        self.meta = load_meta(self.metaPath)
    
    def getDetections(self, image:np.ndarray, thresh=.5, hier_thresh=.5, nms=.45):
        tempImgPath = self.saveImg(image, temp=True)
        result = detect(self.net, self.meta, tempImgPath.encode(), thresh, hier_thresh, nms)
        boxes, scores, labels = [], [], []
        for res in result:
            labels.append(res[0].decode())
            scores.append(res[1])
            boxes.append(res[2])
        self.delImgTemp(tempImgPath)
        detections = [(box, score, label) for box, score, label in zip(boxes, scores, labels) if score > 0.8]
        return detections
    
    def getDetectionsSort(self, image:np.ndarray, thresh=.5, hier_thresh=.5, nms=.45):
        tempImgPath = self.saveImg(image, temp=True)
        result = detect(self.net, self.meta, tempImgPath.encode(), thresh, hier_thresh, nms)
        boxes, scores, labels = [], [], []
        for res in result:
            labels.append(res[0].decode())
            scores.append(res[1])
            boxes.append(res[2])
        self.delImgTemp(tempImgPath)
        detections = [np.array([box[0], box[1], box[2], box[3], score]) for box, score, label in zip(boxes, scores, labels) if score > 0.8 and label in self.vehicle_classes]
        return np.array(detections)
    
    def saveImg(self, image:np.ndarray, temp:bool=False):
        dir = self.tempDir if temp else self.outputDir
        if not os.path.isdir(dir):
            os.mkdir(dir)
        tempImgPath = os.path.join(dir, str(int(time.time()*1e7))+'.jpg')
        cv2.imwrite(tempImgPath, image)
        return tempImgPath

    def delImgTemp(self, imgPath:os.PathLike):
        os.remove(imgPath)
        if len(os.listdir(self.tempDir))==0:
            os.rmdir(self.tempDir)
        
        
        
        
# class YOLOV2Detector:
#     def __init__(self, confPath:os.PathLike="cfg/yolov2.cfg", weightsPath:os.PathLike="yolov2.weights", metaPath:os.PathLike="cfg/coco.data", outputDir:os.PathLike="yolo_output", tempDir:os.PathLike=".yolo_temp"):
#         self.confPath = confPath.encode()
#         self.weightsPath = weightsPath.encode()
#         self.metaPath = metaPath.encode()
#         self.outputDir = outputDir
#         self.tempDir = tempDir
#         self.net = load_net(self.confPath, self.weightsPath, 0)
#         self.meta = load_meta(self.metaPath)

#     def detect(self, image:np.ndarray, thresh=.5, hier_thresh=.5, nms=.45):
#         tempImgPath = self.saveImg(image, temp=True)
#         result = detect(self.net, self.meta, tempImgPath.encode(), thresh, hier_thresh, nms)
#         resDict = {"labels": [], "scores": [], "boxes": []}
#         for res in result:
#             resDict['labels'].append(res[0].decode())
#             resDict['scores'].append(res[1])
#             resDict['boxes'].append(res[2])
#         # Boxes of the format (x, y, w, h)
#         self.delImgTemp(tempImgPath)
#         return resDict

#     def saveImg(self, image:np.ndarray, temp:bool=False):
#         dir = self.tempDir if temp else self.outputDir
#         if not os.path.isdir(dir):
#             os.mkdir(dir)
#         tempImgPath = os.path.join(dir, str(int(time.time()*1e7))+'.jpg')
#         cv2.imwrite(tempImgPath, image)
#         return tempImgPath

#     def delImgTemp(self, imgPath:os.PathLike):
#         os.remove(imgPath)
#         if len(os.listdir(self.tempDir))==0:
#             os.rmdir(self.tempDir)
            