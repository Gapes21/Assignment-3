from darknet import load_meta, load_net, detect
import numpy as np
import cv2
import os
import time

class YOLOV2Detector:
    def __init__(self, confPath:os.PathLike="cfg/yolov2.cfg", weightsPath:os.PathLike="yolov2.weights", metaPath:os.PathLike="cfg/coco.data", outputDir:os.PathLike="yolo_output", tempDir:os.PathLike=".yolo_temp"):
        self.confPath = confPath.encode()
        self.weightsPath = weightsPath.encode()
        self.metaPath = metaPath.encode()
        self.outputDir = outputDir
        self.tempDir = tempDir

        self.net = load_net(self.confPath, self.weightsPath, 0)
        self.meta = load_meta(self.metaPath)

    def detect(self, image:np.ndarray, thresh=.5, hier_thresh=.5, nms=.45):
        tempImgPath = self.saveImg(image, temp=True)
        result = detect(self.net, self.meta, tempImgPath.encode(), thresh, hier_thresh, nms)
        resDict = {"labels": [], "scores": [], "boxes": []}
        for res in result:
            resDict['labels'].append(res[0].decode())
            resDict['scores'].append(res[1])
            resDict['boxes'].append(res[2])
        # Boxes of the format (x, y, w, h)
        self.delImgTemp(tempImgPath)
        return resDict

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