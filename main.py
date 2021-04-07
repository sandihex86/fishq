import sys
import cv2
import depthai as dai
from pathlib import Path
import time
import numpy as np

'''
YoloTiny CNN arsitektur

'''


print("DEPTHAI VERSION",dai.__version__)
print("OPENCV",cv2.__version__)
#create piple
pipeline = dai.Pipeline()

#color camera
camRgb = pipeline.createColorCamera()
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setPreviewSize(416,416)
camRgb.setInterleaved(False)
camRgb.setFps(24)

xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("rgb")


#monocamera
monoRight = pipeline.createMonoCamera()
monoLeft = pipeline.createMonoCamera()
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)

xoutLeft = pipeline.createXLinkOut()
xoutLeft.setStreamName("monoleft")
xoutRight = pipeline.createXLinkOut()
xoutRight.setStreamName("monoright")

monoRight.out.link(xoutLeft.input)
monoLeft.out.link(xoutRight.input)

#stereo depth
stereo = pipeline.createStereoDepth()
stereo.setOutputDepth(True)
stereo.setConfidenceThreshold(255)
median = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7
stereo.setMedianFilter(median)

xoutDepth = pipeline.createXLinkOut()
xoutDepth.setStreamName("depth")
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)


# Model Detection
nnPath = str((Path(__file__).parent / Path('models/tiny-yolo-v4_openvino_2021.2_6shave.blob')).resolve().absolute())
print(nnPath)
labelMap = [
    "person",         "bicycle",    "car",           "motorbike",     "aeroplane",   "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",   "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",       "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",    "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball", "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",      "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",      "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",       "donut",         "cake",
    "chair",          "sofa",       "pottedplant",   "bed",           "diningtable", "toilet",        "tvmonitor",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",  "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",       "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"
]


detectionNet = pipeline.createYoloSpatialDetectionNetwork()
detectionNet.setConfidenceThreshold(0.5)
detectionNet.setNumClasses(80)
detectionNet.setCoordinateSize(4)
detectionNet.setAnchors(np.array([10,14, 23,27, 37,58, 81,82, 135,169, 344,319]))
detectionNet.setAnchorMasks({"side26": np.array([1, 2, 3]), "side13": np.array([3, 4, 5])})
detectionNet.setIouThreshold(0.5)
detectionNet.setBlobPath(nnPath)
detectionNet.setNumInferenceThreads(2)
detectionNet.input.setBlocking(False)
camRgb.preview.link(detectionNet.input)

syncNN = True
if(syncNN):
    detectionNet.passthrough.link(xoutRgb.input)
    print("Stream from passthrough CNN .....")
else:
    camRgb.preview.link(xoutRgb.input)
    print("Stream from camera rgb......")

xoutDet = pipeline.createXLinkOut()
xoutDet.setStreamName("detection")
xoutBbDepth = pipeline.createXLinkOut()
xoutBbDepth.setStreamName("bbDepth")

stereo.depth.link(detectionNet.inputDepth)
detectionNet.out.link(xoutDet.input)
detectionNet.passthroughDepth.link(xoutDepth.input)
detectionNet.boundingBoxMapping.link(xoutBbDepth.input)

with dai.Device(pipeline) as device:
    device.startPipeline()

    qLeft = device.getOutputQueue(name="monoleft", maxSize=4, blocking=False)
    qRight = device.getOutputQueue(name="monoright", maxSize=4, blocking=False)
    qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    qBbDepth = device.getOutputQueue(name="bbDepth", maxSize=4, blocking=False)
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="detection", maxSize=4, blocking=False)

    def printResult(frame,inDet):
        height = frame.shape[0]
        weight = frame.shape[1]
        detections = inDet.detections
        for detection in detections:
            print("FISH-Q DETECTED:",
                  str(labelMap[detection.label])," :",
                  "Conf.{:.2f}".format(detection.confidence*100)," :",
                  f"X: {int(detection.spatialCoordinates.x)} mm"," :",
                  f"Y: {int(detection.spatialCoordinates.y)} mm"," :",
                  f"Z: {int(detection.spatialCoordinates.z)} mm"," :")

    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox),frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox),0,1)*normVals).astype(int)

    def displayFrame(name,frame):
        for detection in detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(255,255,0),1)
            cv2.putText(frame,labelMap[detection.label],(bbox[0]+10,bbox[1]+20),cv2.FONT_HERSHEY_TRIPLEX, 0.5,(255,255,0))
            cv2.putText(frame,"X:{:.1f}".format(bbox[0]),(bbox[0]+10,bbox[1]+40),cv2.FONT_HERSHEY_TRIPLEX, 0.5,(255,255,0))
            cv2.putText(frame,"Y:{:.1f}".format(bbox[1]),(bbox[0]+10,bbox[1]+60),cv2.FONT_HERSHEY_TRIPLEX, 0.5,(255,255,0))
            cv2.imshow(name,frame)
    
    def displayDepth(name,frame):
        depthframeColor = cv2.normalize(frame,None,255,0,cv2.NORM_INF,cv2.CV_8UC1)
        depthframeColor = cv2.equalizeHist(depthframeColor)
        depthframeColor = cv2.applyColorMap(depthframeColor,cv2.COLORMAP_HOT)

        cv2.imshow(name,depthframeColor)
    
    frame = None
    counter = 0
    startTime = time.monotonic()
    color = (255,255,255)

    while True:
       inRgb = qRgb.get()
       inDet = qDet.get()
       inDepth = qDepth.get()
       frameDepth = inDepth.getFrame()
       frame = inRgb.getCvFrame()
       detections = inDet.detections
       printResult(frame,inDet)
       counter+=1

       if len(detections)!=0:
           bbMapping = qBbDepth.get()
           roiDatas = bbMapping.getConfigData()

           for roiData in roiDatas:
               roi = roiData.roi
               roi = roi.denormalize(frameDepth.shape[1],frameDepth.shape[0])
               topLeft = roi.topLeft()
               bottomRight = roi.bottomRight()
               xmin = int(topLeft.x)
               ymin = int(topLeft.y)
               xmax = int(bottomRight.x)
               ymax = int(bottomRight.y)
               cv2.putText(frameDepth,"ROI", (xmin,ymin), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255,255,255))
               cv2.rectangle(frameDepth,(xmin,ymin),(xmax,ymax), (255,255,255), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

       

       cv2.putText(frame,"FISH-QI sensing: {:.2f}".format(counter / (time.monotonic() - startTime)),
                   (2,frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.3, color = (255,255,255))
              #display detections
       displayFrame("FISHQ",frame)
       displayDepth("FISHQ-DEPTH",frameDepth)
       
       if cv2.waitKey(1) == ord("q"):
            break