"""YOLO v3 output
"""
import numpy as np
import cv2

class YOLO:
    def __init__(self, cfg, wts):
        """Init.

        # Arguments
            cfg: YOLOv4tiny configuration file.
            wts: YOLOv4tiny weights file.
        """
        
        # load the YOLOv4 model into OpenCV (in this case, YOLOv4tiny)
        self.net = cv2.dnn.readNetFromDarknet(cfg, wts)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

        # get a list of all the layers used in the neural network (254 components)
        # https://stackoverflow.com/questions/57706412/what-is-the-working-and-output-of-getlayernames-and-getunconnecteddoutlayers
        self.ln = self.net.getLayerNames()
        # index the output layers into an array
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]


    def predict(self, image, shape):
        """Detect the objects with yolo.
            Adapted from this tutorial (Identify Objects):
            https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html


        # Arguments
            image: ndarray, processed input image.
            shape: shape of original image.

        # Returns
            boxes: ndarray, boxes of objects.
            classes: ndarray, classes of objects.
            scores: ndarray, scores of objects.
        """

        # Alter image so it can be read into the model
        # Blob is a 4D numpy array object made of:
        #   the images, 
        #   the color channels, 
        #   the image width, and 
        #   the image height
        # Params: image, scaled pixel-values, (width, height), swap RB in RGB, do not crop
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

        # Calculate the neural network response 
        # outputs are vectors with 85 values each:
        #   4 bounding box points (center x/y, and width/height)
        #   1 valuefor box confidence
        #   80 values for class confidence
        self.net.setInput(blob)
        outputs = self.net.forward(self.ln)

        
        # setup vars
        boxes = []
        confidences = []
        classIDs = []
        h, w = image.shape[:2]

        # for each output from the neural network response
        for output in outputs:
            for detection in output:
                # get highest scoring class, and it's score (i.e. confidence)
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # set threshold of what to allow YOLO to classify through confidence value
                if(confidence > 0.2):# and classID == 0): 
                    # define center x/y, and width/height, scaled by image w/h
                    box = detection[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    box = [x, y, int(width), int(height)]
                    boxes.append(box)
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        if(len(confidences) > 0):
            best = confidences.index(max(confidences))
            return [boxes[best]], [classIDs[best]], [confidences[best]]
        
        return boxes, classIDs, confidences
