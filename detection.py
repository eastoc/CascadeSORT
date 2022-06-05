# Author: Fangdong Wu
# Detecting objects on one picture via YOLO-v3
import cv2
import numpy as np
import copy

class Candidates:
    def __init__(self, bbox, img_crop):
        self.tlwh = bbox
        self.img_crop = img_crop
        self.convert()

    def convert(self):
        """
        convert tlwh to tlbr
        tlwh: np.array, type=int, [x, y, w, h]
        tlbr: np.array, typr=int, [top-left.x, top-left.y, bottom-right.x, bottom-right.y]
        """
        self.tlbr = np.empty((4,), dtype=int)
        self.tlbr[0] = self.tlwh[0]  # left line
        self.tlbr[1] = self.tlwh[1]  # top line
        self.tlbr[2] = self.tlwh[0] + self.tlwh[2]  # right line = left + width
        self.tlbr[3] = self.tlwh[1] + self.tlwh[3]  # bottom line = top + height

    def update(self):
        """
        convert tlbr to tlwh
        tlwh: np.array, type=int, [x, y, w, h]
        tlbr: np.array, typr=int, [top-left.x, top-left.y, bottom-right.x, bottom-right.y]
        """
        self.tlwh[0] = self.tlbr[0]  # left line
        self.tlwh[1] = self.tlbr[1]  # top line
        self.tlwh[2] = self.tlbr[2] - self.tlbr[0]  # width = right - left
        self.tlwh[3] = self.tlbr[3] - self.tlbr[1]  # height = bottom - top


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def detector(tracker):
    #classes = None

    #COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    net = tracker.net
    frame = tracker.frame
    classes = tracker.classes
    #scale = 0.00392
    scale = 1/255
    blob = cv2.dnn.blobFromImage(frame, scale, (608, 608), (0, 0, 0), True, crop=False)
    Width = frame.shape[1]
    Height = frame.shape[0]
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    class_ids = []#lable 编号
    confidences = []#置信度
    boxes = []#bboxes矩阵，x,y,w,h
    conf_threshold = tracker.conf_threshold
    nms_threshold = tracker.nms_threshold
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence >= conf_threshold:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2 # 左下角位置
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])#bbox类矩阵，每个bbox4个参数[x，y，w，h]

    # 通过NMS对bboxes筛选得到最终bboxes的id，即indice，是一个向量
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    img_crop = []
    candidates = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        img_crop = copy.deepcopy(frame[round(y):round(y+h), round(x):round(x+w)])
        temp = Candidates(box, img_crop)
        revise_cross_border(temp, frame)
        candidates.append(temp)
    return candidates

def revise_cross_border(candidate, frame):
    flag = 0
    if candidate.tlbr[0] < 0:
        candidate.tlbr[0] = 0
        flag = 1
    if candidate.tlbr[1] < 0:
        candidate.tlbr[1] = 0
        flag = 1
    if candidate.tlbr[2] >= frame.shape[1]:
        candidate.tlbr[2] = frame.shape[1] - 1
        flag = 1
    if candidate.tlbr[3] >= frame.shape[0]:
        candidate.tlbr[3] = frame.shape[0] - 1
        flag = 1
    if flag == 1:
        candidate.update()
        [x, y, w, h] = candidate.tlwh
        candidate.img_crop = copy.deepcopy(frame[int(y):int(y + h), int(x):int(x + w)])