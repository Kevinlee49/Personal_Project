#-*- encoding: utf8 -*-
import json
import math
import cv2
import numpy as np
import os
import datetime as dt
import time
import torch
import tensorflow as tf
import argparse
from scipy import ndimage
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import L2
from tensorflow.keras.layers import MaxPool2D, Conv2D, Dense, Dropout, Activation, Flatten, BatchNormalization
import matplotlib.pyplot as plt
# from models.experimental import attempt_load
# from utils.datasets import letterbox
# from utils.general import non_max_suppression, scale_coords
import csv
##################for gpu memory (using tf)####################################
#################################################################################
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#################################################################################


def decimal_process(b, nbox, classes):

    y_min, y_max = b[1], b[3]

    #y_min = min(nbox[0][1], nbox[1][1])

    #y_max = max(nbox[0][3], nbox[1][3])

    lower_limit = ((y_min + y_max) / 2) - (y_max - y_min) * 0.15
    upper_limit = ((y_min + y_max) / 2) + (y_max - y_min) * 0.15

    box_center_y_last = (nbox[-1][1] + nbox[-1][3])/2
    box_center_y_last2 = (nbox[-2][1] + nbox[-2][3])/2

    if (classes[-1] == 0) and (box_center_y_last > upper_limit):
        if box_center_y_last2 > upper_limit:
            classes[-2] += -1
            classes[-1] = 9 #9.5

        elif box_center_y_last2 < upper_limit:
            classes[-1] = 9 #9.5

    elif (classes[-1] == 9) and (box_center_y_last < lower_limit):

        if box_center_y_last2 > upper_limit:
            classes[-2] += -1
            classes[-1] = 9 #9.5

        elif box_center_y_last2 < upper_limit:
            classes[-1] = 9 #9.5

    else:
        if box_center_y_last > upper_limit:
            if classes[-1] == 0:
                classes[-1] = 9 #9.5

            else:
                classes[-1] -= 1 #0.5

        elif box_center_y_last < lower_limit:
            classes[-1] += 1 #0.5
    return classes


def check_double(boxes, scores, classes):
    pick = []
    previous_box = [boxes[0], scores[0]]
    for i, current_box in enumerate(zip(boxes, scores)):
        if i == 0:
            pick.append(i)
            continue

        x1, x2 = previous_box[0][0], previous_box[0][2]
        xx1, xx2 = current_box[0][0], current_box[0][2]
        overlap = max(max((xx2-x1), 0) - max((xx2-x2), 0) - max((xx1-x1), 0), 0) != 0

        if overlap:
            pick.pop()
            if previous_box[1] > current_box[1]:
                pick.append(i-1)
            else:
                pick.append(i)
        else:
            pick.append(i)
        previous_box = current_box

    return [boxes[j] for j in pick], [scores[j] for j in pick], [classes[j] for j in pick]





class NpEncoder(json.JSONEncoder):

    def default(self, obj):

        if isinstance(obj, np.integer):

            return int(obj)

        elif isinstance(obj, np.floating):

            return float(obj)

        elif isinstance(obj, np.ndarray):

            return obj.tolist()

        else:

            return super(NpEncoder, self).default(obj)





def horizontal_balance(img):

    img_before = img
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.blur(img_gray, (3,3))
    img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(img_edges, 2, math.pi / 180.0, 50, minLineLength=70, maxLineGap=5)
    lines_sorted = []

    if lines is not None:
        for [[x1, y1, x2, y2]] in lines:
            if (x1 == x2):
                continue
            else:
                lines_sorted.append([x1, y1, x2, y2])

        angles = []

        for [x1, y1, x2, y2] in lines_sorted:
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if ((angle <= 50) and (1 < angle)) or ((-50<= angle) and ( angle < -1)):
                angles.append(angle)

        if len(angles) == 0:
            img_rotated, median_angle = img, None

        else:
            median_angle = np.median(angles)
            img_rotated = ndimage.rotate(img_before, median_angle)

    else:
        img_rotated, median_angle = img, None

    return img_rotated, median_angle





def read_text_file_by_line(path):

    with open(path, mode='r', encoding='utf-8') as f:

        for i in f:

            i = i.strip()

            if i:

                yield i



def parse_categories_file(csv_path):

    result = {}

    for line in read_text_file_by_line(csv_path):

        split = line.split(',')

        if len(split) < 2 or any(not i for i in split):

            continue

        result[split[0]] = int(split[1])

    return result





def sort_np_array(x, column=None, flip=False):

    x = x[np.argsort(x[:, column])]

    if flip:

        x = np.flip(x, axis=0)

    return x





def make_classification_model(IM_SHAPE, NC):

    model = Sequential()

    # Block 1
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=L2(l2=0.01), input_shape=IM_SHAPE))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=L2(l2=0.01)))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=L2(l2=0.01)))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(BatchNormalization())

    # Block 2
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=L2(l2=0.01)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=L2(l2=0.01)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=L2(l2=0.01)))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    # Block 3
    model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=L2(l2=0.01)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=L2(l2=0.01)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=L2(l2=0.01)))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(BatchNormalization())

    # Block 4
    model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=L2(l2=0.01)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=L2(l2=0.01)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=L2(l2=0.01)))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(BatchNormalization())

    # Dense
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    # Softmax
    model.add(Dense(NC, activation='softmax'))

    return model



class MeterDetection:

    def __init__(self, meter_type):

        isCuda = torch.cuda.is_available()
        print(f'cuda is_available:{torch.cuda.is_available()}')
        torch.cuda.device(0)

        if isCuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.input_size = 320

        # define model

        self.model_yolo = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=f"./weights/{meter_type}_box.pt",
            force_reload = True,
        )
        if isCuda:
            self.model_yolo = self.model_yolo.cuda()

        self.model_yolo = self.model_yolo.autoshape()



    def detect(self, x, threshold=0.7):

        self.model_yolo.conf = threshold
        x = cv2.cvtColor(np.float32(x), cv2.COLOR_BGR2RGB)
        results = self.model_yolo([x], size=self.input_size)
        results = results.xyxy[0].cpu().numpy()
        boxes = results[:, :4]
        scores = results[:, 4]
        classes = results[:, 5]

        return boxes, scores, classes

class NumberDetection:

    def __init__(self, meter_type):
        isCuda = torch.cuda.is_available()
        print(f'cuda is_available:{torch.cuda.is_available()}')
        torch.cuda.device(0)
        if isCuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.input_size = 320

        # define model
        self.model_yolo = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=f"./weights/{meter_type}_num.pt",
            force_reload = True,
        )
        if isCuda:
            self.model_yolo = self.model_yolo.cuda()
        self.model_yolo = self.model_yolo.autoshape()

    def detect(self, x, threshold=0.7):
        self.model_yolo.conf = threshold
        x = cv2.cvtColor(np.float32(x), cv2.COLOR_BGR2RGB)
        results = self.model_yolo([x], size=self.input_size)
        results = results.xyxy[0].cpu().numpy()

        boxes = results[:, :4]
        scores = results[:, 4]
        classes = results[:, 5]

        return boxes, scores, classes



def load_all_model():

    current_time = time.time()
    global ND, MD, NumberC
    ND = {}
    MD = {}
    NumberC = {}

    type_list = [1, 2, 4, 5, 6, 9]
    # type_list = [9]

    IM_SHAPE = (44, 27, 3)
    NC = 10


    for type_ in type_list:
        print(f'start load model of type: {type_}')
        ND[type_] = NumberDetection(type_)
        MD[type_] = MeterDetection(type_)

        model = make_classification_model(IM_SHAPE, NC)
        model.load_weights(f'./weights/{type_}_classification.h5')
        NumberC[type_] = model

    print(f'Models loading time = {time.time() - current_time}')


def detect_number2(img,  pure_img_name, meter_type=0, box=None, rot_angle=None):
    ''' image should be numpy array, box list of x1,y1,x2,y2, rot_angle[degree]'''

    if MD.get(meter_type) is None:
        meter_type = 0

    frmt_date = dt.datetime.utcfromtimestamp(
        time.time()).strftime("%Y%m%d %H:%M")

    res_dict = {}
    res_dict["name"] = pure_img_name
    res_dict["DateTime"] = frmt_date[:8]
    res_dict["Image_height"] = img.shape[0]
    res_dict["Image_width"] = img.shape[1]

    result = []

    if rot_angle is None:
        img, ang = horizontal_balance(img)
        if ang is None:
            res_dict["Rotation_angle"] = 0
        else:
            res_dict["Rotation_angle"] = ang
        print('rotation compute complete')
    else:
        img = ndimage.rotate(img, rot_angle)
        res_dict["Rotation_angle"] = rot_angle

    draw = img
    h, w, _ = img.shape

    if box is None:
        boxes, scores, labels = MD[meter_type].detect(img, 0.5)

        if len(scores) != 0:
            idx = np.argmax(scores)
            box = boxes[idx]

            num_patches = []

            b = box.astype(int)
            res_dict["box"] = b

            cropped = draw[b[1]:b[3], b[0]:b[2]]
            current_time = time.time()
            nboxes,_,_ = ND[meter_type].detect(cropped, 0.6)

            if len(nboxes) <= 2:
                print(f'no number is detected for {pure_img_name}')
            else:
                nboxes = nboxes.astype(int)
                nboxes = sort_np_array(nboxes, column=0, flip=False)

                for box in nboxes:
                    cropped_number = cropped[ box[1]:box[3], box[0]:box[2] ]
                    num_patches.append(cropped_number)

                current_time = time.time()
                #num_detected는 check_double후에 처리.
                #res_dict["num_detected"] = len(num_patches)
                res_dict["objects_info"] = []

                class_res = []

                score_res = []



                for patch, box in zip(num_patches, nboxes):

                    patch_resized = cv2.resize(patch, (27, 44))
                    patch_resized = patch_resized[np.newaxis,...]
                    predictions = NumberC[meter_type].predict(patch_resized)
                    result = np.argmax(predictions)
                    score = np.max(predictions)
                    class_res.append(result)
                    score_res.append(score)

                nboxes, score_res, class_res = check_double(nboxes, score_res, class_res)

                if len(nboxes) <= 2:
                    print(f'filtered len nboxes : {len(nboxes)} {pure_img_name}')
                    return 0

                res_dict["num_detected"] = len(nboxes)
                class_res = decimal_process(box, nboxes, class_res)

                for box, result, score in zip(nboxes, class_res, score_res):
                    temp_dict = {}
                    # cv2.rectangle(draw, (box[0]+b[0], box[1]+b[1]), (box[2]+b[0], box[3]+b[1]), (0,0,255), 2)
                    temp_dict["class"] = result
                    temp_dict["coordinate"] = list(box)
                    temp_dict["probability"] = np.round(score,5)
                    res_dict["objects_info"].append(temp_dict)
    else:
        print(f'no box is detected for {pure_img_name}')

    return json.dumps(res_dict, indent=4, cls=NpEncoder)


def detect_number(img_path, meter_type, box=None, rot_angle=None):
    ''' image should be numpy array, box list of x1,y1,x2,y2, rot_angle[degree]'''

    print(f"loading image from : {img_path}")

    img = cv2.imread(img_path)
    pure_img_name = os.path.split(img_path)[-1]

    frmt_date = dt.datetime.utcfromtimestamp(
        time.time()).strftime("%Y%m%d %H:%M")

    jsonDump = detect_number2(img, pure_img_name, meter_type, box, rot_angle)

    res_json_path = os.path.join(os.path.dirname(
        img_path), f'{frmt_date[:8]}_{pure_img_name.split(".")[0]}.json')

    print(f'json file saved to {res_json_path}')

    with open(res_json_path, "w") as json_file:
        json_file.write(jsonDump)

    resultjson = json.loads(jsonDump)

    csvrow = []
    csvrow.append(resultjson["name"])
    strval = ''

    if resultjson.get("objects_info") is not None:
        for info in resultjson["objects_info"]:
            strval += str(info["class"])

        csvrow.append(strval)

        for info in resultjson["objects_info"]:
            csvrow.append(str(info["probability"]))

    with open('result.csv', 'a', newline='') as resultfile:
        wr = csv.writer(resultfile)
        wr.writerow(csvrow)

    return jsonDump


if __name__ == '__main__':
    # img = cv2.imread('./54.jpg')
    # cv2.imshow('',img)
    # cv2.waitKey(0)

    #########################################

    load_all_model()
    img_path = './dataset/test_data/8.jpg'
    detect_number(img_path)

    #########################################

    load_all_model()
    IN_IMAGES_PATH = 'dataset/type-9'
    # IN_IMAGES_PATH = 'E:/DM/PROJECTS/HY.CHECK/imageprocessing/20210608-test/'

    print(f'Reading images from {IN_IMAGES_PATH}')
    for root, dirs, files in os.walk(IN_IMAGES_PATH):
        for filename in files:
            if os.path.splitext(filename)[-1].lower() not in ['.jpg']:
                continue
            detect_number(os.path.join(root, filename), 9)
