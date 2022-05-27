import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import json
import time
import torch

name = 'det'
labelDir = "labels"
coco_label_dir = './' + labelDir + '/ground_truth/'
hh_label_dir = './' + labelDir + '/handheld/'
images_dir = './images/'

hh_fnames = os.listdir(hh_label_dir)
handheld_map = {29: 'frisbee', 32: 'sports ball', 35: 'baseball glove', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 64: 'mouse', 65: 'remote', 67: 'cell phone', 73: 'book', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
IOU_THRESH = 0.5

# creating json for new stats if needed then retrieving
if not os.path.isfile(labelDir + "results.json"):
    with open(labelDir + "results.json", 'w') as file:
        file.write(json.dumps({}))

# creating folder for storing test results if needed
if not os.path.isdir(labelDir + "TestRes/"):
    os.mkdir(labelDir + "TestRes/")

def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y
    
def bbox_iou(boxA, boxB):    
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def read_answer_label_file(filename: str, w, h, thres, thresCheck, answer):
    label = None
    name = filename.split("/")[-1]
    label = allLabels[answer][name]
    
    if thresCheck:
        oldLabel = [row for row in label if row[5] >= thres]
        label = [row[:5] for row in label if row[5] >= thres]
        if len(label) == 0: 
            return False, False, False, False
    
    if len(label) == 0:
        return np.array(label), np.array(label)
    label = np.asarray(label)
    label_cats = np.array(label[:, 0], dtype=int)
    
    if thresCheck:
        return label_cats, xywhn2xyxy(label[:, 1:], w, h), [row[5] for row in oldLabel], [row[0] for row in oldLabel]
    else:
        return label_cats, xywhn2xyxy(label[:, 1:], w, h)


def read_label_file(filename: str, w, h, thres, thresCheck):
    f = open(filename, mode='r')
    line = f.read()
    f.close()
    # print(line)
    # print(line.split(' '))
    label = line.split('\n')
    label = [s.split(' ') for s in label if s != '']
    label = [list(map(float, l)) for l in label]
    
    if thresCheck:
        oldLabel = [row for row in label if row[5] >= thres]
        label = [row[:5] for row in label if row[5] >= thres]
        if len(label) == 0: 
            return False, False, False, False
    
    if len(label) == 0:
        return np.array(label), np.array(label)
    label = np.asarray(label)
    label_cats = np.array(label[:, 0], dtype=int)
    
    if thresCheck:
        return label_cats, xywhn2xyxy(label[:, 1:], w, h), [row[5] for row in oldLabel], [row[0] for row in oldLabel]
    else:
        return label_cats, xywhn2xyxy(label[:, 1:], w, h)

def filter_labels(input_cats: np.ndarray, input_bboxes: np.ndarray, filter_cats: np.ndarray, filter_bboxes: np.ndarray, threshold=0.95):
    input_matched = np.zeros(input_cats.shape, dtype=bool)
    filter_matched = np.zeros(filter_bboxes.shape, dtype=bool)
    
    for i in range(input_bboxes.shape[0]):
        matched = False
        remaining_f = (filter_matched == 0).nonzero()[0]  # indicies of 0 (falses in array)
        for f in remaining_f:
            if input_cats[i] == filter_cats[f]:
                if (bbox_iou(input_bboxes[i], filter_bboxes[f]) >= threshold):
                    filter_matched[f] = True
                    input_matched[i] = True
    
    return input_cats[input_matched == False], input_bboxes[input_matched == False]


def main(CONF_THRESH, det_label_dir, iou_thresh):
    # getting det name of json
    det_name = det_label_dir.split("/")[-2]

    # on with normal
    det_fnames = os.listdir(det_label_dir)
    hh_not_in_det_fnames = set(hh_fnames).difference(set(det_fnames))
    
    tp = np.zeros((80,), dtype=float)
    fp = np.zeros((80,), dtype=float)
    fn = np.zeros((80,), dtype=float)
    
    for det_fname in det_fnames:
        img = cv2.imread(images_dir + det_fname[:-4] + '.jpg')
        w = img.shape[1]
        h = img.shape[0]

        det_cat, det_bboxes, det_conf, det_class = read_answer_label_file(det_label_dir + det_fname, w, h, CONF_THRESH, True, det_name)
        hh_cat, hh_bboxes = read_answer_label_file(hh_label_dir + det_fname, w, h, 0, False, "handheld")
        hh_matched = np.zeros(len(hh_cat), dtype=bool)
        # Label file is blank
        if type(det_cat) == bool:
            hh_not_in_det_fnames.add(det_fname)
            # fn[hh_cat[np.flatnonzero(hh_matched == False)]] += 1    # all false negatives
            continue
        coco_cat, coco_bboxes = read_answer_label_file(coco_label_dir + det_fname, w, h, 0, False, "ground_truth")
        coco_cat, coco_bboxes = filter_labels(coco_cat, coco_bboxes, hh_cat, hh_bboxes)
        coco_matched = np.zeros(len(coco_cat), dtype=bool)
        for i in range(len(det_cat)):
            matched = False
            for h in range(len(hh_cat)):
                if (det_cat[i] == hh_cat[h]) and not hh_matched[h]:
                    if (bbox_iou(det_bboxes[i], hh_bboxes[h]) >= iou_thresh):                        
                        hh_matched[h] = True
                        tp[det_cat[i]] += 1
                        matched = True
                        break
            if not matched:
                for c in range(len(coco_cat)):
                    if (det_cat[i] == coco_cat[c]) and not coco_matched[c]:
                        if (bbox_iou(det_bboxes[i], coco_bboxes[c]) >= iou_thresh):
                            coco_matched[c] = True
                            matched = True
                            break
            if not matched:
                fp[det_cat[i]] += 1
        
        for i in range(len(hh_cat)):
            if hh_matched[i] == False:
                fn[hh_cat[i]] += 1
        #fn[hh_cat[np.flatnonzero(hh_matched == False)]] += 1

    for hh_not_in_det_fname in hh_not_in_det_fnames:
        hh_cat, hh_bboxes = read_answer_label_file(hh_label_dir + hh_not_in_det_fname, 100, 100, 0, False, "handheld")
        for i in range(len(hh_cat)):
            fn[hh_cat[i]] += 1


    p = tp / (tp + fp + 0.00001)
    AP = np.sum(tp) / (np.sum(tp) + np.sum(fp) + 0.00001)
    fN = np.sum(fn) 
    r = tp / (tp + fn + 0.00001)
    AR = np.sum(tp) / (np.sum(tp) + fN + 0.00001)
    #xAR = np.sum(tp) / (np.sum(tp) + np.sum(fn) + 0.00001)
    
    # uncomment if you would like to see specific stats at any given confidence threshold
    #"""
    if CONF_THRESH in [0.25]:
        print("confidence", CONF_THRESH)
        #print('P per class', p)
        #print('R per class', r)
        print("true positives", np.sum(tp))
        print("false positives", np.sum(fp))
        print("false negatives", fN)
        #print("false negatives", np.sum(fn))
        print('AP', AP)
        print('AR', AR)
        print("\n")
        #"""
    
    return [AP, AR, [np.sum(tp), np.sum(fp), fN]]


if __name__ == '__main__':
    fig, ax = plt.subplots()
    ax.set_title("PR  Curve")
    ax.set_ylabel("precision")
    ax.set_xlabel("recall")
    ax.set_ylim([0, 1])
    splits = 20
    confThres = [i/splits for i in list(range(splits))]
    
    ################## Getting handheld and G.T. jsons #########
    allLabels = {}
    with open("./labels/allLabels.json", 'r') as file:
        allLabels = json.loads(file.read())
    ############################################################    
    
    # can only have specific size results be displayed
    for label in os.listdir(labelDir):
        # early exiting conditions
        if label in ["handheld", "ground_truth", "allLabels.json", "ignore"]:
            continue        
        
        # looking for existing stats else making new
        if not os.path.isdir("./" + labelDir + "/" + label + "/"):
            continue
        if os.path.isfile(labelDir + "TestRes/" + label + ".json"):
            with open(labelDir + "TestRes/" + label + ".json", 'r') as file:
                scatter = json.loads(file.read())
            scatter1 = scatter[0]
            scatter2 = scatter[1]
            #scatter2.pop(0)
            newScatter = list(zip(scatter[0], scatter[1]))
            newScatter.append(scatter[2])
            scatter = newScatter
        else:
            print(label)
            print(confThres)
            t0 = time.time()
            scatter = [main(i, "./" + labelDir + "/" + label + "/", IOU_THRESH) for i in confThres]
            print("Time Needed", time.time() - t0)
            scatter1 = [i[0] for i in scatter]
            scatter2 = [i[1] for i in scatter]
            tpTracks = [i[2] for i in scatter]
            
            scatter = [scatter1, scatter2, tpTracks]
            with open(labelDir + "TestRes/" + label + ".json", 'w') as file:
                file.write(json.dumps(scatter))
                
        # scattering annotation plots and mAP
        ax.scatter(scatter2, scatter1, label=label)
        scatter2.insert(0, 0)
        mAP = []
        for i in range(1, splits+1):
            mAP.append(scatter1[i-1] * (scatter2[i-1]-scatter2[i]))
            
        # uncomment if you want the confidences to be labelled
        #for i in range(len(scatter1)):
         #   ax.annotate(str(confThres[i]), xy = [scatter2[i], scatter1[i]])
        ax.grid()
        
        # Adding a legend
        ax.legend(bbox_to_anchor=(1.0, 1.0))
        print("mAP:", label, sum(mAP))
        print()
    
    # saves a graph of the comparison to your folder
    plt.savefig("comparison_graph.jpg", dpi=300)
    
#print(existLabel)
# saving new labels
#with open(labelDir + "results.json", 'w') as file:
 #   file.write(json.dumps(existLabel))