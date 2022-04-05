import math

"""
Constant for inner ratio to use for crop boxes
"""
INNER_RATIO = 1


"""
Constant for filtering out non-handheld detections from COCO output 
"""
HANDHELD_MAP = {29: 'frisbee', 32: 'sports ball', 35: 'baseball glove', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 64: 'mouse', 65: 'remote', 67: 'cell phone', 73: 'book', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}


"""
Basic function to get distance between two points
"""
def distGet(point1, point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)


"""
Gets overlap of box1 into box2 (not same as IOU) 
"""
def bbox_overlap(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    
    # intersection area
    interArea = abs(xB - xA + 1) * abs(yB - yA + 1)
     
    # box1 area
    box1Area = abs((box1[2] - box1[0]) * (box1[3] - box1[1]))
    
    # returning percentage of overlap if intersection is not 0
    if box1Area != 0:
        overlap = interArea / box1Area
        return overlap
    else:
        return 0