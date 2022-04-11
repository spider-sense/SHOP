# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 18:05:45 2021

@author: derph
"""

# augments the images
import cv2
import numpy as np
import os
import random

INPUT_DIR = "images"
OUTPUT_DIR = "linBlur"

BLUR_SIZE = 10
ANGLE_RANGE = (0, 179)

if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
images = os.listdir(INPUT_DIR)

#size - in pixels, size of motion blur
#angle - in degrees, direction of motion blur
print("Applying Linear Blur!")
def apply_motion_blur(img, size, angle):
    k = np.zeros((size, size), dtype=np.float32)
    k[ (size-1)// 2 , :] = np.ones(size, dtype=np.float32)
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D( (size / 2 -0.5 , size / 2 -0.5 ) , angle, 1.0), (size, size) )
    k = k * ( 1.0 / np.sum(k) )
    return cv2.filter2D(img, -1, k)

for image in images:
    print(image)
    src_path = f"{INPUT_DIR}/{image}"
    dst_path = f"{OUTPUT_DIR}/{image}"
    cv2.imwrite(dst_path, apply_motion_blur(cv2.imread(src_path), BLUR_SIZE, random.uniform(*ANGLE_RANGE)))

# rotation + all blur    
import os
import random

from wand.image import Image

INPUT_DIR = "linBlur"
OUTPUT_DIR = "allBlur"
# Rotational blur
BLUR_ROT_RANGE = (1.0, 3.0)    

print("Applying Rotational blur!")
if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
images = os.listdir(INPUT_DIR)

i = 0
for image in images:
    print(image)
    with Image(filename=f"{INPUT_DIR}/{image}") as img:
        i += 1
        img.rotational_blur(angle=random.uniform(*BLUR_ROT_RANGE))
        img.save(filename=f"{OUTPUT_DIR}/{image}")
print(f"Processed {i} of {len(images)}")








