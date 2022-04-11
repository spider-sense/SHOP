# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 01:06:40 2022

@author: derph
"""

import os
import json

labels = os.listdir("labels")
allLabels = {}
for name in labels:
    print(name)
    handJson = {}
    handDir = os.listdir("./labels/" + name)
    for hand in handDir:
        with open("./labels/" + name + "/" + hand, 'r') as file:
            data = [[float(i) for i in row.split(" ")] for row in file.read().split("\n") if row != ""]
            handJson[hand] = data
    allLabels[name] = handJson
print("labels for", allLabels.keys())
with open("labels/allLabels.json", 'w') as file:
    file.write(json.dumps(allLabels))