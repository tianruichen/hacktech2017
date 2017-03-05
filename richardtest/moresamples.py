import numpy as np
from os import listdir
from os.path import isfile, join
import os
onlyfiles = [f for f in listdir('dataset')]
onlyfiles = onlyfiles[1:]
data = []
label = []
for f in onlyfiles:
    root = 'dataset/' + f
    rotate1 = []
    rotate2 = []
    rotate3 = []
    for item in listdir(root):
        path = os.path.join(root, item)
        if item.startswith('rotate') and os.path.isfile(path):
            print(path)
            os.remove(path)

