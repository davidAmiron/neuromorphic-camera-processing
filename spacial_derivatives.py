import sys
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
from utils import *

assert(len(sys.argv) == 2)
filename = sys.argv[1]

width = 240
height = 180
dt = 5 / 1000 # length of displaying interval in seconds

x_derivs = SpatialDerivs('x', width, height)
y_derivs = SpatialDerivs('y', width, height)

curr_frame = 0
with open(filename, 'r') as fh:
    for line in fh:
        data = line.split(' ')
        timestamp = float(data[0])
        x = int(data[1])
        y = int(data[2])
        polarity = int(data[3])

        x_derivs.update(timestamp, x, y, polarity, decay='exponential')
        y_derivs.update(timestamp, x, y, polarity, decay='exponential')

        if timestamp > curr_frame * dt:
            #a = np.zeros((height, width))
            #a.fill(0.5)
            cv2.imshow('x derivatives', x_derivs.derivs)
            cv2.imshow('y derivatives', y_derivs.derivs)
            curr_frame += 1
            #img.fill(0)
            cv2.waitKey(1)
