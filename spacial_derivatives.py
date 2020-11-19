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
t_start_optical_flow = 50e-3 # Wait this many second before starting to compute optical flow

x_derivs = SpatialDerivs('x', width, height)
y_derivs = SpatialDerivs('y', width, height)
optical_flow = OpticalFlow(width, height)

curr_frame = 0
with open(filename, 'r') as fh:
    for line in fh:
        data = line.split(' ')
        timestamp = float(data[0])
        x = int(data[1])
        y = int(data[2])
        polarity = int(data[3])
        # MAKE GRADIENTS GREY AND THEN POSITIVE AND NEGATIVE? HOW EVEN IS NEGATIVE GRADIENT DISPLAYED
        #x_derivs.update(timestamp, x, y, polarity, decay='exponential')
        #y_derivs.update(timestamp, x, y, polarity, decay='exponential')
        #if timestamp > t_start_optical_flow:
        #    optical_flow.update(timestamp, x, y, polarity)
        x_derivs.update(timestamp, x, y, polarity, decay='constant')
        y_derivs.update(timestamp, x, y, polarity, decay='constant')

        if timestamp > curr_frame * dt:
            #a = np.zeros((height, width))
            #a.fill(0.5)
            xd = x_derivs.derivs
            yd = y_derivs.derivs
            grads = np.ones((height, width)) * 0.5
            grads = draw_gradients(grads, xd, yd)
            flows = np.ones((height, width)) * 0.5
            flows = optical_flow.draw_flows(flows)
            cv2.imshow('x derivatives', xd + 0.5)
            cv2.imshow('y derivatives', yd + 0.5)
            #cv2.imshow('visual gradients', grads)
            #cv2.imshow('optical flow', flows)
            curr_frame += 1
            #img.fill(0)
            cv2.waitKey(1)

        """if timestamp > 2:
            plt.ioff()
            plt.plot_surface(optical_flow.sae)
            plt.show()"""
