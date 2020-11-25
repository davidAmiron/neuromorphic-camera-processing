import sys
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
from utils import *

assert(len(sys.argv) >= 2)
filename = sys.argv[1]

width = 240
height = 180
dt = 5 / 1000 # length of displaying interval in seconds
t_start_optical_flow = 50e-3 # Wait this many second before starting to compute optical flow

x_derivs = SpatialDerivs('x', width, height)
y_derivs = SpatialDerivs('y', width, height)
optical_flow = OpticalFlow(width, height)

decay_method = 'exponential'
postprocess_method = None
median_space_size = 10
mean_filter_size = 5
min_filter_size = 3
frame_memory = 15
frames_x = []
frames_y = []

curr_frame = 0
with open(filename, 'r') as fh:
    for line in fh:
        data = line.split(' ')
        timestamp = float(data[0])
        x = int(data[1])
        y = int(data[2])
        polarity = int(data[3])
        x_derivs.update(timestamp, x, y, polarity, decay=decay_method)
        y_derivs.update(timestamp, x, y, polarity, decay=decay_method)
        #if timestamp > t_start_optical_flow:
        #    optical_flow.update(timestamp, x, y, polarity)

        if timestamp > curr_frame * dt:
            #a = np.zeros((height, width))
            #a.fill(0.5)
            xd = x_derivs.derivs
            yd = y_derivs.derivs
            frames_x.append(xd)
            frames_y.append(yd)
            if len(frames_x) > frame_memory:
                frames_x.pop(0)
            if len(frames_y) > frame_memory:
                frames_y.pop(0)
            grads = np.ones((height, width)) * 0.5
            grads = draw_gradients(grads, xd, yd)
            #flows = np.ones((height, width)) * 0.5
            #flows = optical_flow.draw_flows(flows)

            # Postprocess frame before visualizing
            if postprocess_method == 'median_space':
                xd = ndimage.median_filter(xd, size=median_space_size)
                yd = ndimage.median_filter(yd, size=median_space_size)
            elif postprocess_method == 'median_time':
                xd = np.median(frames_x, 0)
                yd = np.median(frames_y, 0)
            elif postprocess_method == 'mean_space':
                xd = ndimage.filters.convolve(xd, np.full((mean_filter_size, mean_filter_size),
                                                          1/(mean_filter_size**2)))
                yd = ndimage.filters.convolve(yd, np.full((mean_filter_size, mean_filter_size),
                                                          1/(mean_filter_size**2)))
            elif postprocess_method == 'mean_time':
                xd = np.mean(frames_x, 0)
                yd = np.mean(frames_y, 0)
            elif postprocess_method == 'min_space':
                x_sign = np.sign(xd)
                y_sign = np.sign(yd)
                xd = ndimage.filters.minimum_filter(np.abs(xd), size=(min_filter_size, min_filter_size))
                yd = ndimage.filters.minimum_filter(np.abs(yd), size=(min_filter_size, min_filter_size))
                xd = x_sign * xd
                yd = y_sign * yd
            elif postprocess_method == 'min_time':
                xd = np.min(frames_x, 0)
                yd = np.min(frames_y, 0)


            cv2.imshow('x derivatives', xd + 0.5)
            cv2.imshow('y derivatives', yd + 0.5)
            """if timestamp > 3:
                cv2.imwrite(sys.argv[2], xd + 0.5)
                cv2.waitKey(0)
                sys.exit(0)"""
            #cv2.imshow('visual gradients', grads)
            #cv2.imshow('optical flow', flows)
            curr_frame += 1
            #img.fill(0)
            cv2.waitKey(1)

        """if timestamp > 2:
            plt.ioff()
            plt.plot_surface(optical_flow.sae)
            plt.show()"""
