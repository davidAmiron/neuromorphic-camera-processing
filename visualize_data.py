import sys
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation

assert(len(sys.argv) >= 2)
filename = sys.argv[1]
if len(sys.argv) > 2:
    method = sys.argv[2]
else:
    method = 'bluered'


width = 240
height = 180
dt = 5 / 1000 # length of accumulating interval in seconds

# Method of visualization:
# bluered: accumulate frames over dt, show positive as blue and negative as red
# grey: keep track of state of every pixel by adding or subtracting a small constant, display
#       every dt seconds

# fig = plt.figure()
# ax = fig.gca()
# plot = ax.imshow(img)

"""img[10, 10] = 1
plot = plt.imshow(img, animated=True)

def update_fig(*args):
    return img,

anim = animation.FuncAnimation(fig, update_fig, interval=10)
plt.show()"""

if method == 'bluered':
    img = np.zeros((height, width, 3), dtype=np.uint8)
elif method == 'grey':
    img = np.zeros((height, width))

blue = np.array([255, 0, 0], dtype=np.uint8)
red = np.array([0, 0, 255], dtype=np.uint8)

curr_frame = 0
with open(filename, 'r') as fh:
    for line in fh:
        data = line.split(' ')
        timestamp = float(data[0])
        x = int(data[1])
        y = int(data[2])
        polarity = int(data[3])
        
        if method == 'bluered':
            img[y, x] = blue if polarity else red
        elif method == 'grey':
            img[y, x] += (2 * polarity - 1) * 0.1


        if timestamp > curr_frame * dt:
            cv2.imshow('image', img)
            curr_frame += 1
            if method == 'bluered':
                img.fill(0)
            cv2.waitKey(1)

print('Done')
