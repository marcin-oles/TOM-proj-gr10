##segmentacja ręczna

import numpy as np
import os
from skimage.filters import rank
from skimage import io
from skimage import color
from skimage import feature
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from PIL import Image as im
from skimage import transform
from skimage import segmentation
from skimage.morphology import disk
from skimage.morphology import binary_closing
from skimage.morphology import binary_dilation
from skimage import restoration
from skimage.segmentation import active_contour
from skimage.filters import gaussian

#Wyznaczanie linni obrysu nowotworu sposobem wyklikania - snake
class LineBuilder:
    def __init__(self, line):
        self.line = line
        self.xs = []
        self.ys = []
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
        self.cid = line.figure.canvas.mpl_connect('key_press_event', self)

    def __call__(self, event):
        if (event.name == 'key_press_event'):
            self.line.figure.canvas.mpl_disconnect(self.cid)
            plt.close()
        if event.inaxes != self.line.axes: return
        if (event.name == 'button_press_event'):
            self.xs.append(int(event.xdata))
            self.ys.append(int(event.ydata))
            self.line.set_data(self.xs, self.ys)
            self.line.figure.canvas.draw()


#Wyklikanie punktów do obrysu 
def click_points(image):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='gray')
    ax.set_title('click to build line segments')
    line, = ax.plot([0], [0])
    linebuilder = LineBuilder(line)
    plt.show()
    xs = np.asarray(linebuilder.xs)
    ys = np.asarray(linebuilder.ys)
    return xs, ys

#Reinterpolacja konturu aby odleglosc pomiedzy punktami miescila sie pomiedzy dmin, a dmax 
#odleglosc Euklidesowa
def reinterpolate_contours(xs, ys, dmin=2, dmax=3):
    n_xs = []
    n_ys = []
    for i in range(0, len(xs)):
        in_x = []
        in_y = []
        new_len = int(np.sqrt(np.square(xs[i] - xs[i - 1]) + np.square(ys[i] - ys[i - 1])) / dmax)
        for j in range(0, new_len):
            in_x.append(xs[i - 1] + ((xs[i] - xs[i - 1]) / new_len) * (j + 1))
            in_y.append(ys[i - 1] + ((ys[i] - ys[i - 1]) / new_len) * (j + 1))
        n_xs.append(xs[i - 1])
        n_ys.append(ys[i - 1])
        n_xs = n_xs + in_x
        n_ys = n_ys + in_y
    return np.asarray(n_xs), np.asarray(n_ys)

# Active contour model Skimage dla case_00
x, y = click_points(case_00_denoise)
x, y = reinterpolate_contours(x, y)
init = np.array([x, y]).T
snake = active_contour(case_00_canny, init, alpha=0.1, beta=50, gamma=0.001, w_edge=0.5)

#Naniesiony snake na obraz oryginalny
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(case_00_denoise, cmap='gray')
ax.plot(snake[:, 0], snake[:, 1], '*b', lw=3)
plt.show()


# Active contour model Skimage dla case_11
x, y = click_points(case_11_denoise)
x, y = reinterpolate_contours(x, y)
init = np.array([x, y]).T
snake = active_contour(case_00_canny, init, alpha=0.1, beta=50, gamma=0.001, w_edge=0.5)

#Naniesiony snake na obraz oryginalny
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(case_11_denoise, cmap='gray')
ax.plot(snake[:, 0], snake[:, 1], '*b', lw=3)
plt.show()

# Active contour model Skimage dla case_25
x, y = click_points(case_25_denoise)
x, y = reinterpolate_contours(x, y)
init = np.array([x, y]).T
snake = active_contour(case_25_canny, init, alpha=0.1, beta=50, gamma=0.001, w_edge=0.5)

#Naniesiony snake na obraz oryginalny
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(case_25_denoise, cmap='gray')
ax.plot(snake[:, 0], snake[:, 1], '*b', lw=3)
plt.show()

# Active contour model Skimage dla case_50
x, y = click_points(case_50_denoise)
x, y = reinterpolate_contours(x, y)
init = np.array([x, y]).T
snake = active_contour(case_50_canny, init, alpha=0.1, beta=50, gamma=0.001, w_edge=0.5)

#Naniesiony snake na obraz oryginalny
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(case_50_denoise, cmap='gray')
ax.plot(snake[:, 0], snake[:, 1], '*b', lw=3)
plt.show()

# Active contour model Skimage dla case_100
x, y = click_points(case_100_denoise)
x, y = reinterpolate_contours(x, y)
init = np.array([x, y]).T
snake = active_contour(case_100_canny, init, alpha=0.1, beta=50, gamma=0.001, w_edge=0.5)

#Naniesiony snake na obraz oryginalny
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(case_100_denoise, cmap='gray')
ax.plot(snake[:, 0], snake[:, 1], '*b', lw=3)
plt.show()

# Active contour model Skimage dla case_200
x, y = click_points(case_200_denoise)
x, y = reinterpolate_contours(x, y)
init = np.array([x, y]).T
snake = active_contour(case_200_canny, init, alpha=0.1, beta=50, gamma=0.001, w_edge=0.5)

#Naniesiony snake na obraz oryginalny
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(case_200_denoise, cmap='gray')
ax.plot(snake[:, 0], snake[:, 1], '*b', lw=3)
plt.show()
