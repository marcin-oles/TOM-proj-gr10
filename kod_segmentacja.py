#szereg działań, który pozwala na pracę w github
! curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
! sudo apt-get install git-lfs
! git lfs install
! git clone https://github.com/neheller/kits19.git

#instalacja danych z kits19
%cd kits19
! python -m starter_code.get_imaging

#załączenie bibliotek potrzebnych w algorytmie
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
import matplotlib.pyplot as plt

#instalacja biblioteki do analizy danych z rozszerzeniem .nii
pip install nibabel
!pip install python-utils
from starter_code.utils import load_case

#załadowanie danych i pobranie wycinków
volume0, segmentation0 =load_case("case_00000")
volume0 = volume0.get_fdata() 
segmentation0 = segmentation0.get_fdata()
volume_shape=np.shape(volume0)
segmentation_shape=np.shape(segmentation0)
print(f"shape of volume {volume_shape}")
print(f"shape of segmentation {segmentation_shape}")

#wyswietlenie przekrojów 
#for i in range (0, volume_shape[0],1):
 # plt.figure()
 # plt.imshow(volume0[i],cmap='gray')

#funkcja która umożliwi pracę na wybranych przekrojach z widocznymi nerkami
def show_slices(slices):
   """ Function to display row of image slices """
   fig, axes = plt.subplots(1, len(slices))
   for i, slice in enumerate(slices):
       axes[i].imshow(slice.T, cmap="gray", origin="lower")
slice_0 = volume0[260, :, :]
slice_1 = volume0[:, 30, :]
slice_2 = volume0[:, :, 16]
show_slices([slice_0, slice_1, slice_2])

#wyświetlenie przekroju case_00000
image=color.rgb2gray(slice_0)
plt.imshow(image, cmap='gray')

#wybór przekrojów dla innych przypadków

#case_00001
volume1, segmentation1 =load_case("case_00001")
volume1 = volume1.get_fdata() 
segmentation1 = segmentation1.get_fdata()
volume_shape=np.shape(volume1)
segmentation_shape=np.shape(segmentation1)
print(f"shape of volume {volume_shape}")
print(f"shape of segmentation {segmentation_shape}")

#ywyodrębnienie przekroju zawierającego nerki
slice_0 = volume1[270, :, :]
image1=color.rgb2gray(slice_0)
plt.imshow(image1, cmap='gray')


#case_00025
volume2, segmentation2 =load_case("case_00025")
volume2 = volume2.get_fdata() 
segmentation2 = segmentation2.get_fdata()
volume_shape=np.shape(volume2)
segmentation_shape=np.shape(segmentation2)
print(f"shape of volume {volume_shape}")
print(f"shape of segmentation {segmentation_shape}")


#ywyodrębnienie przekroju zawierającego nerki
slice_0 = volume2[52, :, :]
image2=color.rgb2gray(slice_0)
plt.imshow(image2, cmap='gray')

#case_00050
volume3, segmentation3 =load_case("case_00050")
volume3 = volume3.get_fdata() 
segmentation3 = segmentation3.get_fdata()
volume_shape=np.shape(volume3)
segmentation_shape=np.shape(segmentation3)
print(f"shape of volume {volume_shape}")
print(f"shape of segmentation {segmentation_shape}")


#ywyodrębnienie przekroju zawierającego nerki
slice_0 = volume3[40, :, :]
image3=color.rgb2gray(slice_0)
plt.imshow(image3, cmap='gray')

#case_00100
volume4, segmentation4 =load_case("case_00100")
volume4 = volume4.get_fdata() 
segmentation4 = segmentation4.get_fdata()
volume_shape=np.shape(volume4)
segmentation_shape=np.shape(segmentation4)
print(f"shape of volume {volume_shape}")
print(f"shape of segmentation {segmentation_shape}")


#ywyodrębnienie przekroju zawierającego nerki
slice_0 = volume4[190, :, :]
image4=color.rgb2gray(slice_0)
plt.imshow(image4, cmap='gray')

#case_00200
volume5, segmentation5 =load_case("case_00200")
volume5 = volume5.get_fdata() 
segmentation5 = segmentation5.get_fdata()
volume_shape=np.shape(volume5)
segmentation_shape=np.shape(segmentation5)
print(f"shape of volume {volume_shape}")
print(f"shape of segmentation {segmentation_shape}")


#ywyodrębnienie przekroju zawierającego nerki
slice_0 = volume5[33, :, :]
image5=color.rgb2gray(slice_0)
plt.imshow(image5, cmap='gray')

#przypisane obrazów aby umżliwic dalsze operacje na wybranych przekrojach
case_00 = image
case_11 = image1
case_25 = image2
case_50 = image3
case_100 =image4
case_200 =image5

#Wycinanie niepotrzebnego pola obrazu #[y, x]
case_00 = case_00[250:370, 130:360] 
case_11 = case_11[200:350, 100:410]
case_25 = case_25[230:390, 130:440]
case_50 = case_50[250:380, 115:415]
case_100 = case_100[230:340, 140:380]
case_200 = case_200[260:370, 120:410]

#Filtracja (Skimage)
case_00_denoise = restoration.denoise_nl_means(case_00, multichannel=False)
case_11_denoise = restoration.denoise_nl_means(case_11, multichannel=False)
case_25_denoise = restoration.denoise_nl_means(case_25, multichannel=False)
case_50_denoise = restoration.denoise_nl_means(case_50, multichannel=False)
case_100_denoise = restoration.denoise_nl_means(case_100, multichannel=False)
case_200_denoise = restoration.denoise_nl_means(case_200, multichannel=False)

#Canny edge detector #dla każdego przypadku progi ustawione osobno recznie #metoda prob i bledow
case_00_canny = feature.canny(case_00_denoise, sigma = 9, low_threshold = 0.10, high_threshold = 0.15)
case_11_canny = feature.canny(case_11_denoise, sigma = 13, low_threshold = 0.01, high_threshold = 0.05) #13, 0.01, 0.05
case_25_canny = feature.canny(case_25_denoise, sigma = 12, low_threshold = 0.05, high_threshold = 0.10)
case_50_canny = feature.canny(case_50_denoise, sigma = 11, low_threshold = 0.01, high_threshold = 0.04)
case_100_canny = feature.canny(case_100_denoise, sigma = 11, low_threshold = 0.03, high_threshold = 0.07)
case_200_canny = feature.canny(case_200_denoise, sigma = 13, low_threshold = 0.03, high_threshold = 0.05)

#Closing dla case_25 polepsza wyniki snake'a
case_25_canny = binary_closing(case_25_canny)

#Active contour model (Skimage) dla case_00
s = np.linspace(0, 2*np.pi, 400)
x = 45 + 30*np.cos(s) #wspolrzedna x kola wewnatrz ktorego szukany jest obrys + promien
y = 65 + 50*np.sin(s) #wspolrzednia y kola wewnatrz ktorego szukany jest obrys + promien
init = np.array([x, y]).T
snake = active_contour(gaussian(case_00_canny, 3), init, alpha=0.0001, beta=10, gamma=0.001, w_edge=0.5)

#Wizualizacja snake'a na obrazie binarnym dla case_00
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(case_00_canny, cmap= 'gray')
ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3) #obrys snake
plt.show()

#Naniesiony snake na obraz oryginalny bez obrysu dla case_00
fig, ax = plt.subplots(figsize = (7, 7))
ax.imshow(case_00, cmap = 'gray')
ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
plt.show()

#Active contour model dla case_11 
s = np.linspace(0, 2*np.pi, 400)
x = 50 + 40*np.cos(s) 
y = 80 + 40*np.sin(s) 
init = np.array([x, y]).T
snake = active_contour(gaussian(case_11_canny, 3), init, alpha=0.0001, beta=10, gamma=0.001, w_edge=0.5)

#Naniesiony snake na obraz oryginalny dla case_11 
fig, ax = plt.subplots(figsize = (7, 7))
ax.imshow(case_11, cmap = 'gray')
ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
plt.show()

#Active contour model Skimage dla case_25 
s = np.linspace(0, 2*np.pi, 400)
x = 45 + 30*np.cos(s) #wspolrzedna x kola wewnatrz ktorego szukany jest obrys + promien
y = 103 + 40*np.sin(s) #wspolrzednia y kola wewnatrz ktorego szukany jest obrys + promien
init = np.array([x, y]).T
snake = active_contour(gaussian(case_25_canny, 1), init, alpha=0.0001, beta=10, gamma=0.001, w_edge=0.5)

#Naniesiony snake na obraz oryginalny dla case_25 
fig, ax = plt.subplots(figsize = (7, 7))
ax.imshow(case_25, cmap = 'gray')
ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
plt.show()

#Active contour model Skimage dla case_50 
s = np.linspace(0, 2*np.pi, 400)
x = 55 + 45*np.cos(s) #wspolrzedna x kola wewnatrz ktorego szukany jest obrys + promien
y = 65 + 42*np.sin(s) #wspolrzednia y kola wewnatrz ktorego szukany jest obrys + promien
init = np.array([x, y]).T
snake = active_contour(gaussian(case_50_canny, 1), init, alpha=0.0001, beta=10, gamma=0.001, w_edge=0.5)

#Naniesiony snake na obraz oryginalny dla case_50
fig, ax = plt.subplots(figsize = (7, 7))
ax.imshow(case_50, cmap = 'gray')
ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
plt.show()

#Active contour model Skimage dla case_100 
s = np.linspace(0, 2*np.pi, 400)
x = 45 + 30*np.cos(s) #wspolrzedna x kola wewnatrz ktorego szukany jest obrys + promien
y = 45 + 40*np.sin(s) #wspolrzednia y kola wewnatrz ktorego szukany jest obrys + promien
init = np.array([x, y]).T
snake = active_contour(gaussian(case_100_canny, 1), init, alpha=0.0001, beta=10, gamma=0.001, w_edge=0.5)

#Naniesiony snake na obraz oryginalny dla case_100 
fig, ax = plt.subplots(figsize = (7, 7))
ax.imshow(case_100, cmap = 'gray')
ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
plt.show()

#Active contour model Skimage dla case_200 
s = np.linspace(0, 2*np.pi, 400)
x = 50 + 45*np.cos(s) #wspolrzedna x kola wewnatrz ktorego szukany jest obrys + promien
y = 60 + 40*np.sin(s) #wspolrzednia y kola wewnatrz ktorego szukany jest obrys + promien
init = np.array([x, y]).T
snake = active_contour(gaussian(case_200_canny, 1), init, alpha=0.0001, beta=10, gamma=0.001, w_edge=0.5)

#Naniesiony snake na obraz oryginalny dla case_200
fig, ax = plt.subplots(figsize = (7, 7))
ax.imshow(case_200, cmap = 'gray')
ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
plt.show()

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

