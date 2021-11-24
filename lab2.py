import numpy as np
# from numpy.fft import fft2, ifft2, fftshift
from scipy.signal import convolve2d, correlate2d
import matplotlib.pyplot as plt

from Functions import *
from gaussfft import gaussfft


def deltax():
    # central
    dxmask = [[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]]
    return dxmask


def deltay():
    # central
    dymask = [[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]]
    return dymask


def Lv(inpic, shape='same'):
    dxmask = deltax()
    dymask = deltay()
    Lx = convolve2d(inpic, dxmask, shape)
    Ly = convolve2d(inpic, dymask, shape)
    return np.sqrt(Lx ** 2 + Ly ** 2)


# def Lvvtilde(inpic, shape='same'):
#     # ...
#     return result
#
#
# def Lvvvtilde(inpic, shape='same'):
#     # ...
#     return result
#
#
# def extractedge(inpic, scale, threshold, shape):
#     # ...
#     return contours
#
#
# def houghline(curves, magnitude, nrho, ntheta, threshold, nlines=20, verbose=False):
#     # ...
#     return linepar, acc
#
#
# def houghedgeline(pic, scale, gradmagnthreshold, nrho, ntheta, nlines=20, verbose=False):
#     # ...
#     return linepar, acc

def differenceOperators():
    tools = np.load("Images-npy/few256.npy")
    dxtools = convolve2d(tools, deltax(), 'valid')
    dytools = convolve2d(tools, deltay(), 'valid')
    f = plt.figure()
    f.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.rc('axes', titlesize=10)

    a1 = f.add_subplot(1, 3, 1)
    showgrey(tools, False)
    a1.title.set_text("original")
    a1 = f.add_subplot(1, 3, 2)
    showgrey(dxtools, False)
    a1.title.set_text("dx")
    a1 = f.add_subplot(1, 3, 3)
    showgrey(dytools, False)
    a1.title.set_text("dy")
    plt.show()
    print("original size", tools.shape)
    print("dx size", dxtools.shape)
    print("dy size", dytools.shape)


def gradientThresholding():
    tools = np.load("Images-npy/godthem256.npy")
    dxtools = convolve2d(tools, deltax(), 'valid')
    dytools = convolve2d(tools, deltay(), 'valid')
    gradmagntools = np.sqrt(dxtools ** 2 + dytools ** 2)
    threshold = 10
    sigma = 0.5
    newThreshold = 10

    f = plt.figure()
    f.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.rc('axes', titlesize=10)
    a1 = f.add_subplot(2, 2, 1)
    showgrey(tools, False)
    a1.title.set_text("original")
    a1 = f.add_subplot(2, 2, 2)
    showgrey((gradmagntools > threshold).astype(int), False)
    a1.title.set_text("threshold = " + str(threshold))

    smoothed = discgaussfft(tools, sigma)
    dxtools = convolve2d(smoothed, deltax(), 'valid')
    dytools = convolve2d(smoothed, deltay(), 'valid')
    gradmagntools = np.sqrt(dxtools ** 2 + dytools ** 2)
    a1 = f.add_subplot(2, 2, 3)
    showgrey(smoothed, False)
    a1.title.set_text("smoothed, sigma = " + str(sigma))
    a1 = f.add_subplot(2, 2, 4)
    showgrey((gradmagntools > newThreshold).astype(int), False)
    a1.title.set_text("after smoothing, threshold = " + str(newThreshold))
    plt.show()

if __name__ == '__main__':
    # differenceOperators()
    gradientThresholding()