import numpy as np
# from numpy.fft import fft2, ifft2, fftshift
from scipy.signal import convolve2d, correlate2d
import matplotlib.pyplot as plt

from Functions import *
from gaussfft import gaussfft


def deltax():
    # central
    dxmask = [[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]]
    dxmask = np.flip(dxmask, 0)
    dxmask = np.flip(dxmask, 1)
    return dxmask


def deltay():
    # central
    dymask = [[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]]
    dymask = np.flip(dymask, 0)
    dymask = np.flip(dymask, 1)
    return dymask


def Lv(inpic, shape='same'):
    dxmask = deltax()
    dymask = deltay()
    Lx = convolve2d(inpic, dxmask, shape)
    Ly = convolve2d(inpic, dymask, shape)
    return np.sqrt(Lx ** 2 + Ly ** 2)


def Lvvtilde(inpic, shape='same'):
    dx = [[0, 0, 0, 0, 0],
          [0, 0, -0.5, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0.5, 0, 0],
          [0, 0, 0, 0, 0],]
    dx = np.flip(dx, 0)
    dx = np.flip(dx, 1)
    dy = np.transpose(dx)
    dxx = [[0, 0, 0, 0, 0],
          [0, 0, 1, 0, 0],
          [0, 0, -2, 0, 0],
          [0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0],]
    dxx = np.flip(dxx, 0)
    dxx = np.flip(dxx, 1)
    dyy = np.transpose(dxx)
    dxy = convolve2d(dx, dy, "same")
    Lx = convolve2d(inpic, dx, shape)
    Ly = convolve2d(inpic, dy, shape)
    Lxx = convolve2d(inpic, dxx, shape)
    Lxy = convolve2d(inpic, dxy, shape)
    Lyy = convolve2d(inpic, dyy, shape)
    result = Lx * Lx * Lxx + 2 * Lx * Ly * Lxy + Ly * Ly * Lyy
    return result

def LvvtildeTest():
    house = np.load("Images-npy/godthem256.npy")
    scale = [0.0001, 1, 4, 16, 64]
    f = plt.figure()
    f.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.rc('axes', titlesize=10)

    for i in range(len(scale)):
        a1 = f.add_subplot(2, 3, i + 1)
        showgrey(contour(Lvvtilde(discgaussfft(house, scale[i]), 'same')), False)
        a1.title.set_text("scale = " + str(scale[i]))
    plt.show()


def Lvvvtilde(inpic, shape='same'):
    dx = [[0, 0, 0, 0, 0],
          [0, 0, -0.5, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0.5, 0, 0],
          [0, 0, 0, 0, 0]]
    dx = np.flip(dx, 0)
    dx = np.flip(dx, 1)
    dxx = [[0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, -2, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0]]
    dxx = np.flip(dxx, 0)
    dxx = np.flip(dxx, 1)
    dy = np.transpose(dx)
    dyy = np.transpose(dxx)
    dxy = convolve2d(dx, dy, "same")
    dxxx = convolve2d(dx, dxx, "same")
    dxxy = convolve2d(dxx, dy, "same")
    dxyy = convolve2d(dxy, dy, "same")
    dyyy = convolve2d(dyy, dy, "same")
    Lx = convolve2d(inpic, dx, shape)
    Ly = convolve2d(inpic, dy, shape)
    Lxxx = convolve2d(inpic, dxxx, shape)
    Lxxy = convolve2d(inpic, dxxy, shape)
    Lxyy = convolve2d(inpic, dxyy, shape)
    Lyyy = convolve2d(inpic, dyyy, shape)
    result = Lx * Lx * Lx * Lxxx + 3 * Lx * Lx * Ly * Lxxy + 3 * Lx * Ly * Ly * Lxyy + Ly * Ly * Ly * Lyyy
    return result


def LvvvtildeTest():
    house = np.load("Images-npy/godthem256.npy")
    scale = [0.0001, 1, 4, 16, 64]
    f = plt.figure()
    f.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.rc('axes', titlesize=10)

    for i in range(len(scale)):
        a1 = f.add_subplot(4, 3, i + 1)
        showgrey((Lvvvtilde(discgaussfft(house, scale[i]), "same")<0).astype(int), False)
        a1.title.set_text("scale = " + str(scale[i]))

    for i in range(len(scale)):
        a1 = f.add_subplot(4, 3, i + 7)
        showgrey((Lvvvtilde(discgaussfft(house, scale[i]), "same")<-5).astype(int), False)
        a1.title.set_text("scale = " + str(scale[i]))
    plt.show()


def assembleSecondThird():
    house = np.load("Images-npy/godthem256.npy")
    scale = [0.0001, 1, 4, 16, 64]
    f = plt.figure(figsize=(8, 12))
    f.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.rc('axes', titlesize=10)

    for i in range(len(scale)):
        a1 = f.add_subplot(4, 3, i + 1)
        showgrey(contour(Lvvtilde(discgaussfft(house, scale[i]), 'same')), False)
        a1.title.set_text("lvvt, " + str(scale[i]))

    for i in range(len(scale)):
        a1 = f.add_subplot(4, 3, i + 7)
        showgrey((Lvvvtilde(discgaussfft(house, scale[i]), "same")<-5).astype(int), False)
        a1.title.set_text("lvvv, " + str(scale[i]))

    plt.show()


def thirdOrderTest():
    [y, x] = np.meshgrid(range(-5, 6), range(-5, 6))
    dx = [[0, 0, 0, 0, 0],
          [0, 0, -0.5, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0.5, 0, 0],
          [0, 0, 0, 0, 0]]
    dx = np.flip(dx, 0)
    dx = np.flip(dx, 1)
    dxx = [[0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, -2, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0]]
    dxx = np.flip(dxx, 0)
    dxx = np.flip(dxx, 1)
    dy = np.transpose(dx)
    dyy = np.transpose(dxx)
    dxy = convolve2d(dx, dy, "same")
    dxxx = convolve2d(dx, dxx, "same")
    dxxy = convolve2d(dxx, dy, "same")
    dxyy = convolve2d(dxy, dy, "same")
    dyyy = convolve2d(dyy, dy, "same")
    print(convolve2d(x ** 3, dxxx, "valid"))
    print(convolve2d(x ** 3, dxx, "valid"))
    print(convolve2d(x ** 2 * y, dxxy, "valid"))


def extractedge(inpic, scale, threshold, shape):
    gaussianSmooth = discgaussfft(inpic, scale)
    gradmagn = Lv(gaussianSmooth, "same")

    Lvv = Lvvtilde(gaussianSmooth, shape)
    Lvvv = Lvvvtilde(gaussianSmooth, shape)

    Lvmask = gradmagn > threshold
    LvvvMask = Lvvv < 0
    curves = zerocrosscurves(Lvv, LvvvMask)
    contours = thresholdcurves(curves, Lvmask)
    return contours


def houghline(curves, magnitude, nrho, ntheta, threshold, nlines=20, verbose=False):
    acc = np.zeros((nrho, ntheta))
    x, y = magnitude.shape
    r = np.sqrt(x * x + y * y)
    rho = np.linspace(-r, r, nrho)
    theta = np.linspace(-np.pi/2, np.pi/2, ntheta)


    # for i in range(len(curves)):
    #
    #     for j in

    linepar = []


    return linepar, acc


def houghedgeline(pic, scale, gradmagnthreshold, nrho, ntheta, nlines=20, verbose=False):
    curves = extractedge(pic, scale, gradmagnthreshold, "same")
    gaussianSmooth = discgaussfft(pic, scale)
    gradmagn = Lv(gaussianSmooth, "same")

    linepar, acc = houghline(curves, gradmagn, nrho, ntheta, gradmagnthreshold, nlines, verbose)
    return linepar, acc


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
    threshold = [6, 10, 25]
    sigma = 0.5

    f = plt.figure(dpi= 160, figsize=(10,8))
    f.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.rc('axes', titlesize=10)
    a1 = f.add_subplot(3, 3, 1)
    showgrey(tools, False)
    a1.title.set_text("original")
    a1 = f.add_subplot(3, 3, 2)
    showgrey(gradmagntools, False)
    a1.title.set_text("gradient magnitudes")
    a1 = f.add_subplot(3, 3, 3)
    hist, bin = np.histogram(gradmagntools)
    plt.plot(bin[1:], hist)
    a1.title.set_text("histogram")

    for i in range(3):
        a1 = f.add_subplot(3, 3, 4 + i)
        showgrey((gradmagntools > threshold[i]).astype(int), False)
        a1.title.set_text("threshold = " + str(threshold[i]))

    smoothed = discgaussfft(tools, sigma)
    gradmagntools = Lv(smoothed)
    a1 = f.add_subplot(3, 3, 7)
    showgrey(smoothed, False)
    a1.title.set_text("smoothed, sigma = " + str(sigma))
    a1 = f.add_subplot(3, 3, 8)
    showgrey((gradmagntools > 10).astype(int), False)
    a1.title.set_text("after smoothing, threshold = " + str(2))
    a1 = f.add_subplot(3, 3, 9)
    showgrey((gradmagntools > 25).astype(int), False)
    a1.title.set_text("after smoothing, threshold = " + str(6))
    plt.show()


def extraction():
    scale = 1.5
    threshold = 10
    tools = np.load("Images-npy/godthem256.npy")
    edgecurves = extractedge(tools, scale, threshold, "same")

    f = plt.figure()
    f.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.rc('axes', titlesize=10)
    a1 = f.add_subplot(1, 2, 1)
    overlaycurves(tools, edgecurves)
    a1.title.set_text("scale " + str(scale) + ", threshold " +str(threshold))

    tools = np.load("Images-npy/few256.npy")
    edgecurves = extractedge(tools, scale, threshold, "same")

    a1 = f.add_subplot(1, 2, 2)
    overlaycurves(tools, edgecurves)
    a1.title.set_text("scale " + str(scale) + ", threshold " + str(threshold))
    plt.show()

if __name__ == '__main__':
    # differenceOperators()
    # gradientThresholding()
    # LvvtildeTest()
    # thirdOrderTest()
    # LvvvtildeTest()
    # assembleSecondThird()
    extraction()