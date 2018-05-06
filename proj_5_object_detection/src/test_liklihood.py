#!/usr/bin/python

from __future__ import print_function

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
# from mcmc_functions import likelihood, drawcircle
from matplotlib.animation import FuncAnimation
from scipy.signal import convolve2d, convolve
from scipy.stats import poisson


def drawcircle(Ii, Uxy, K):
    """
    :type Ii: np.ndarray
    :type Uxy: np.ndarray
    :type K:  np.ndarray
    """

    X, Y = Ii.shape
    Io = Ii.copy()
    for i in range(X):
        for j in range(Y):
            ma = 0
            for k in range(K):
                cx = Uxy[k, 0]
                cy = Uxy[k, 1]
                te = (i - cx)**2 + (j - cy)**2
                tr = abs(te - 100)
                if tr < 10:
                    ma = 1
                    break
            if ma == 1:
                Io[i, j] = 1
            else:
                Io[i, j] = Ii[i, j]
    return Io


def im2bw(I, threshold):
    """
    :type I: np.ndarray
    :type threshold: np.ndarray
    :return:
    """
    binary_image = np.zeros(I.shape)
    ind = I > threshold
    binary_image[ind] = 1
    return binary_image


def maketarget(T, Ix, Iy, Cxy, K):
    Ii = np.zeros((Ix, Iy))
    Uxy = Cxy.copy()
    Uxy = np.clip(Uxy, 0, Ix-1)
    for k in range(K):
        tx, ty = Cxy[k, :]
        Ii[tx, ty] = 1.
    Io = convolve2d(Ii, T, 'same')
    It = im2bw(Io, 0.5)
    return It


def likelihood(Im, T, Uxy, K):
    """
    :type Im: np.ndarray
    :type T: np.ndarray
    :type Uxy: np.ndarray
    :type K: np.ndarray
    :rtype: np.ndarray
    """
    X, Y = Im.shape
    It = maketarget(T, X, Y, Uxy, K)
    # plt.imshow(It, cmap='gray')
    # plt.show()
    Ie = (Im + It * 0.25) - 0.5
    # plt.imshow(Ie, cmap='gray')
    # plt.show()
    e = np.sum(Ie**2.)
    like = np.exp(-0.5 * e)
    return like


def test_lik(path_im, path_target):
    in_img = cv2.imread("../project5/{}".format(path_im))
    in_tar = cv2.imread("../project5/{}".format(path_target))
    I = cv2.cvtColor(in_img, cv2.COLOR_RGB2GRAY) / 255.
    T = cv2.cvtColor(in_tar, cv2.COLOR_RGB2GRAY) / 255.

    # Uxy1 = np.array([
    #     [50, 80],
    #     [34, 103],
    #     [95, 105]
    # ])

    Uxy1 = np.array([
        [50, 80],
        [50, 80],
        [50, 80]
    ])

    Uxy2 = np.array([
        [37, 25],
        [80, 50],
        [50, 70]
    ])

    Uxy3 = np.array([
        [50, 80],
        [34, 103],
        [82, 50]
    ])

    Uxy4 = np.array([
        [50, 80],
        [34, 103],
        [90, 78]
    ])

    Uxy5 = np.array([
        [50, 80],
        [34, 103]
    ])

    Uxy6 = np.array([[95, 105]])

    Uxy7 = np.array([
        [50, 80],
        [34, 103],
        [95, 95]
    ])
    Uxy8 = np.array([
        [50, 80],
        [34, 103],
        [30, 85]
    ])
    Uxy9 = np.array([
        [55, 82],
        [33, 100]
    ])

    lik1 = likelihood(I, T, Uxy=Uxy1, K=3)
    lik2 = likelihood(I, T, Uxy=Uxy2, K=3)
    lik3 = likelihood(I, T, Uxy=Uxy3, K=3)
    lik4 = likelihood(I, T, Uxy=Uxy4, K=3)
    lik5 = likelihood(I, T, Uxy=Uxy5, K=2)
    lik6 = likelihood(I, T, Uxy=Uxy6, K=1)
    lik7 = likelihood(I, T, Uxy=Uxy7, K=3)
    lik8 = likelihood(I, T, Uxy=Uxy8, K=3)
    lik9 = likelihood(I, T, Uxy=Uxy9, K=2)

    prior = poisson.pmf(k=3, mu=3)

    fig = plt.figure(figsize=(16,16))
    ax1 = fig.add_subplot(331)
    ax2 = fig.add_subplot(332)
    ax3 = fig.add_subplot(333)
    ax4 = fig.add_subplot(334)
    ax5 = fig.add_subplot(335)
    ax6 = fig.add_subplot(336)
    ax7 = fig.add_subplot(337)
    ax8 = fig.add_subplot(338)
    ax9 = fig.add_subplot(339)

    I1 = drawcircle(I, Uxy=Uxy1, K=3)
    I2 = drawcircle(I, Uxy=Uxy2, K=3)
    I3 = drawcircle(I, Uxy=Uxy3, K=3)
    I4 = drawcircle(I, Uxy=Uxy4, K=3)
    I5 = drawcircle(I, Uxy=Uxy5, K=2)
    I6 = drawcircle(I, Uxy=Uxy6, K=1)
    I7 = drawcircle(I, Uxy=Uxy7, K=3)
    I8 = drawcircle(I, Uxy=Uxy8, K=2)
    I9 = drawcircle(I, Uxy=Uxy9, K=1)

    ax1.imshow(I1, cmap='gray')
    ax2.imshow(I2, cmap='gray')
    ax3.imshow(I3, cmap='gray')
    ax4.imshow(I4, cmap='gray')
    ax5.imshow(I5, cmap='gray')
    ax6.imshow(I6, cmap='gray')
    ax7.imshow(I7, cmap='gray')
    ax8.imshow(I8, cmap='gray')
    ax9.imshow(I9, cmap='gray')

    ax1.set_title("K={} Likelihood={:.4e}".format(3, lik1, lik1 * prior))
    ax2.set_title("K={} Likelihood={:.4e}".format(3, lik2, lik2 * prior))
    ax3.set_title("K={} Likelihood={:.4e}".format(3, lik3, lik3 * prior))
    ax4.set_title("K={} Likelihood={:.4e}".format(3, lik4, lik4 * prior))
    ax5.set_title("K={} Likelihood={:.4e}".format(2, lik5, lik5 * prior))
    ax6.set_title("K={} Likelihood={:.4e}".format(1, lik6, lik6 * prior))
    ax7.set_title("K={} Likelihood={:.4e}".format(3, lik7, lik7 * prior))
    ax8.set_title("K={} Likelihood={:.4e}".format(2, lik8, lik8 * prior))
    ax9.set_title("K={} Likelihood={:.4e}".format(1, lik9, lik9 * prior))

    plt.savefig("../report/likelihood_fault.png")
    plt.show()


if __name__ == '__main__':
    test_lik("discs3.bmp", "target.bmp")