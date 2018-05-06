#!/usr/bin/python

from __future__ import print_function

import cv2
import numpy as np
from scipy.spatial import distance
from scipy.stats import multivariate_normal
from scipy.signal import convolve2d
from bwmorph_thin import bwmorph_thin


def drawcircle(Ii, Uxy, K):
    """
    :type Ii: np.ndarray
    :type Uxy: np.ndarray
    :type K:  np.ndarray
    """
    X, Y = Ii.shape
    Io = Ii.copy()
    for i in xrange(X):
        for j in xrange(Y):
            ma = 0
            for k in xrange(K):
                cx = Uxy[k, 0]
                cy = Uxy[k, 1]
                te = (i - cx) * (i - cx) + (j - cy) * (j - cy)
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
    for k in xrange(K):
        tx, ty = Uxy[k, :]
        Ii[tx, ty] = 1.
    Io = convolve2d(Ii, T, 'same')
    It = im2bw(Io, 0.5)
    return It


def likelihood(Im, T, Uxy, K):
    X, Y = Im.shape
    It = maketarget(T, X, Y, Uxy, K)
    Ie = (Im + It * 0.25) - 0.5
    e = np.sum(Ie**2.)
    like = np.exp(-0.5 * e)
    return like


if __name__ == '__main__':
    pass