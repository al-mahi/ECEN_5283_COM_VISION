#!/usr/bin/python

from __future__ import print_function
import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift
import cv2
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from scipy.spatial import distance
import os
# matlab comments


def cv2_gabor(im, nscale, norient, minWaveLength, mult, sigmaOnf, dThetaOnSigma):
    fig = plt.figure(0, frameon=False)
    ki = 1
    for s in range(nscale):
        waveLength = minWaveLength
        for orientation in range(norient):
            ang = orientation * np.pi / norient  # Calculate filter angle.
            K = cv2.getGaborKernel(ksize=(33, 33), sigma=(s+1)*2, theta=ang, lambd=waveLength, gamma=sigmaOnf, psi=0.)
            waveLength *= mult

            ax = fig.add_subplot(nscale, norient, ki)
            ax.imshow(np.real(K), cmap='gray')
            ki += 1


def gaborconvolve(im, nscale, norient, minWaveLength, mult, sigmaOnf, dThetaOnSigma):
    """
    :type im: np.ndarray
    :param nscale:
    :param norient:
    :param minWaveLength:
    :param mult:
    :param sigmaOnf:
    :param dThetaOnSigma:
    :return:
    """
    # fig = plt.figure(0, frameon=False)

    M, N = im.shape
    imagefft = fft2(im)  # Fourier transform of image
    E = np.empty((nscale, norient))
    RES = []

    X = Y = np.linspace(start=-1., stop=1., num=N)
    x, y = np.meshgrid(X, Y)

    radius = x**2. + y**2.
    radius[M/2, N/2] = 1.
    theta = np.arctan2(-y, x)
    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    # Calculate the standard deviation of the
    # angular Gaussian function used to
    # construct filters in the freq plane.
    thetaSigma = np.pi / norient / dThetaOnSigma

    ki = 1
    for orientation in range(norient):
        # print('Processing orientation {}'.format(orientation));
        ang = orientation * np.pi / norient  # Calculate filter angle.
        wavelength = minWaveLength  # Initialize filter wavelength.
        ds = sintheta * np.cos(ang) - costheta * np.sin(ang)  # Difference in sine.
        dc = costheta * np.cos(ang) + sintheta * np.sin(ang)  # Difference in cosine.
        dtheta = np.abs(np.arctan2(-ds, dc))  # Absolute angular distance.
        spread = np.exp((-dtheta**2.) / (2. * thetaSigma**2.))  # Calculate the angular filter component.

        for s in range(nscale):
            # Construct the filter - first calculate the radial filter component.
            fo = 1.0 / wavelength  # Centre frequency of filter.
            rfo = fo / 0.5  # Normalised radius from centre of frequency plane corresponding to fo.
            logGabor = np.exp((-(np.log(radius / rfo))**2.) / (2. * np.log(sigmaOnf)**2))
            logGabor[M/2, N/2] = 0.
            K = fftshift(logGabor * spread)
            E = np.real(ifft2(imagefft * K))
            RES.append(E)
            wavelength = wavelength * mult

            # ax = fig.add_subplot(nscale, norient, ki)
            # ax.imshow(np.real(fftshift(logGabor)), cmap='gray')
            ki += 1

    moment1 = lambda x: np.mean(x)
    moment2 = lambda x: np.var(x)
    moment3 = lambda x: skew(x, axis=None)
    moment4 = lambda x: kurtosis(x, axis=None)
    features = []
    # features.extend(map(moment1, RES))
    features.extend(map(moment2, RES))
    features.extend(map(moment3, RES))
    features.extend(map(moment4, RES))
    return np.array(features)


if __name__ == '__main__':
    Num_scale = 4
    Num_orien = 6
    # cv2_gabor(im=texture, nscale=Num_scale, norient=Num_orien, minWaveLength=3, mult=2, sigmaOnf=0.65, dThetaOnSigma=1.5)
    # E = gaborconvolve(im=texture, nscale=Num_scale, norient=Num_orien, minWaveLength=3, mult=2, sigmaOnf=0.65, dThetaOnSigma=1.5)
    # plt.show()
    os.system("rm ../figabors/D*/*.png")
    path = '../project3'
    M = 640
    s = 64
    feature_factor = 3
    texture_num = 59
    gau_kernel = 7
    gau_sigma = .85
    gau_layer = Num_scale * Num_orien * feature_factor
    moment_order = 2

    N = np.zeros((texture_num, gau_layer))
    F = np.zeros((texture_num, gau_layer))
    T = np.zeros((texture_num, M, M))
    for i in range(texture_num):
        rgb_img = cv2.imread("{}/D{}.bmp".format(path, i + 1))
        T[i] = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        F[i] = gaborconvolve(im=T[i], nscale=Num_scale, norient=Num_orien, minWaveLength=3, mult=2, sigmaOnf=0.65, dThetaOnSigma=1.5)
    max_moment = np.zeros(gau_layer)
    min_moment = np.zeros(gau_layer)
    for i in range(gau_layer):
        max_moment[i] = max(F[:, i])
        min_moment[i] = min(F[:, i])
        N[:, i] = (F[:, i] - min_moment[i]) / (max_moment[i] - min_moment[i])
    sum_res = 0

    for i in range(texture_num):
        f = np.zeros((100, gau_layer))
        n = np.zeros((100, gau_layer))
        block_i = 0
        for r in range(0, M, 64):
            for c in range(0, M, 64):
                block = T[i, r:r + s, c:c + s]
                f[block_i] = gaborconvolve(im=block, nscale=Num_scale, norient=Num_orien, minWaveLength=3, mult=2, sigmaOnf=0.65, dThetaOnSigma=1.5)
                block_i += 1
        for j in range(gau_layer):
            n[:, j] = (f[:, j] - min_moment[j]) / (max_moment[j] - min_moment[j])

        res = 0

        # d = distance.cdist(N, n, metric='euclidean')
        # d = distance.cdist(N, n, metric='correlation', VI=None)
        # d = distance.cdist(N, n, metric='mahalanobis', VI=np.identity(gau_layer))
        d = distance.cdist(N, n, metric='cosine')
        res = np.sum(d.argmin(axis=0)==i)
        sum_res += res
        print("D{:02},{:02}".format(i + 1, res))
        # if (j+1)%20==0 or j==58: print()
    print("stat={} accuracy={:2.2f}".format(feature_factor, 1. * sum_res / texture_num))