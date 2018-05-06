#!/usr/bin/python

from __future__ import print_function
import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift
import cv2
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from scipy.spatial import distance
import os


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
            E = ifft2(imagefft * K)
            RES.append(E)
            wavelength = wavelength * mult

            # ax = fig.add_subplot(nscale, norient, ki)
            # ax.imshow(np.real(fftshift(logGabor)), cmap='gray')
            ki += 1

    # plt.show()
    # plt.savefig("../figs/features_{}.png".format(j))
    # plt.close()
    return np.array(RES)


if __name__ == '__main__':
    # cv2_gabor(im=texture, nscale=Num_scale, norient=Num_orien, minWaveLength=3, mult=2, sigmaOnf=0.65, dThetaOnSigma=1.5)
    # E = gaborconvolve(im=texture, nscale=Num_scale, norient=Num_orien, minWaveLength=3, mult=2, sigmaOnf=0.65, dThetaOnSigma=1.5)
    # plt.show()
    Num_scale = 4
    Num_orien = 6
    path = '../project4'
    M = 256
    s = 64
    feature_factor = 3
    texture_num = 2
    texture_name = ['A', 'B']
    gau_kernel = 7
    gau_sigma = .85
    num_features = Num_scale * Num_orien
    moment_order = 2

    N = np.zeros((texture_num, num_features, M, M))
    F = np.zeros((texture_num, num_features, M, M))
    T = np.zeros((texture_num, M, M))
    for i in range(texture_num):
        rgb_img = cv2.imread("{}/mosaic{}.bmp".format(path, texture_name[i]))
        T[i] = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        F[i] = gaborconvolve(im=T[i], nscale=Num_scale, norient=Num_orien, minWaveLength=3, mult=2, sigmaOnf=0.65, dThetaOnSigma=1.5)


