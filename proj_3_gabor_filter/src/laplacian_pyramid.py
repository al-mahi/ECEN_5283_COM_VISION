#!/usr/bin/python

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import signal
# from scipy.ndimage.filters import gaussian_filter
from scipy.stats import kurtosis, skew
from scipy.spatial import distance
import os


def gaussian(i, j, k, sigma):
    v = (i - k) ** 2 + (j - k) ** 2
    return (1. / (sigma * np.sqrt(2. * np.pi))) * np.exp(-v / (2 * sigma ** 2))


def gauss_filter(input, k, sigma):
    """
    :type input: np.ndarray
    :rtype: np.ndarray
    """
    kspace = (k, k)
    K = np.zeros(kspace)

    pixels = np.ndindex(kspace)
    for i, j in pixels:
        K[i, j] = gaussian(i, j, (k - 1) / 2., sigma)
    return signal.convolve2d(input, K, mode='same')


def laplace_pyramid(num, prefix, test_img, gau_kernel, gau_sigma, gau_layer, moment_order, block_i=0):
    gauss_layers = [test_img]
    for i in range(1, gau_layer):
        smoothed = gauss_filter(input=gauss_layers[i - 1], k=gau_kernel, sigma=gau_sigma-(.05 * (i-1)))
        down_sampled = cv2.pyrDown(smoothed)
        gauss_layers.append(down_sampled)
    laplace_layers = [gauss_layers[gau_layer - 1]]
    for i in range(gau_layer - 1, 0, -1):
        up_sampled = cv2.pyrUp(gauss_layers[i])
        lap = gauss_layers[i - 1] - up_sampled
        laplace_layers.append(lap)
    if moment_order == 1: moment = lambda x: np.mean(x)
    if moment_order == 2: moment = lambda x: np.var(x)
    if moment_order == 3: moment = lambda x: skew(x, axis=None)
    if moment_order == 4: moment = lambda x: kurtosis(x, axis=None)
    features = np.array(map(moment, laplace_layers))

    # for plotting
    if prefix=="D":
        fig = plt.figure(num, frameon=False)
        fig.set_size_inches(21, 7)
        ax1 = fig.add_subplot(111)
        ax1.set_axis_off()
        # vertically stack smaller layer then horizontal stack
        g0 = gauss_layers[0]
        g1 = gauss_layers[1]
        g2 = np.hstack((np.zeros(shape=(160, 160)), gauss_layers[2]))
        g3 = np.hstack((np.zeros(shape=(80,  240)), gauss_layers[3]))
        g4 = np.zeros((640-320-160-80, 320))
        g5 = np.vstack((
            g1,
            np.vstack((
                g2,
                np.vstack((
                    g3,
                    g4
                ))))))
        plotg= np.hstack((g0, g5))

        l3 = laplace_layers[3]
        l2 = laplace_layers[2]
        l1 = np.hstack((laplace_layers[1], np.zeros(shape=(160, 160))))
        l0 = np.hstack((laplace_layers[0], np.zeros(shape=(80,  240))))
        l4 = np.zeros((640-320-160-80, 320))
        l5 = np.vstack((
            l2,
            np.vstack((
                l1,
                np.vstack((
                    l0,
                    l4
                ))))))
        plotl= np.hstack((l5, l3))
        final_plot = np.hstack((plotg, plotl))
        ax1.imshow(final_plot, cmap='gray')
        plt.title("D{}:k={} sig={} stat={}.Left is Gaussian Right is Lapalcian Pyramid".format(
            num, gau_kernel, gau_sigma, moment_order))
        plt.savefig("../figs/D{}/{}_k_{}_sig_{}_stat_{}.png".format(
            num+1, "lib", gau_kernel, gau_sigma, moment_order), bbox_inches='tight', pad_inches=0)
        plt.close()
    if prefix == "BLK" and (block_i+1)%10==0:
        fig = plt.figure(num, frameon=False)
        fig.set_size_inches(6, 2)
        ax1 = fig.add_subplot(111)
        ax1.set_axis_off()
        # vertically stack smaller layer then horizontal stack
        g0 = gauss_layers[0]
        g1 = gauss_layers[1]
        g2 = np.hstack((np.zeros(shape=(16, 16)), gauss_layers[2]))
        g3 = np.hstack((np.zeros(shape=(8,  24)), gauss_layers[3]))
        g4 = np.zeros((64-32-16-8, 32))
        g5 = np.vstack((
            g1,
            np.vstack((
                g2,
                np.vstack((
                    g3,
                    g4
                ))))))
        plotg= np.hstack((g0, g5))

        l3 = laplace_layers[3]
        l2 = laplace_layers[2]
        l1 = np.hstack((laplace_layers[1], np.zeros(shape=(16, 16))))
        l0 = np.hstack((laplace_layers[0], np.zeros(shape=( 8, 24))))
        l4 = np.zeros((64-32-16-8, 32))
        l5 = np.vstack((
            l2,
            np.vstack((
                l1,
                np.vstack((
                    l0,
                    l4
                ))))))
        plotl= np.hstack((l5, l3))
        final_plot = np.hstack((plotg, plotl))
        ax1.imshow(final_plot, cmap='gray')
        plt.title("D{}:k={} sig={} stat={}. Right Half is Lapalcian Pyr of BLK{}".format(
            num, gau_kernel, gau_sigma, moment_order, block_i+1))
        plt.savefig("../figs/D{}/k_{}_sig_{}_stat_{}_blk_{}.png".format(
            num+1, gau_kernel, gau_sigma, moment_order, block_i+1), bbox_inches='tight', pad_inches=0)
        plt.close()
    return features


if __name__ == '__main__':
    os.system("rm ../figs/D*/*.png")
    gau_kernel = 7
    gau_sigma = 0.5
    texture_num = 59
    gau_layer = 4
    moment_order = 3
    M = 640
    s = 64
    path = '../project3'
    for gau_kernel in [7]:
        for gau_sigma in [.35, .31]:
            for moment_order in [2]:
                N = np.zeros((texture_num, gau_layer))
                F = np.zeros((texture_num, gau_layer))
                T = np.zeros((texture_num, M, M))
                saved_feature_file = "../precomputed_features/k_{}_sig_{}_stat_{}.csv".format(
                    gau_kernel, gau_sigma, moment_order)
                for i in range(texture_num):
                    rgb_img = cv2.imread("{}/D{}.bmp".format(path, i + 1))
                    T[i] = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
                    F[i] = laplace_pyramid(i, prefix="D", test_img=T[i], gau_kernel=gau_kernel, gau_sigma=gau_sigma, gau_layer=gau_layer,
                                       moment_order=moment_order)

                max_moment = np.zeros(gau_layer)
                min_moment = np.zeros(gau_layer)

                for i in range(gau_layer):
                    max_moment[i] = max(F[:, i])
                    min_moment[i] = min(F[:, i])
                    N[:, i] = (F[:, i] - min_moment[i]) / (max_moment[i] - min_moment[i])

                # print(" ", end='.')
                sum_res = 0
                for i in range(texture_num):
                    f = np.zeros((100, gau_layer))
                    n = np.zeros((100, gau_layer))
                    block_i = 0
                    for r in range(0, M, 64):
                        for c in range(0, M, 64):
                            block = T[i, r:r + s, c:c + s]
                            f[block_i] = laplace_pyramid(i, block_i=block_i, prefix="BLK", test_img=block, gau_kernel=gau_kernel, gau_sigma=gau_sigma,
                                                         gau_layer=gau_layer, moment_order=moment_order)
                            block_i += 1
                    for j in range(gau_layer):
                        n[:, j] = (f[:, j] - min_moment[j]) / (max_moment[j] - min_moment[j])
                    res = 0
                    # d = distance.cdist(N, n, metric='euclidean')
                    # d = distance.cdist(N, n, metric='correlation', VI=None)
                    d = distance.cdist(N, n, metric='mahalanobis', VI=None)
                    # d = distance.cdist(N, n, metric='cosine')
                    res = np.sum(d.argmin(axis=0)==i)
                    sum_res += res
                    print("D{:02} {:03}".format(i + 1, res), end=' ')
                    if (i+1)%20==0 or i==58: print()
                print("k={} sigma={} stat={} accuracy={:2.2f}".format(gau_kernel, gau_sigma, moment_order, 1. * sum_res / texture_num))
