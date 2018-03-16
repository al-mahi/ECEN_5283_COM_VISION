#!/usr/bin/python

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.stats import kurtosis, skew
from scipy.spatial import distance
from scipy.ndimage.filters import correlate1d
import os


def gaussian_filter1d(img, sigma, axis, k=None):
    # if k is None: k = int(4.0 * sigma + .5)
    if img.shape[0] == 8:
        sigma = .883
    if img.shape[0] == 16:
        sigma = .865
    if img.shape[0] == 32:
        sigma = .854
    if img.shape[0] == 64:
        sigma = .8525
    if img.shape[0] == 80:
        sigma = .825
    if img.shape[0] == 160:
        sigma = .848
    if img.shape[0] == 320:
        sigma = .849
    if img.shape[0] == 640:
        sigma = .847

    w = np.zeros((2 * k + 1))
    w[k] = 1.
    var = sigma * sigma
    for x in range(1, k + 1):
        v = np.exp(-0.5 * x**2. / var)
        w[k + x] = v
        w[k - x] = v
    if img.shape[0] == 8:
        w[0] = w[-1] = 0.
    w /= np.sum(w)
    return correlate1d(img, weights=w, axis=axis)


def gaussian_filter(img, sigma, k=None):
    # Gaussian filtering is separable. 2N vs N^2.
    for x in range(2):
        img = gaussian_filter1d(img, sigma, axis=x, k=k)
    return img


def pyramid_laplace(image, max_layer=-1, k=None, sigma=None):
    m = image.shape[0]
    smoothed = gaussian_filter(image, sigma, k=k)
    yield image - smoothed

    for i in range(max_layer-1):
        down_sampled = cv2.resize(smoothed, dsize=(m/2, m/2), interpolation=cv2.INTER_LINEAR)
        smoothed = gaussian_filter(down_sampled, sigma, k=k)
        m = down_sampled.shape[0]
        yield down_sampled - smoothed


def laplace_pyramid(num, prefix, test_img, gau_kernel, gau_sigma, gau_layer, moment_order, block_i=0):
    k = int((gau_kernel - 1) / 2.)
    laplace_layers = list(pyramid_laplace(image=test_img, sigma=gau_sigma, k=k, max_layer=gau_layer))
    if moment_order == 1: moment = lambda x: np.mean(x)
    if moment_order == 2: moment = lambda x: np.var(x)
    if moment_order == 3: moment = lambda x: skew(x, axis=None)
    if moment_order == 4: moment = lambda x: kurtosis(x, axis=None)
    features = map(moment, laplace_layers)
    # moment = lambda x: np.mean(x)
    # features.extend(map(moment, laplace_layers))

    # moment = lambda x: skew(x, axis=None)
    # features.extend(map(moment, laplace_layers))

    # moment = lambda x: kurtosis(x, axis=None)
    # features.extend(map(moment, laplace_layers))

    # for plotting
    # if prefix=="D":
    #     fig = plt.figure(num, frameon=False)
    #     fig.set_size_inches(14, 7)
    #     ax1 = fig.add_subplot(111)
    #     ax1.set_axis_off()
    #     l0 = laplace_layers[0]
    #     l4 = np.vstack((
    #         laplace_layers[1],
    #         np.hstack((laplace_layers[2], np.zeros((160, 160)))),
    #         np.hstack((laplace_layers[3], np.zeros((80, 160+80)))),
    #         np.zeros((80, 320))
    #     ))
    #     plotl= np.hstack((l0, l4))
    #     ax1.imshow(plotl, cmap='gray')
    # #     plt.title("D{}:k={} sig={} stat={}.Left is Gaussian Right is Lapalcian Pyramid".format(
    # #         num, gau_kernel, gau_sigma, moment_order))
    #     plt.savefig("../figs/D{}/{}_k_{}_sig_{}_stat_{}.png".format(
    #         num+1, "lib", gau_kernel, gau_sigma, moment_order), bbox_inches='tight', pad_inches=0)
    #     plt.close()
    # if prefix == "BLK":
    #     fig = plt.figure(num, frameon=False)
    #     fig.set_size_inches(4, 2)
    #     ax1 = fig.add_subplot(111)
    #     ax1.set_axis_off()
    #     l0 = laplace_layers[0]
    #     l4 = np.vstack((
    #         laplace_layers[1],
    #         np.hstack((laplace_layers[2], np.zeros((16, 16)))),
    #         np.hstack((laplace_layers[3], np.zeros((8, 16+8)))),
    #         np.zeros((8, 32))
    #     ))
    #     plotl= np.hstack((l0, l4))
    #     ax1.imshow(plotl, cmap='gray')
    #     # plt.title("D{}:k={} sig={} stat={}. Right Half is Lapalcian Pyr of BLK{}".format(
    #     #     num, gau_kernel, gau_sigma, moment_order, block_i+1))
    #     plt.savefig("../figs/D{}/k_{}_sig_{}_stat_{}_blk_{}.png".format(
    #         num+1, gau_kernel, gau_sigma, moment_order, block_i+1), bbox_inches='tight', pad_inches=0)
    #     plt.close()
    return np.array(features)


if __name__ == '__main__':
    os.system("rm ../figs/D*/*.png")
    path = '../project3'
    M = 640
    s = 64
    feature_factor = 1
    texture_num = 59
    gau_kernel = 7
    gau_sigma = .85
    gau_layer = 4
    moment_order = 2

    N = np.zeros((texture_num, gau_layer * feature_factor))
    F = np.zeros((texture_num, gau_layer * feature_factor))
    T = np.zeros((texture_num, M, M))
    saved_feature_file = "../precomputed_features/k_{}_sig_{}_stat_{}.csv".format(
        gau_kernel, gau_sigma, moment_order)
    for i in range(texture_num):
        rgb_img = cv2.imread("{}/D{}.bmp".format(path, i + 1))
        T[i] = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        F[i] = laplace_pyramid(i, prefix="D", test_img=T[i], gau_kernel=gau_kernel, gau_sigma=gau_sigma, gau_layer=gau_layer,
                           moment_order=moment_order)
    max_moment = np.zeros(gau_layer * feature_factor)
    min_moment = np.zeros(gau_layer * feature_factor)
    for i in range(gau_layer):
        max_moment[i] = max(F[:, i])
        min_moment[i] = min(F[:, i])
        N[:, i] = (F[:, i] - min_moment[i]) / (max_moment[i] - min_moment[i])
    sum_res = 0

    for i in range(texture_num):
        f = np.zeros((100, gau_layer * feature_factor))
        n = np.zeros((100, gau_layer * feature_factor))
        block_i = 0
        for r in range(0, M, 64):
            for c in range(0, M, 64):
                block = T[i, r:r + s, c:c + s]
                rr = block_i / 10
                cc = block_i % 10
                f[block_i] = laplace_pyramid(i, block_i=block_i, prefix="BLK", test_img=block, gau_kernel=gau_kernel, gau_sigma=gau_sigma,
                                             gau_layer=gau_layer, moment_order=moment_order)
                block_i += 1
        for j in range(gau_layer):
            n[:, j] = (f[:, j] - min_moment[j]) / (max_moment[j] - min_moment[j])

        res = 0

        # d = distance.cdist(N, n, metric='euclidean')
        # d = distance.cdist(N, n, metric='correlation', VI=None)
        d = distance.cdist(N, n, metric='mahalanobis', VI=None)
        # d = distance.cdist(N, n, metric='cosine'
        res = np.sum(d.argmin(axis=0)==i)
        sum_res += res
        print("D{:02}\t{:02}".format(i + 1, res))
        # if (i+1)%20==0 or i==58: print()
    print("k={} sigma={} stat={} accuracy={:2.2f}".format(gau_kernel, gau_sigma, moment_order, 1. * sum_res / texture_num))
