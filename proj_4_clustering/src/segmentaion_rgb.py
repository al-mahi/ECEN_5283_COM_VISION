#!/usr/bin/python

from __future__ import print_function

import cv2
import numpy as np
import matplotlib.pyplot as plt
from gaborconvolve import gaborconvolve
from scipy.spatial import distance
from scipy.stats import multivariate_normal
import pickle
import os


def k_means(K, x, mu, texture_name):
    """
    :type x: np.ndarray
    :type mu: np.ndarray
    :rtype Z: np.ndarray
    :rtype Z: np.ndarray
    """
    rgb_map = cv2.imread("../project4/map{}.jpg".format(texture_name))
    rgb_img = cv2.imread("../project4/mosaic{}.jpg".format(texture_name))

    ground_truth = cv2.cvtColor(rgb_map, cv2.COLOR_RGB2GRAY)
    T = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

    N, M = x.shape
    limit = 100
    Z = np.zeros(shape=(N, K))

    for t in range(limit):
        dis_mat = distance.cdist(x, mu, metric='euclidean')
        min_k = np.array(dis_mat.argmin(axis=1))
        min_d = np.array([dis_mat[i, min_k[i]] for i in range(N)])
        changed = False
        for i in range(N):
            if Z[i, min_k[i]] == 0:
                Z[i, :] = 0
                Z[i, min_k[i]] = 1
                changed = True

        # re calc cluster mean
        cost = np.sum(min_d)
        print("#{:03} cost={}".format(t, np.sum(min_d)))

        for k in range(K):
            cluster = x[min_k == k]
            mu[k] = np.mean(cluster, axis=0)

        if not changed: break

    seg = np.zeros(ground_truth.shape)
    seg_pixel_value = np.round(np.linspace(0, 255, num=K))
    for i in range(N):
        y_hat = np.argmax(Z[i])
        r = int(np.floor(i / float(ground_truth.shape[0])))
        c = i % ground_truth.shape[0]
        seg[r, c] = seg_pixel_value[y_hat]
    muind = [np.argmin(dis_mat[:, k]) for k in range(K)]
    accuracy(truth=ground_truth, result=seg, texture_name=texture_name, initmu=initmu, algo='kmeans', mu=muind)

    # calculate sigma and alpha for EM initialization
    Nk = np.sum(Z, axis=0, dtype=np.float)
    alpha = Nk / N
    sigma = np.zeros(shape=(K, M, M))
    for k in range(K):
        d = x[min_k == k] - mu[k]
        sigma[k] = d.T.dot(d) / Nk[k]
    return Z, mu, sigma, alpha


def expectation_maximization(x, K, mu, sigma, alpha, texture_name):
    rgb_map = cv2.imread("../project4/map{}.jpg".format(texture_name))
    rgb_img = cv2.imread("../project4/mosaic{}.jpg".format(texture_name))

    ground_truth = cv2.cvtColor(rgb_map, cv2.COLOR_RGB2GRAY)
    T = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

    limit = 100
    N, M = x.shape
    likelihood = np.zeros(shape=(N, K))
    start = 0
    rgb_map = cv2.imread("../project4/map{}.jpg".format(texture_name))
    rgb_img = cv2.imread("../project4/mosaic{}.jpg".format(texture_name))

    ground_truth = cv2.cvtColor(rgb_map, cv2.COLOR_RGB2GRAY)
    T = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

    prev_cost = np.inf
    for t in range(start, limit):
        # print("E Step")
        dist = []
        cost = 0.
        for k in range(K):
            dist.append(multivariate_normal(mean=mu[k], cov=sigma[k]))
        for n in range(N):
            for k in range(K):
                likelihood[n, k] = alpha[k] * dist[k].pdf(x=x[n])
            cost += np.log(np.sum(likelihood[n, :]))

        print("#{:04} cost={}".format(t, cost))

        if np.abs(prev_cost - cost) < 0.2:
            break
        prev_cost = cost

        Z = np.apply_along_axis(func1d=lambda xx: xx / xx.sum(), axis=1, arr=likelihood)

        # print("M step")
        Nk = np.sum(Z, axis=0)
        alpha = np.mean(Z, axis=0)

        for k in range(K):
            ex = (Z[:, k][:, None]) * x
            mu[k] = np.mean(ex, axis=0)

        for k in range(K):
            d = (x - mu[k])
            var = d.T.dot(Z[:, k][:, None] * d)
            sigma[k] = var / Nk[k]

    seg = np.zeros(ground_truth.shape)
    seg_pixel_value = np.round(np.linspace(0, 255, num=K))
    for i in range(N):
        y_hat = np.argmax(Z[i])
        r = int(np.floor(i / float(ground_truth.shape[0])))
        c = i % ground_truth.shape[0]
        seg[r, c] = seg_pixel_value[y_hat]
    muind = [np.argmax(likelihood[:, k]) for k in range(K)]
    accuracy(truth=ground_truth, result=seg, texture_name=texture_name, initmu=initmu, algo='em', mu=muind)

    return Z


def accuracy(truth, result, texture_name, initmu, algo, mu):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    markers = [(int(m / 256.), m % 256) for m in mu]
    x, y = zip(*markers)
    ax1.plot(x, y, 'o', color='r')
    ax1.imshow(result, interpolation='nearest')
    plt.savefig("../figs/{}{}/{}{}_prog_{}.png".format(algo, texture_name, algo, initmu, texture_name))


def test_case(texture_name='A', K=4, initmu="good"):
    n = 256
    N = 256 * 256
    if texture_name == "A":
        Num_scale = 5
        Num_orien = 8
    else:
        Num_scale = 5
        Num_orien = 8
    num_channels = Num_scale * Num_orien

    rgb_map = cv2.imread("../project4/map{}.jpg".format(texture_name))
    rgb_img = cv2.imread("../project4/mosaic{}.jpg".format(texture_name))

    ground_truth = cv2.cvtColor(rgb_map, cv2.COLOR_RGB2GRAY)
    T = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    F = gaborconvolve(im=T, nscale=Num_scale, norient=Num_orien, minWaveLength=3, mult=2, sigmaOnf=0.65,
                          dThetaOnSigma=1.5)

    F = np.absolute(F)

    for i in range(num_channels):
        F[i] = (F[i] - F[i].min()) / (F[i].max() - F[i].min())

    x = np.zeros(shape=(N, num_channels + 3 + 2))
    xi = 0
    print(rgb_img.shape)
    for i in range(n):
        for j in range(n):
            x[xi, :-3 - 2] = F[:, i, j]
            x[xi, -3 - 2] = rgb_img[i, j, 0] / 255.
            x[xi, -2 - 2] = rgb_img[i, j, 1] / 255.
            x[xi, -1 - 2] = rgb_img[i, j, 2] / 255.
            x[xi, -2] = i / 255.
            x[xi, -1] = j / 255.
            xi += 1

    N, M = x.shape
    # init mu
    mu = np.zeros(shape=(K, M))

    if initmu == "good":
        if texture_name == 'C':
            mu[0] = x[(150 -1)*255 + 175]
            mu[1] = x[(150 -1)*255 + 150]
            mu[2] = x[(100-1)*255 + 100]
            mu[3] = x[(25-1)*255 + 5]
            mu[4] = x[(200 - 1) * 255 + 75]
            mu[5] = x[(150 - 1) * 255 + 5]
            # mu[6] = x[(175 - 1) * 255 + 20]
            # mu[7] = x[(50 - 1) * 255 + 75]
            # mu[8] = x[(25 - 1) * 255 + 240]
            # mu[9] = x[(25 - 1) * 255 + 240]
        elif texture_name == 'D':
            mu[0, :] = x[(150 - 1)*255 + 160]
            mu[1, :] = x[(90 - 1)*255 + 27]
            mu[2, :] = x[(100 - 1)*255 + 38]
            mu[3, :] = x[(34 - 1) * 255 + 186]

    print("K Means:")
    Z, m1, sigma, alpha = k_means(K, x, mu, texture_name)
    print("EM:")
    expectation_maximization(x=x, K=K, mu=mu, sigma=sigma, alpha=alpha, texture_name=texture_name)


if __name__ == "__main__":
    for texture_name in ["D"]:
        for initmu in ["good"]:
            K = 6 if texture_name == 'C' else 4
            test_case(texture_name, K=K, initmu=initmu)
    # plt.show()
    plt.close()
