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
    rgb_map = cv2.imread("../project4/map{}.bmp".format(texture_name))
    rgb_img = cv2.imread("../project4/mosaic{}.bmp".format(texture_name))

    ground_truth = cv2.cvtColor(rgb_map, cv2.COLOR_RGB2GRAY)
    T = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

    N, M = x.shape
    limit = 21
    Z = np.zeros(shape=(N, K))

    kmeans_accuracies = []
    kmeans_costs = []

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

        seg = np.zeros(ground_truth.shape)
        seg_pixel_value = np.round(np.linspace(0, 255, num=K))
        for i in range(N):
            y_hat = np.argmax(Z[i])
            r = int(np.floor(i / float(ground_truth.shape[0])))
            c = i % ground_truth.shape[0]
            seg[r, c] = seg_pixel_value[y_hat]
        muind = [np.argmin(dis_mat[:, k]) for k in range(K)]
        ac = accuracy(truth=ground_truth, result=seg, texture_name=texture_name, t=t, initmu=initmu, algo='kmeans', mu=muind)
        kmeans_accuracies.append(ac)
        kmeans_costs.append(cost)

        for k in range(K):
            cluster = x[min_k == k]
            mu[k] = np.mean(cluster, axis=0)

        if not changed: break

    plot_cost_acc(kmeans_costs, kmeans_accuracies, initmu, algo="kmeans")
    # calculate sigma and alpha for EM initialization
    Nk = np.sum(Z, axis=0, dtype=np.float)
    alpha = Nk / N
    sigma = np.zeros(shape=(K, M, M))
    for k in range(K):
        d = x[min_k == k] - mu[k]
        sigma[k] = d.T.dot(d) / Nk[k]
    return Z, mu, sigma, alpha


def plot_cost_acc(costs, accuracies, initmu, algo):
    costs = costs[1:]
    accuracies = accuracies[1:]
    fig = plt.figure(figsize=(5, 10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.plot(costs)
    ax2.plot(np.array(accuracies) * 100.)
    ax1.set_title("{}: obj func with {} initialization ".format(algo, initmu))
    ax2.set_title("{}: acc(%) with {} initialization ".format(algo, initmu))
    ax1.set_xlim((0, 20))
    ax2.set_xlim((0, 20))
    ax2.set_ylim((50, 100))
    plt.savefig("../figs/{}{}/perf_{}{}_{}.png".format(algo, texture_name, algo, texture_name, initmu))


def expectation_maximization(x, K, mu, sigma, alpha, texture_name):
    rgb_map = cv2.imread("../project4/map{}.bmp".format(texture_name))
    rgb_img = cv2.imread("../project4/mosaic{}.bmp".format(texture_name))

    ground_truth = cv2.cvtColor(rgb_map, cv2.COLOR_RGB2GRAY)
    T = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

    limit = 21
    N, M = x.shape
    likelihood = np.zeros(shape=(N, K))
    start = 0
    rgb_map = cv2.imread("../project4/map{}.bmp".format(texture_name))
    rgb_img = cv2.imread("../project4/mosaic{}.bmp".format(texture_name))

    ground_truth = cv2.cvtColor(rgb_map, cv2.COLOR_RGB2GRAY)
    T = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

    em_accuracies = []
    em_costs = []

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

        if np.abs(prev_cost-cost) < 0.2:
            break
        prev_cost = cost

        Z = np.apply_along_axis(func1d=lambda xx: xx/xx.sum(), axis=1, arr=likelihood)

        seg = np.zeros(ground_truth.shape)
        seg_pixel_value = np.round(np.linspace(0, 255, num=K))
        for i in range(N):
            y_hat = np.argmax(Z[i])
            r = int(np.floor(i / float(ground_truth.shape[0])))
            c = i % ground_truth.shape[0]
            seg[r, c] = seg_pixel_value[y_hat]
        muind = [np.argmax(likelihood[:, k]) for k in range(K)]
        ac = accuracy(truth=ground_truth, result=seg, texture_name=texture_name, t=t, initmu=initmu, algo='em', mu=muind)
        em_accuracies.append(ac)
        em_costs.append(cost)

        # print("M step")
        Nk = np.sum(Z, axis=0)
        alpha = np.mean(Z, axis=0)

        for k in range(K):
            ex = (Z[:, k][:, None]) * x
            mu[k] = np.mean(ex, axis=0)

        for k in range(K):
            d = Z[:, k][:, None] * (x - mu[k])
            var = d.T.dot(d)
            sigma[k] = var / Nk[k]
        plot_cost_acc(em_costs, em_accuracies, initmu, algo="em")
    return Z


def accuracy(truth, result, texture_name, t, initmu, algo, mu):
    X, Y = truth.shape
    Z = np.zeros((256, 256))
    for i in range(X):
        for j in range(Y):
            p = int(truth[i, j])
            q = int(result[i, j])
            Z[p, q] += 1.
    T = np.sum(np.max(Z, axis=0))
    per = T / X / Y
    print("Accuracy: {}".format(per))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    markers = [(int(m / 256.), m%256) for m in mu]
    x, y = zip(*markers)
    ax1.plot(x, y, 'o', color='r')

    ax1.imshow(result, cmap='gray')
    plt.title("{}:t={} map{} using {} initialization acc={:.2%}".format(algo, t, texture_name, initmu, per))
    plt.savefig("../figs/{}{}/{}{}_{}_prog_{}.png".format(algo, texture_name, algo, initmu, t, texture_name))
    return per


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

    rgb_map = cv2.imread("../project4/map{}.bmp".format(texture_name))
    rgb_img = cv2.imread("../project4/mosaic{}.bmp".format(texture_name))

    ground_truth = cv2.cvtColor(rgb_map, cv2.COLOR_RGB2GRAY)
    T = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

    if texture_name == 'A':
        F = gaborconvolve(im=T, nscale=Num_scale, norient=Num_orien, minWaveLength=3, mult=2, sigmaOnf=0.65, dThetaOnSigma=1.5)
    if texture_name == 'B':
        F = gaborconvolve(im=T, nscale=Num_scale, norient=Num_orien, minWaveLength=3, mult=2, sigmaOnf=0.65, dThetaOnSigma=1.5)

    F = np.absolute(F)

    for i in range(num_channels):
        F[i] = (F[i] - F[i].min()) / (F[i].max()-F[i].min())

    x = np.zeros(shape=(N, num_channels))
    xi = 0
    for i in range(n):
        for j in range(n):
            x[xi, :] = F[:, i, j]
            xi += 1

    N, M = x.shape
    # init mu
    mu = np.zeros(shape=(K, M))

    if initmu == "good":
        if texture_name == 'A':
            mu[0, :] = F[:, 68, 56]
            mu[1, :] = F[:, 57, 188]
            mu[2, :] = F[:, 193, 49]
            mu[3, :] = F[:, 194, 197]
        elif texture_name == 'B':
            mu[0, :] = F[:, 50, 60]
            mu[1, :] = F[:, 128, 128]
            mu[2, :] = F[:, 225, 225]
    elif initmu == "random":
        mu[0, :] = F[:, np.random.randint(low=1, high=n-1), np.random.randint(low=10, high=n-2)]
        mu[1, :] = F[:, np.random.randint(low=1, high=n-1), np.random.randint(low=10, high=n-2)]
        mu[2, :] = F[:, np.random.randint(low=1, high=n-1), np.random.randint(low=10, high=n-2)]
        if texture_name == 'A':
            mu[3, :] = F[:, np.random.randint(low=10, high=n-3), np.random.randint(low=0, high=n-5)]
    elif initmu == "bad":
        if texture_name == 'A':
            mu[0, :] = F[:, 25, 30]
            mu[1, :] = F[:, 30, 25]
            mu[2, :] = F[:, 35, 20]
            mu[3, :] = F[:, 50, 30]
        elif texture_name == 'B':
            mu[0, :] = F[:, 120, 125]
            mu[1, :] = F[:, 130, 128]
            mu[2, :] = F[:, 125, 135]

    print("K Means:")
    Z, m1, sigma, alpha = k_means(K, x, mu, texture_name)

    initmu = "good"
    if initmu == "good":
        if texture_name == 'A':
            mu[0, :] = F[:, 68, 56]
            mu[1, :] = F[:, 57, 188]
            mu[2, :] = F[:, 193, 49]
            mu[3, :] = F[:, 194, 197]
        elif texture_name == 'B':
            mu[0, :] = F[:, 50, 60]
            mu[1, :] = F[:, 128, 128]
            mu[2, :] = F[:, 225, 225]
    elif initmu == "random":
        pass
    elif initmu == "bad":
        if texture_name == 'A':
            mu[0, :] = F[:, 25, 30]
            mu[1, :] = F[:, 30, 25]
            mu[2, :] = F[:, 35, 20]
            mu[3, :] = F[:, 50, 30]
        elif texture_name == 'B':
            mu[0, :] = F[:, 120, 125]
            mu[1, :] = F[:, 130, 128]
            mu[2, :] = F[:, 125, 135]
    # Expectation Maximization EM
    print("EM:")
    expectation_maximization(x=x, K=K, mu=mu, sigma=sigma, alpha=alpha, texture_name=texture_name)


if __name__ == "__main__":
    for texture_name in ["A", "B"]:
        for initmu in ["random", "good", "bad"]:
            K = 4 if texture_name == 'A' else 3
            test_case(texture_name, K=K, initmu=initmu)
    # plt.show()
    # plt.close()

