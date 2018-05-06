#!/usr/bin/python

from __future__ import print_function

import cv2
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from scipy.stats import poisson
from mcmc_functions import likelihood, drawcircle
import os


if __name__ == '__main__':
    m = 12
    n = 5
    P = 72  # top eigen vectors make it multiple of 3 to avoid exception in plotting arrangemet
    I = None
    for j in range(m):
        for k in range(n):
            path = "../project6/{}/{}_{}.bmp".format(j + 1, j + 1, k)
            rgb_img = cv2.imread(path)
            img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
            if I is None: I = img.ravel()
            else: I = np.vstack((I, img.ravel()))

    mu = np.mean(I, axis=0)
    F = I - mu
    C = F.T.dot(F)  # variance ghodsi lec
    D, U = sparse.linalg.eigs(C, k=P)
    U = np.real(U)

    fig = plt.figure()
    ax = [None for _ in range(P)]
    for i in range(P):
        ax[i] = fig.add_subplot(3, P/3, i+1)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].imshow(U[:, i].reshape(90, 60), cmap='gray')
    fig.suptitle("Eigen Faces")
    fig.savefig("eigen_face_U_{}.png".format(P))

    W = np.zeros(shape=(m, P))
    for j in range(m):
        for i in range(P):
            alpha = 0
            for k in range(n):
                alpha += np.dot(I[j*n + k] - mu, U[:, i])
            W[j, i] = alpha / n

    I_hat = [None for _ in range(m)]

    fig2 = plt.figure(2)
    ax2 = [None for _ in range(m)]
    for j in range(m):
        I_hat[j] = mu.copy()
        for i in range(P):
            I_hat[j] += np.dot(W[j, i], U[:, i])
        ax2[j] = fig2.add_subplot(3, m/3, j+1)
        ax2[j].set_xticks([])
        ax2[j].set_yticks([])
        ax2[j].imshow(I_hat[j].reshape(90, 60), cmap='gray')

    fig2.suptitle("Representative Images")
    fig2.savefig("Representative_Images_U_{}.png".format(P))
    # plt.show()

    # test image
    rank_mat = np.zeros(shape=(m, P))
    min_distance_known_face = []
    min_distance_unknown_face = []
    min_distance_nonface = []
    for j in range(m):
        for k in range(n, 10):
            path = "../project6/{}/{}_{}.bmp".format(j + 1, j + 1, k)
            rgb_img = cv2.imread(path)
            img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
            It = img.ravel()
            Wt = np.zeros(P)
            for i in range(P):
                beta = np.dot(It - mu, U[:, i])
                Wt[i] = beta
            It_hat = mu.copy()
            for i in range(P):
                It_hat += np.dot(Wt[i], U[:, i])
            d = np.linalg.norm(W - Wt, axis=1)
            ranks = d.argsort() + 1
            min_distance_known_face.append(d[ranks[0]-1])
            # print("j={:02d}_{:02d} {} {}".format(j+1, k+1, ranks, j+1 == ranks))
            for r in range(P):
                if j+1 in ranks[:r]:
                    rank_mat[j, r] += 1.
    rank_mat /= (10-n)
    rank_mat *= 100
    np.savetxt("rank_mat_P_{}.csv".format(P), rank_mat[:, 1:4])
    print(rank_mat[:, 1:4])

    fig8 = plt.figure(8)
    ax8 = [None for _ in range(6)]
    for j in range(3):
        path = "../project6/unknown/{}.jpg".format(j + 1)
        rgb_img = cv2.imread(path)
        img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        It = img.ravel()
        Wt = np.zeros(P)
        for i in range(P):
            beta = np.dot(It - mu, U[:, i])
            Wt[i] = beta
        It_hat = mu.copy()
        for i in range(P):
            It_hat += np.dot(Wt[i], U[:, i])
        d = np.linalg.norm(W - Wt, axis=1)
        ranks = d.argsort() + 1
        min_distance_unknown_face.append(d[ranks[0]-1])

        ax8[j] = fig8.add_subplot(3, 2, j*2 + 1)
        ax8[j].imshow(It.reshape(90, 60), cmap='gray')
        ax8[j].set_xticks([])
        ax8[j].set_yticks([])
        ax8[j + 1] = fig8.add_subplot(3, 2, j*2 + 2)
        ax8[j + 1].imshow(It_hat.reshape(90, 60), cmap='gray')
        ax8[j + 1].set_title(d[ranks[0]-1])
        ax8[j + 1].set_xticks([])
        ax8[j + 1].set_yticks([])
    fig8.suptitle("Unknown Face Reconstruction with cost")
    fig8.savefig("Unknown_Face_cost_P_{}.png".format(P))


    fig9 = plt.figure(9)
    ax9 = [None for _ in range(6)]
    for j in range(3):
        path = "../project6/nonface/{}.jpg".format(j + 1)
        rgb_img = cv2.imread(path)
        img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        It = img.ravel()
        Wt = np.zeros(P)
        for i in range(P):
            beta = np.dot(It - mu, U[:, i])
            Wt[i] = beta
        It_hat = mu.copy()
        for i in range(P):
            It_hat += np.dot(Wt[i], U[:, i])
        d = np.linalg.norm(W - Wt, axis=1)
        ranks = d.argsort() + 1
        min_distance_nonface.append(d[ranks[0]-1])
        ax9[j] = fig9.add_subplot(3, 2, j * 2 + 1)
        ax9[j].imshow(It.reshape(90, 60), cmap='gray')
        ax9[j].set_xticks([])
        ax9[j].set_yticks([])
        ax9[j + 1] = fig9.add_subplot(3, 2, j * 2 + 2)
        ax9[j + 1].imshow(It_hat.reshape(90, 60), cmap='gray')
        ax9[j + 1].set_title(d[ranks[0] - 1])
        ax9[j + 1].set_xticks([])
        ax9[j + 1].set_yticks([])
    fig9.suptitle("Non Face Reconstruction with cost")
    fig9.savefig("Non_Face_cost_P_{}.png".format(P))

    fig3 = plt.figure(3)
    ax = fig3.add_subplot(111)
    ax.scatter(range(1, 61), min_distance_known_face, color='green', label='known face')
    ax.scatter(range(61, 64), min_distance_unknown_face, color='yellow', label='unknown face')
    ax.scatter(range(64, 67), min_distance_nonface, color='red', label='nonface')
    ax.legend()
    ax.axhline(y=min(min_distance_unknown_face), label='unkown face', color='yellow')
    ax.axhline(y=min(min_distance_nonface), label='non face', color='red')
    fig3.savefig("thresholds.png")
    # plt.show()

    DD, UU = sparse.linalg.eigs(C, k=30)
    DD = np.real(DD)

    Y = np.zeros(30)
    for i in range(30):
        Y[i] = DD[i]/np.sum(DD)

    fig4 = plt.figure(4)
    ax = fig4.add_subplot(111)
    ax.set_xlabel("# of Components")
    ax.set_ylabel("Eigen value")
    ax.plot(range(30), Y)
    fig4.savefig("Eigen values")

    Z = np.zeros(30)
    for i in range(30):
        Z[i] = np.sum(DD[:i+1])/np.sum(DD)

    fig5 = plt.figure(5)
    ax = fig5.add_subplot(111)
    ax.plot(range(30), Z * 100)
    ax.set_xlabel("# of Components")
    ax.set_ylabel("PCA Accuracy(%)")
    fig5.savefig("PCA accuracy")

    print(I.shape)
    fig6 = plt.figure(6)
    ax6 = [None for _ in range(m)]
    for i in range(m):
        mm = np.mean(I[i*5:i*5+5, :], axis=0)
        ax6[i] = fig6.add_subplot(3, 4, i + 1)
        ax6[i].set_xticks([])
        ax6[i].set_yticks([])
        ax6[i].imshow(mm.reshape(90, 60), cmap='gray')
    fig6.savefig("mean_image.png")

    fig7 = plt.figure(7)
    ax7 = [None for _ in range(m)]

    for j in range(m):
        k = np.random.randint(low=6, high=9)
        path = "../project6/{}/{}_{}.bmp".format(j + 1, j + 1, k)
        rgb_img = cv2.imread(path)
        img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        It = img.ravel()
        Wt = np.zeros(P)
        for i in range(P):
            beta = np.dot(It - mu, U[:, i])
            Wt[i] = beta
        It_hat = mu.copy()
        for i in range(P):
            It_hat += np.dot(Wt[i], U[:, i])
        d = np.linalg.norm(W - Wt, axis=1)
        ranks = d.argsort() + 1
        ax7[j] = fig7.add_subplot(3, 4, j+1)
        ax7[j].set_xticks([])
        ax7[j].set_yticks([])
        ax7[j].set_title("{}".format(d[ranks[0]-1]))
        ax7[j].imshow(It_hat.reshape(90, 60), cmap='gray')
    fig7.suptitle("Distances in Reconstruction from Sampled Test from Each Person P={}".format(P))
    fig7.savefig("reconstruction_cost_{}.png".format(P))
    # plt.show()

