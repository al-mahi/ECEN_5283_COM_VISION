#!/usr/bin/python

from __future__ import print_function

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from mcmc_functions import likelihood, drawcircle
import os
from sklearn.cluster import KMeans

out = cv2.VideoWriter('../figs/outpy.avi', cv2.VideoWriter_fourcc(*'MJPG'), 1, (128, 128))


def diffusion(t, path_im, I, T, theta, K, lik):
    iter = 100
    X, Y = I.shape
    for ki in xrange(K):
        step_size = 45
        theta_p = theta.copy()
        for _ in xrange(iter):
            # to manage object movement in corner cases. simply adding random move would have 1/4 chance move moving
            # when it is in the corner. thanks to a classmate pointed this out.
            theta_p[ki, 0] += np.random.randint(low=max(-theta_p[ki, 0], -step_size), high=min(X - theta_p[ki, 0]-1, step_size))
            theta_p[ki, 1] += np.random.randint(low=max(-theta_p[ki, 1], -step_size), high=min(Y - theta_p[ki, 1]-1, step_size))

            # uncomment if u want to see gibbs movements
            # Io = drawcircle(I, theta_p, K)
            # cv2.imshow('frame2', Io)
            # cv2.waitKey(1)

            lik_p = likelihood(I, T, theta_p, K)
            v = min(1., lik_p / lik)
            u = np.random.uniform(low=0., high=1.)
            if v > u:  # accept
                # print("accept lik_p={:.4e}".format(lik_p))
                lik = lik_p
                theta[ki] = theta_p[ki]
                step_size = max(step_size-1, 30)

        Io = drawcircle(I, theta, K)
        # cv2.imshow('frame', Io)
        # cv2.waitKey(1)
        plt.imshow(Io, cmap='gray')
        plt.savefig("../figs/{}/{:03d}_{:02d}.png".format(path_im[:-4], t, ki))
        # out.write(np.uint8(Io))
    return theta


def jump(t, path_im, I, T, theta, k, dk, lik):
    k_p = k + dk
    theta_init = theta[:k_p].copy()

    if dk == 1:
        init_new_sample = np.random.randint(low=0, high=min(I.shape), size=2)
        theta_init = np.vstack((theta_init, init_new_sample))

    theta_p = diffusion(t, path_im, I, T, theta_init, k_p, lik)
    return theta_p


def jump_diffusion_mcmc(path_im, path_target, k):
    in_img = cv2.imread("../project5/{}".format(path_im))
    in_tar = cv2.imread("../project5/{}".format(path_target))
    I = cv2.cvtColor(in_img, cv2.COLOR_RGB2GRAY)/255.
    T = cv2.cvtColor(in_tar, cv2.COLOR_RGB2GRAY)/255.

    X, Y = I.shape
    k_max = 40
    N = 20
    # lamda and k are tuning param. lamda is the prior knowledge. thumb of rule can be chosing a lambda by eyeballing
    # at the image gusseing number of objects that migh be there. when chosing lamda choose it to be at least greated
    # than true number of objects then chosing k 5 to 10 less than lamda.
    num = int(path_im[len("discs"):-len(".bmp")])
    lamda = num
    k = lamda + 2
    theta = np.random.randint(low=5, high=X-5, size=(k, 2))
    Io = drawcircle(I, theta, k)
    plt.imshow(Io, cmap='gray')
    plt.savefig("../figs/{}_initial.png".format(path_im[:-4]))

    theta_final = np.zeros(shape=2)
    K_final = np.zeros(N)

    for t in xrange(N):
        print("iter:{} k:{}".format(t, k))
        lik = likelihood(I, T, theta, k)
        post = lik * poisson.pmf(k=k, mu=lamda)

        a = np.random.uniform(low=0., high=1.)
        if a < 1/3. and k > 1: dk = -1
        elif a < 1 / 6. and k < k_max: dk = 1
        else: dk = 0

        theta_p = jump(t, path_im, I, T, theta, k, dk, lik)
        lik_p = likelihood(I, T, theta_p, k + dk)
        post_p = lik_p * poisson.pmf(k=k + dk, mu=lamda)

        v = min(1., post_p / post)
        u = np.random.uniform(low=0., high=1.)
        if v > u:
            theta = theta_p
            k += dk

        theta_final = np.vstack((theta_final, theta))
        K_final[t] = k

    M = 5
    theta_final = theta_final[M:]
    K_final = K_final[M:]
    K_star = int(np.average(K_final))

    theta_star = KMeans(n_clusters=K_star, random_state=0).fit(theta_final).cluster_centers_
    Io = drawcircle(I, theta_star, K_star)
    plt.title("Final")
    plt.imshow(Io, cmap='gray')
    plt.savefig("../figs/{}_final.png".format(path_im[:-4]))


if __name__ == '__main__':
    for im in [1, 2, 3]:
        os.system("rm ../figs/discs{}/*.png".format(im))
        jump_diffusion_mcmc("discs{}.bmp".format(im), "target.bmp", k=5)

        cv2.waitKey(10)
        out.release()
        # Closes all the frames
        cv2.destroyAllWindows()
    plt.show()



