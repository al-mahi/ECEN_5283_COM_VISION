#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import signal
from skimage.measure import label
from bwmorph_thin import bwmorph_thin
# from scipy.misc import imrotate  # does not clip or preserve range
from skimage.transform import rotate
from scipy.stats import describe


def gaussian1d(x, y, k, m0, sigma):
    return (-1 / np.sqrt(2. * sigma ** 2.)) * np.exp(-(y - k) ** 2 / (2 * sigma ** 2)) - m0


def matched(path='retina1.jpg', k=8, sigma=1.5, N=120, threshold=30, neighbor=8, fig_num=0):
    params = "of img={} with \nk={} $\sigma$={} $\\tau_z$={} $\\tau_l$={} Neighbor={}".format(path, k, sigma, threshold,
                                                                                              N, neighbor)
    img = cv2.imread(path)
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    green_img = RGB_img[:, :, 1]
    kspace = (2 * k + 1, 2 * k + 1)

    f = np.zeros(kspace)
    m0 = -0.0999
    pixels = np.ndindex(kspace)
    for x, y in pixels:
        f[x, y] = gaussian1d(x, y, k, m0, sigma)

    print("m0={}".format(m0))

    angles = range(0, 180, 15)
    N = len(angles)
    pad = np.pad(f, ((2, 2), (2, 2)), 'constant', constant_values=((0, 0), (0, 0)))
    fig2 = plt.figure(1)
    ax = fig2.add_subplot(111)
    ax.imshow(pad, cmap='gray')
    ax.set_title("Matched Filter{}".format(params))
    plt.savefig("match_fig/{}_fig1_{}.png".format(path, fig_num))
    # pad += np.abs(np.min(pad))
    # pad /= np.max(pad)
    # pad *= 255.

    low = np.min(pad)
    high = np.max(pad)
    B = {0: pad}
    I = {}
    fig3 = plt.figure(3)
    for i in range(N):
        B[angles[i]] = rotate(pad, angles[i], clip=True, preserve_range=True)
        ax3 = fig3.add_subplot(3, 4, i+1)
        ax3.imshow(B[angles[i]], cmap='gray')
        ax3.set_title("$\\theta$={}".format(angles[i]))
        fig3.suptitle("Mathced FIlter:Matched group")
    plt.savefig("match_fig/{}_fig3_{}.png".format(path, fig_num))

    fig4 = plt.figure(4)
    for i in range(N):
        I[angles[i]] = signal.convolve2d(green_img, B[angles[i]], mode="same")
        ax4 = fig4.add_subplot(3, 4, i+1)
        ax4.imshow(I[angles[i]], cmap='gray')
        ax4.set_title("$\\theta$={}".format(angles[i]))
        fig4.suptitle("Mathced FIlter:Convolution With Matched Group")
    plt.savefig("match_fig/{}_fig4_{}.png".format(path, fig_num))

    r, c = I[0].shape
    I_max = np.zeros((r, c))
    Bin = np.zeros((r, c))

    # threshold = -50.
    for i in range(r):
        for j in range(c):
            I_max[i, j] = max([I[angle][i, j] for angle in angles])
            if I_max[i, j] > threshold:
                Bin[i, j] = 1
    print np.mean(I_max),  np.average(I_max), np.min(I_max), np.max(I_max), np.median(I_max)

    fig5 = plt.figure(5)
    ax = fig5.add_subplot(111)
    ax.imshow(I_max, cmap='gray')
    ax.set_title("Mathced FIlter:Fusion{}".format(params))
    plt.savefig("match_fig/{}_fig5_{}.png".format(path, fig_num))

    fig6 = plt.figure(6)
    ax = fig6.add_subplot(111)
    ax.imshow(Bin, cmap='gray')
    ax.set_title("Mathced FIlter:Binary Image{}".format(params))
    plt.savefig("match_fig/{}_fig6_{}.png".format(path, fig_num))

    thin = bwmorph_thin(Bin)
    fig7 = plt.figure(7)
    ax = fig7.add_subplot(111)
    ax.imshow(thin, cmap='gray')
    ax.set_title("Mathced FIlter:After thining{}".format(params))
    plt.savefig("match_fig/{}_fig7_{}.png".format(path, fig_num))

    labeled_img = label(thin,neighbors=neighbor)
    mx_label = np.max(labeled_img)

    clearer_img = thin

    # N = 120
    for i in range(1, mx_label+1):
        ind = np.where(labeled_img == i)
        M = len(ind[0])
        if M < N:
            clearer_img[ind] = False

    fig8 = plt.figure(8)
    ax = fig8.add_subplot(111)
    ax.imshow(clearer_img, cmap='gray')
    ax.set_title("Mathced FIlter:Length Filtering{}".format(params))
    plt.savefig("match_fig/{}_fig8_{}.png".format(path, fig_num))

    r, c, d = RGB_img.shape
    super_imposed = RGB_img
    for i in range(2, r):
        for j in range(2, c):
            if clearer_img[i, j] ==1:
                super_imposed[i, j, :] = 255

    fig9 = plt.figure(9)
    ax = fig9.add_subplot(111)
    ax.imshow(super_imposed)
    ax.set_title("Mathced FIlter:Super Imposed{}".format(params))
    plt.savefig("match_fig/{}_fig9_{}.png".format(path, fig_num))

    plt.show()


if __name__ == '__main__':
    fig_num = 0
    matched('retina1.jpg', N=120, threshold=-50, fig_num=fig_num)
    fig_num += 1
    matched('retina2.jpg', N=120, threshold=-50, fig_num=fig_num)
    fig_num += 1
    matched('retina3.jpg', N=120, threshold=-80, fig_num=fig_num)
    fig_num += 1
    matched('retina4.jpg', N=120, threshold=-80, fig_num=fig_num)
    fig_num += 1

    fig_num = 4
    matched('retina1.jpg', sigma=.5, N=120, threshold=-40, fig_num=fig_num)
    fig_num += 1
    matched('retina2.jpg', sigma=.5, N=120, threshold=-40, fig_num=fig_num)
    fig_num += 1
    matched('retina3.jpg', sigma=.5, N=120, threshold=-50, fig_num=fig_num)
    fig_num += 1
    matched('retina4.jpg', sigma=.5, N=120, threshold=-50, fig_num=fig_num)
    fig_num += 1
