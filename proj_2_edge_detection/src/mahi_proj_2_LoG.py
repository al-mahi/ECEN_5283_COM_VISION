#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import signal
from skimage.measure import label
from bwmorph_thin import bwmorph_thin


def laplace_of_gaussian(i, j, k, sigma):
    v = (i - k) ** 2 + (j - k) ** 2
    return (1 / (2. * sigma ** 4.)) * (v / sigma ** 2. - 2) * np.exp(-v / (2 * sigma**2))


def py_bwlabel(img, N=8):
    if N!=4 and N!=8: raise Exception("Only 4 and 8 are acceptable")

    labeled = np.zeros(img.shape, dtype='bool')
    pixels = np.ndindex(img.shape)

    num = 1
    for i, j in pixels:
        if not img[i, j] or labeled[i, j]: continue
        q = [(i, j)]
        diff = [-1, 0, 1]
        while q:
            x, y = q.pop(0)
            labeled[x, y] = num
            for dx in diff:
                for dy in diff:
                    if N == 4 and (dx != 0 and dy != 0): continue
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < img.shape[0] and 0 <= ny < img.shape[1]:
                        if img[nx, ny] and not labeled[nx, ny]:
                            q.append((nx, ny))
        num += 1
    return labeled, num


def LoG(path='retina1.jpg', k=2, sigma=0.5, threshold = 1000, N = 120, neighbor=8, fig_num=1):
    params = "of img={} with \nk={} $\sigma$={} $\\tau_z$={} $\\tau_l$={} Neighbor={}".format(path, k, sigma, threshold, N, neighbor)
    img = cv2.imread(path)
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    green_img = RGB_img[:, :, 1]
    fig1 = plt.figure(1)
    ax = fig1.add_subplot(111)
    ax.imshow(RGB_img)
    ax.set_title("LoG:Original \n{}".format(params))
    plt.savefig("log_fig/{}_fig1_{}.png".format(path, fig_num))

    kspace = (2*k+1, 2*k+1)
    K = np.zeros(kspace)

    pixels = np.ndindex(kspace)
    for i, j in pixels:
        K[i, j] = laplace_of_gaussian(i, j, k, sigma)

    np.set_printoptions(precision=3, suppress=True)
    print("K=\n{}".format(K))

    if k==2:
        K_star = np.array([
            [0, 0,  1, 0, 0],
            [0, 1,  2, 1, 0],
            [1, 2,-16, 2, 1],
            [0, 1,  2, 1, 0],
            [0, 0,  1, 0, 0]
        ])
    elif k==3:
        K_star = np.array([
            [0, 0, 0,  1, 0, 0, 0],
            [0, 0, 1,  2, 1, 0, 0],
            [0, 1, 1, -3, 1, 1, 0],
            [1, 2,-3,-12,-3, 2, 1],
            [0, 1, 1, -3, 1, 1, 0],
            [0, 0, 1,  2, 1, 0, 0],
            [0, 0, 0,  1, 0, 0, 0]
        ])

    conv_img = signal.convolve2d(green_img, K_star, mode='same')

    fig2 = plt.figure(2)
    ax = fig2.add_subplot(111)
    ax.imshow(conv_img, cmap='gray')
    ax.set_title("LoG:Edges after convolution {}".format(params))
    plt.savefig("log_fig/{}_fig2_{}.png".format(path, fig_num))

    print "connected component......."
    r, c = conv_img.shape
    pixels = np.ndindex(conv_img.shape)
    zero_crossed = np.zeros(conv_img.shape, dtype='bool')

    for i, j in pixels:
        if i==0 or j==0 or i==r-1 or j==c-1: continue
        if (
                # np.isclose(conv_img[i,j], 0., atol=0.0001) and  # gives bad result when this condition is applied
                   (conv_img[i-1, j-1] * conv_img[i+1, j+1] < 0 and np.abs(conv_img[i-1, j-1] * conv_img[i+1, j+1]) >= threshold) \
                or (conv_img[i-1, j-0] * conv_img[i+1, j+0] < 0 and np.abs(conv_img[i-1, j-0] * conv_img[i+1, j+0]) >= threshold) \
                or (conv_img[i-1, j+1] * conv_img[i+1, j-1] < 0 and np.abs(conv_img[i-1, j+1] * conv_img[i+1, j-1]) >= threshold) \
                or (conv_img[i-0, j+1] * conv_img[i+0, j-1] < 0 and np.abs(conv_img[i-0, j+1] * conv_img[i+0, j-1]) >= threshold)
        ):
            zero_crossed[i, j] = True

    fig3 = plt.figure(3)
    ax = fig3.add_subplot(111)
    ax.imshow(zero_crossed, cmap='gray')
    ax.set_title("LoG:Edges of zero crossing {}".format(params))
    plt.savefig("log_fig/{}_fig3_{}.png".format(path, fig_num))

    print "label image..."
    # my implementation of bwlabel. It takes lot of time as it is not optimized. Some library function probably
    # running C code underline is much faster. Faster implementation can be done using C bit array and bitwise operation
    # Anyway that is not the sole objective of this task so please use library function for faster execution

    # labeled, labels = py_bwlabel(np.array(zc), N=4)
    labeled_img = label(zero_crossed, neighbors=neighbor)
    mx_label = np.max(labeled_img)

    clearer_img = zero_crossed

    for i in range(1, mx_label+1):
        ind = np.where(labeled_img == i)
        M = len(ind[0])
        if M < N:
            clearer_img[ind] = False

    fig4 = plt.figure(4)
    ax = fig4.add_subplot(111)
    ax.imshow(clearer_img, cmap='gray')
    ax.set_title("LoG:After Length Filtering {}".format(params))
    plt.savefig("log_fig/{}_fig4_{}.png".format(path, fig_num))

    thin = bwmorph_thin(clearer_img)

    fig5 = plt.figure(5)
    ax = fig5.add_subplot(111)
    ax.imshow(thin, cmap='gray')
    ax.set_title("LoG:After Thinning {}".format(params))
    plt.savefig("log_fig/{}_fig5_{}.png".format(path, fig_num))

    r, c, d = RGB_img.shape
    super_imposed = RGB_img
    for i in range(2, r):
        for j in range(2, c):
            if clearer_img[i, j] ==1:
                super_imposed[i, j, :] = 255

    fig6 = plt.figure(6)
    ax = fig6.add_subplot(111)
    ax.imshow(super_imposed)
    ax.set_title("LoG: After super imposeing {}".format(params))
    plt.savefig("log_fig/{}_fig6_{}.png".format(path, fig_num))

    plt.show()


if __name__ == '__main__':
    fig_num = 0
    LoG('retina1.jpg', sigma=.5, N=300, threshold=700, fig_num=fig_num, neighbor=8)
    fig_num += 1

    LoG('retina2.jpg', sigma=.5, N=600, threshold=300, fig_num=fig_num, neighbor=8)
    fig_num += 1

    LoG('retina2.jpg', sigma=.5, N=1000, threshold=200, fig_num=fig_num, neighbor=8)
    fig_num += 1

    LoG('retina3.jpg', k=3, sigma=1, N=800, threshold=150, fig_num=fig_num, neighbor=4)
    fig_num += 1

    LoG('retina4.jpg', k=3, sigma=1, N=800, threshold=150, fig_num=fig_num, neighbor=4)
    fig_num += 1


