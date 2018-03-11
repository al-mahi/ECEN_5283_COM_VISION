#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import signal
from skimage.measure import label
from bwmorph_thin import bwmorph_thin


def py_bwlabel(img, N=8):
    if N != 4 and N != 8: raise Exception("Only 4 and 8 are acceptable")

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


def dgdx(i, j, k, sigma):
    v = (i - k) ** 2 + (j - k) ** 2
    return ((-i + k) / (2. * sigma ** 4.)) * np.exp(-v / (2 * sigma ** 2))


def dgdy(i, j, k, sigma):
    v = (i - k) ** 2 + (j - k) ** 2
    return ((-j + k) / (2. * sigma ** 4.)) * np.exp(-v / (2 * sigma ** 2))


def canny(path='retina1.jpg', k=2, sigma=0.5, neighbor=4, threshold=30, N=30, fig_num=0):
    params = "of {} with \nk={} $\sigma$={} $\\tau_l$={} N={} Neighbor={}".format(path, k, sigma, threshold, N, neighbor)

    img = cv2.imread(path)
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    green_img = RGB_img[:, :, 1]

    kspace = (2 * k + 1, 2 * k + 1)
    dfdx_approx = np.zeros(kspace)
    dfdy_approx = np.zeros(kspace)

    pixels = np.ndindex(kspace)
    for i, j in pixels:
        dfdx_approx[i, j] = dgdx(i, j, k, sigma)
        dfdy_approx[i, j] = dgdy(i, j, k, sigma)

    np.set_printoptions(precision=3, suppress=True)

    dfdx = np.array([
        [0, 0, 0, 0,0],
        [0, 1, 2, 1,0],
        [0, 0, 0, 0,0],
        [0,-1,-2,-1,0],
        [0, 0, 0, 0,0]])

    dfdy = dfdx.T

    conv_x = signal.convolve2d(green_img, dfdx)
    conv_y = signal.convolve2d(green_img, dfdy)

    grad_magnitude = np.sqrt(conv_x ** 2 + conv_y ** 2)
    theta = np.arctan2(conv_x, conv_y)
    tan_theta = np.tan(theta)

    # threshold = 30  # retina1=30, retina2=30 retina3=25 retina4= 20

    ind = np.where(grad_magnitude < threshold)
    grad_magnitude[ind] = 0

    fig1 = plt.figure(1)
    ax = fig1.add_subplot(111)
    ax.imshow(RGB_img)
    ax.set_title("Canny: Gradient {}".format(params))
    plt.savefig("canny_fig/{}_fig1_{}.png".format(path, fig_num))

    r, c, d = img.shape
    nmx_supressed = np.zeros(green_img.shape)
    edge_orientation = np.zeros(img.shape, dtype=np.float32)  # color coded orientation. data type is important
    pixels = np.ndindex(green_img.shape)
    for i, j in pixels:
        fx = conv_x[i, j]
        fy = conv_y[i, j]
        m = np.abs(conv_x[i, j]/conv_y[i, j])
        mm = np.abs(conv_y[i, j]/conv_x[i, j])
        g = grad_magnitude[i, j]
        if i == 0 or j == 0 or i == r - 1 or j == c - 1 or grad_magnitude[i, j] == 0: continue
        if fx == 0 or fy == 0:
            if fx == 0 and fy != 0:
                if g >= grad_magnitude[i, j + 1] and g >= grad_magnitude[i, j - 1]:
                    nmx_supressed[i, j] = True
                    edge_orientation[i, j, 0] = 1
            elif fx != 0 and fy == 0:
                if g >= grad_magnitude[i + 1, j] and g >= grad_magnitude[i - 1, j]:
                    nmx_supressed[i, j] = True
                    edge_orientation[i, j, 1] = 1
        elif fx * fy > 0 and abs(fx) >= abs(fy):
            df1 = grad_magnitude[i - 1, j - 1] * m + grad_magnitude[i - 1, j] * (1 - m)
            df2 = grad_magnitude[i + 1, j - 1] * (1 - m) + grad_magnitude[i + 1, j + 1] * m
            if fx < 0 and fy < 0 and g >= df1:
                nmx_supressed[i, j] = True
                edge_orientation[i, j, 2] = 1
            elif fx > 0 and fy > 0 and g >= df2:
                nmx_supressed[i, j] = True
                edge_orientation[i, j, 2] = 1
        elif fx * fy > 0 and abs(fx) < abs(fy):
            df1 = grad_magnitude[i - 1, j - 1] * mm + grad_magnitude[i, j - 1] * (1 - mm)
            df2 = grad_magnitude[i, j + 1] * (1 - mm) + grad_magnitude[i + 1, j + 1] * mm
            if fx < 0 and fy < 0 and g >= df1:
                nmx_supressed[i, j] = True
                edge_orientation[i, j, 2] = 1
            elif fx > 0 and fy > 0 and g >= df2:
                nmx_supressed[i, j] = True
                edge_orientation[i, j, 2] = 1
        elif fx * fy < 0 and abs(fx) > abs(fy):
            df1 = grad_magnitude[i - 1, j + 1] * m + grad_magnitude[i - 1, j] * (1 - m)
            df2 = grad_magnitude[i + 1, j] * (1 - m) + grad_magnitude[i + 1, j - 1] * m
            if fx < 0 < fy and g >= df1:
                nmx_supressed[i, j] = True
                edge_orientation[i, j, 1] = 1
                edge_orientation[i, j, 2] = 1
            elif fx > 0 > fy and g >= df2:
                nmx_supressed[i, j] = True
                edge_orientation[i, j, 1] = 1
                edge_orientation[i, j, 2] = 1
        elif fx * fy < 0 and abs(fx) < abs(fy):
            df1 = grad_magnitude[i, j + 1] * (1 - mm) + grad_magnitude[i - 1, j + 1] * mm
            df2 = grad_magnitude[i, j - 1] * (1 - mm) + grad_magnitude[i + 1, j - 1] * mm
            if fx < 0 < fy and g >= df1:
                nmx_supressed[i, j] = True
                edge_orientation[i, j, 1] = 1
                edge_orientation[i, j, 2] = 1
            elif fx > 0 > fy and g >= df2:
                nmx_supressed[i, j] = True
                edge_orientation[i, j, 1] = 1
                edge_orientation[i, j, 2] = 1

    fig2 = plt.figure(2)
    ax = fig2.add_subplot(111)
    ax.imshow(nmx_supressed, cmap='gray')
    ax.set_title("Canny: Non Max. Supression {}".format(params))
    plt.savefig("canny_fig/{}_fig2_{}.png".format(path, fig_num))

    print "label image..."
    labeled_img, mx_label = label(nmx_supressed,neighbors=4, return_num=True)

    clearer_img = nmx_supressed
    # N = 30
    for i in range(1, mx_label+1):
        ind = np.where(labeled_img == i)
        M = len(ind[0])
        if M < N:
            clearer_img[ind] = 0
            edge_orientation[ind] = 0

    fig3 = plt.figure(3)
    ax = fig3.add_subplot(111)
    ax.imshow(clearer_img, cmap='gray')
    ax.set_title("Canny: After Non Maximum Supression and Length Filter with {}".format(params))
    plt.savefig("canny_fig/{}_fig3_{}.png".format(path, fig_num))

    # img_grad = cv2.cvtColor(orientaion, cv2.COLOR_BGR2RGB)
    fig4 = plt.figure(4)
    ax = fig4.add_subplot(111)
    ax.imshow(edge_orientation)
    ax.set_title("Canny: Edges orientation {}".format(params))
    plt.savefig("canny_fig/{}_fig4_{}.png".format(path, fig_num))


    # this is not a library function unfortunately python cv2 and scikit image library does not have any function
    # similar to bwmorph. I have used a bwmorph by an unofficial implementation of bwmorph
    # kernel = np.ones((3, 3), np.uint8)
    # erosion = cv2.erode(clearer_img, kernel, iterations=1)
    thin = bwmorph_thin(clearer_img)

    fig5 = plt.figure(5)
    ax = fig5.add_subplot(111)
    ax.imshow(thin, cmap='gray')
    ax.set_title("Canny: Edges orientation {}".format(params))
    plt.savefig("canny_fig/{}_fig5_{}.png".format(path, fig_num))

    r, c, d = RGB_img.shape
    super_imposed = RGB_img
    for i in range(2, r):
        for j in range(2, c):
            if clearer_img[i, j] ==1:
                super_imposed[i, j, :] = 255

    fig6 = plt.figure(6)
    ax = fig6.add_subplot(111)
    ax.imshow(super_imposed)
    ax.set_title("Canny: Edges Superimposed {}".format(params))
    plt.savefig("canny_fig/{}_fig5_{}.png".format(path, fig_num))

    plt.show()


if __name__ == '__main__':
    fig_num = 0
    canny('retina1.jpg', sigma=.5, N=30, threshold=30, fig_num=fig_num)
    fig_num += 1
    canny('retina2.jpg', sigma=.5, N=30, threshold=30, fig_num=fig_num)
    fig_num += 1
    canny('retina3.jpg', sigma=.5, N=20, threshold=20, fig_num=fig_num)
    fig_num += 1
    canny('retina4.jpg', sigma=.5, N=20, threshold=20, fig_num=fig_num)
    fig_num += 1

    fig_num = 4
    canny('retina1.jpg', sigma=2, N=50, threshold=50, fig_num=fig_num)
    fig_num += 1
    canny('retina2.jpg', sigma=2, N=50, threshold=50, fig_num=fig_num)
    fig_num += 1
    canny('retina3.jpg', sigma=2, N=10, threshold=20, fig_num=fig_num)
    fig_num += 1
    canny('retina4.jpg', sigma=2, N=10, threshold=20, fig_num=fig_num)
    fig_num += 1
