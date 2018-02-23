#!/usr/bin/python

import numpy as np
import matplotlib as mpl
# mpl.use("Agg")
import matplotlib.pyplot as plt
import cv2
import matplotlib.animation as animation


def camera_calibrate(path_observe='observe.dat', path_model='model.dat', path_img='test_image.bmp'):
    observe = np.loadtxt(path_observe)[:]
    model = np.loadtxt(path_model)[:]

    # public data set for 3d point cloud. adjusted to fit in positive coordinate and scale
    # http: // rgbd - dataset.cs.washington.edu / dataset / rgbd - dataset_pcd_ascii /
    # http: // pointclouds.org / media /
    tablelamp = np.loadtxt('tablelamp.dat')[:]
    tablelamp = .3 * (tablelamp - np.min(tablelamp))
    # mug = np.loadtxt('mug.pcd')[:, :3]
    # mug = 10 * (mug - np.min(mug))

    Mn, Mt = model.shape  # dim
    On, Ot = observe.shape
    img = cv2.imread(path_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    r, c = img.shape

    def plot_image(pixels, window=4, threshold=0):
        """
        :param pixels: pixel coordinate
        :param window: size of the dot
        :param threshold: color of the dot 0 is black 255 white
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.clear()
        new_img = np.array(img)
        for pix in pixels:
            y, x = map(int, pix)
            new_img[x, y] = threshold
            d = range(-window, window + 1)
            for dx in d:
                for dy in d:
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < r and 0 <= ny < c:
                        new_img[nx, ny] = threshold
        ax.imshow(new_img, cmap='gray')
        plt.show()

    def build_Q_2_rows_at_a_time(i):
        """
        :param i: ith iteration of building Q
        :return: 2 rows of Q matrix for of ith iteration
        """
        P_i = np.vstack((model[i].reshape(3, 1), np.ones(1)))
        O = np.zeros((4, 1))
        u_i = observe[i, 0]
        v_i = observe[i, 1]
        return np.vstack((
            np.hstack((np.hstack((P_i.T, O.T)), np.dot(-u_i.T, P_i.T))),
            np.hstack((np.hstack((O.T, P_i.T)), np.dot(-v_i.T, P_i.T)))
        ))

    Q = np.vstack(map(build_Q_2_rows_at_a_time, range(Mn)))

    QtQ = np.dot(Q.T, Q)
    s, V = np.linalg.eig(QtQ)
    uu, ss, vv = np.linalg.svd(QtQ)

    print "uu", uu
    print "ss", ss
    print "vv", vv

    print "s", s
    print "v", V

    M = V[:, 11].reshape(4, 3, order='F').T  # projection matrix
    m1 = M[0, :]
    m2 = M[1, :]
    m3 = M[2, :]

    # estimation of intrinsic and extrinsic parameters
    # M = (A b) where aiT is ith row of A
    a1 = M[0, :3].T
    a2 = M[1, :3].T
    a3 = M[2, :3].T
    b = M[:, 3]

    rho = 1. / np.linalg.norm(a3)
    u0 = rho ** 2. * np.dot(a1, a3)
    v0 = rho ** 2. * np.dot(a2, a3)

    numerator = -np.dot(np.cross(a1, a3), np.cross(a2, a3))
    denominator = np.linalg.norm(np.cross(a1, a3)) * np.linalg.norm(np.cross(a2, a3))
    theta = np.arccos(numerator / denominator)
    alpha = rho ** 2 * np.linalg.norm(np.linalg.norm(np.cross(a1, a3))) * np.sin(theta)
    beta = rho ** 2 * np.linalg.norm(np.linalg.norm(np.cross(a2, a3))) * np.sin(theta)

    r1 = np.cross(a2, a3) / np.linalg.norm(np.cross(a2, a3))
    r3 = rho * a3
    r2 = np.cross(r3, r1)

    cot = lambda x: np.cos(x) / np.sin(x)

    K = np.array([
        [alpha, -alpha * cot(theta), u0],
        [0, beta / np.sin(theta), v0],
        [0, 0, 1]
    ])

    t = rho * np.dot(np.linalg.inv(K), b)

    R = np.vstack((np.vstack((r1, r2)), r3))

    print("Parameters:M=\n{}\ntheta={}, alpha={}, beta={}, u0={}, v0={}\nK=\n{}\nt={}\nR=\n{}\n".format(M, np.rad2deg(theta), alpha, beta, u0, v0, K,
                                                                                              t, R))
    def project_2d(W):
        """
        :param W: world coordinate
        :return: Pixel coordinate
        """
        P = np.array([W[0], W[1], W[2], 1.])  # 4d homogeneous coord of 3D World
        u = np.dot(m1.T, P) / np.dot(m3.T, P)
        v = np.dot(m2.T, P) / np.dot(m3.T, P)
        p = np.array([u, v])  # pixel 2D coordinate
        return np.round(p)

    observe_2d = np.array(map(project_2d, model), dtype='int')
    print("error=", np.linalg.norm(observe_2d - observe) / observe.shape[0])

    observe_2d = np.array(map(project_2d, model), dtype='int')
    plot_image(pixels=observe_2d, window=2)

    observe_2d = np.array(map(project_2d, tablelamp), dtype='int')
    plot_image(pixels=observe_2d, window=2)
    # observe_2d = np.array(map(project_2d, mug), dtype='int')
    # plot_image(pixels=observe_2d, window=2)

    indices = filter(lambda ind: (ind[2] == 0) or (ind[1] == 10) or ind[0] == 10, np.ndindex((11, 11, 11)))
    observe_all_2d = np.array(map(project_2d, indices), dtype='int')
    plot_image(pixels=observe_all_2d, window=2)
    # each row two points of a single line in the box of seven vertices
    lines = [
        [(0, 0, 0), (1, 0, 0)],
        [(0, 0, 0), (0, 1, 0)],
        [(0, 0, 0), (0, 0, 1)],
        [(0, 1, 1), (0, 0, 1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 1, 1), (0, 1, 0)],
        [(1, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 0, 0)],
        [(1, 0, 1), (0, 0, 1)],
    ]

    def sample_100_points_in_a_line(two_3dpoints):
        """
        :param two_3dpoints: two points describing a single line of the box
        :return:
        """
        if two_3dpoints is not None:
            num = 100
            a = two_3dpoints[0]
            b = two_3dpoints[1]
            xs = np.linspace(start=a[0], stop=b[0], endpoint=True, num=num)
            ys = np.linspace(start=a[1], stop=b[1], endpoint=True, num=num)
            zs = np.linspace(start=a[2], stop=b[2], endpoint=True, num=num)
            return zip(xs, ys, zs)

    line_samples = filter(lambda sample: sample is not None, map(sample_100_points_in_a_line, lines))
    sample_list = []
    for tuple_arr in line_samples:
        for tup in tuple_arr:
            sample_list.append(list(tup))
    box_samples = np.array(sample_list)
    box_2d = np.array(map(project_2d, box_samples), dtype='int')
    plot_image(pixels=box_2d, window=1)

    def update(i, fig1, ax1, goal, steps, window=1, threshold=4, start=[0, 0, 0]):
        """
        Updates the video frame
        :param i: frame#
        :param fig1:
        :param ax1:
        :param goal: goal for the box
        :param steps: how many steps to the goal
        :param window: size of the line in pixel
        :param threshold: pixel value for drawing dot
        :param start: starting location of the box
        :return: A frame of the plot
        """
        xs = np.linspace(start[0], goal[0], endpoint=False, num=steps)
        ys = np.linspace(start[1], goal[1], endpoint=False, num=steps)
        zs = np.linspace(start[2], goal[2], endpoint=False, num=steps)
        dx, dy, dz = xs[i], ys[i], zs[i]
        new_box = box_samples + np.array([dx, dy, dz])
        box_2d = np.array(map(project_2d, new_box), dtype='int')
        ax1.clear()
        new_img = np.array(img)
        for pix in box_2d:
            y, x = map(int, pix)
            new_img[x, y] = threshold
            d = range(-window, window + 1)
            for dx in d:
                for dy in d:
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < r and 0 <= ny < c:
                        new_img[nx, ny] = threshold
        p = ax1.imshow(new_img, cmap='gray')
        return p

    def animate_box(start=[0, 0, 0], goal=[10, 0, 0], steps=10, save_file=''):
        """
        :param start: starting location of the box in World coordinate
        :param goal: goal for the box
        :param steps: How many steps during the movement
        :param save_file: relative path to the file to save video for animation
        """
        fig1 = plt.figure(1)
        ax1 = fig1.subplots()
        anims = [None]
        anims[0] = animation.FuncAnimation(fig=fig1, func=update, fargs=(fig1, ax1, goal, steps, 1, 0, start), frames=range(steps),
                                        interval=200, blit=False)
        if save_file:
            print(save_file)
            # Set up formatting for the movie files
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
            anims[0].save(save_file, writer)
        plt.show()

    np.savetxt("M.csv", M)
    np.savetxt("R.csv", R)
    np.savetxt("K.csv", K)
    np.savetxt("t.csv", t.T)
    animate_box(save_file='x_move.mp4')
    animate_box(goal=[10, 10, 0], save_file='diag_move.mp4')
    animate_box(start=[0, 9, 0], goal=[10, 9, 0], save_file="different_origin.mp4")

    # def move_box(goal=[10, 0, 0], steps=10):
    #     xs = np.linspace(0, goal[0], endpoint=False, num=steps)
    #     ys = np.linspace(0, goal[1], endpoint=False, num=steps)
    #     zs = np.linspace(0, goal[2], endpoint=False, num=steps)
    #     for dx, dy, dz in zip(xs, ys, zs):
    #         new_box = box_samples + np.array([dx, dy, dz])
    #         box_2d = np.array(map(project_2d, new_box), dtype='int')
    #         plot_image(pixels=box_2d, window=1)
    #
    # move_box()




if __name__ == '__main__':
    # camera_calibrate()
    camera_calibrate(path_observe = 'observe.dat', path_model = 'model.dat', path_img = 'test_image.bmp')
