#!/usr/bin/python

# from __future__ import print_function
import numpy as np


if __name__ == '__main__':
    # https://www.youtube.com/watch?v=cjlhpaDXihE
    T = 5
    N = 3
    v = "ACBAC"
    Ob = {"A":0, "B":1, "C":2}

    a = np.array([[.2, .3, .5],
                  [.2, .2, .6],
                  [0., .2, .8]])

    b = np.array([[.7, .2, .1],
                  [.3, .4, .3],
                  [0., .1, .9]])

    alpha = np.zeros(shape=(N, T))
    alpha[0, 0] = .23
    alpha[1, 0] = .1
    alpha[2, 0] = 0.

    for t in range(0, T-1):
        for i in range(N):
            for j in range(N):
                alpha[i, t+1] += alpha[j, t] * a[j, i]
            alpha[i, t + 1] *= b[i, Ob[v[t + 1]]]

    np.set_printoptions(precision=5)
    print("forward alpha")
    print(alpha)

    # viterbi
    delta = np.zeros(shape=(N, T))

    pi = 1./3. * np.ones(N)

    delta[0, 0] = pi[0] * b[0, Ob[v[0]]]
    delta[1, 0] = pi[1] * b[1, Ob[v[0]]]
    delta[2, 0] = pi[2] * b[2, Ob[v[0]]]

    for t in range(1, T):
        for i in range(N):
            max_v = -np.inf
            for j in range(N):
                max_v = max(max_v, delta[j, t-1] * a[j, i] * b[i, Ob[v[t]]])
            delta[i, t] = max_v

    print("viterbi delta:")
    print(delta)

    print(delta.argmax(axis=0))
