import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scale = 50
space = (scale, scale, scale)

x = np.linspace(0., scale, scale+1)
y = np.linspace(0., scale, scale+1)
z = np.linspace(0., scale, scale+1)

indices = np.ndindex(space)
c = scale/2.

F = np.zeros(space)


def build_piece(ind):
    x, y, z = ind[:]
    a = scale * .3
    b = scale * .1
    z1 = scale * .1
    z2 = scale * .9
    if z <= z1:
        r = a - (a-b)*z/z1
        if np.abs(x-c) < r and np.abs(y-c) < r:
            F[x, y, z] = 1.
    elif z2 < z < scale:
        r = b + (z-z2)
        if np.abs(x-c) < r and np.abs(y-c) < r:
            F[x, y, z] = 1.
    else:
        r = b
        if np.abs(x-c) < r and np.abs(y-c) < r:
            F[x, y, z] = 1.

np.array(map(build_piece, indices), dtype=np.float32).reshape(space)
ind = np.where((F > .0))

print(ind)
norm = mpl.colors.Normalize(vmin=np.min(F[ind]), vmax=np.max(F[ind]), clip=True)
x, y, z = ind[0], ind[1], ind[2]
p = ax.scatter(x, y, z, c=F[ind], norm=norm, alpha=0.4)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(0, scale)
ax.set_ylim(0, scale)
ax.set_zlim(0, scale)
plt.colorbar(p)
plt.show()


