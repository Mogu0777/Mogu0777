from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import show_image
import get_obj
from random import random

fig = plt.figure()  #定义新的三维坐标轴
ax = plt.axes(projection='3d')

obj_path = 'mould_xifen/blade_xifen.obj'
vertice = np.array(get_obj.read_obj_vertices('mould_xifen/blade_xifen.obj'))
voxel = get_obj.get_voxelgrid(obj_path, 0.02, scale=0.001, show_mould=True)

#作图

# ax.scatter(np.array(vertice[:, 0]), np.array(vertice[:, 1]), np.array(vertice[:, 2]), c=z)
#ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap='rainbow)   #等高线图，要设置offset，为Z的最小值
plt.show()