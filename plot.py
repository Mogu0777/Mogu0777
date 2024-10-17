from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import show_image
import get_obj

def plotxyz(voxel_grid, voxel_size):
    voxel_grid_center, _ = show_image.get_voxel_centerandcolor(voxel_grid)
    minxyz = np.min(voxel_grid_center, axis=0)
    maxxyz = np.max(voxel_grid_center, axis=0)
    num_xyz = list(map(int, (maxxyz-minxyz)/voxel_size+2))
    xyzvalues = np.zeros([max(num_xyz),max(num_xyz),max(num_xyz)])
    # print(num_xyz)
    # print(len(voxel_grid_center))
    for i in range(len(voxel_grid_center)):
        exist_voxel = list(map(round, (voxel_grid_center[i] - minxyz) / voxel_size))
        xyzvalues[exist_voxel[0], exist_voxel[1], exist_voxel[2]] = 1

    # print(np.sum(xyzvalues==1))
    return xyzvalues

def voxle_plot(voxel_grid, voxel_size):
    fig = plt.figure(figsize=(7, 4.5))
    ax = fig.add_subplot(projection='3d')
    xyzvalues = plotxyz(voxel_grid, voxel_size)
    pos=ax.voxels(xyzvalues, edgecolor='k',linewidth=1,shade=False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Voxel Map')
    plt.grid(False)
    plt.axis('off')
    plt.show()

def voxle_plot_area(xyzvalues):
    fig = plt.figure(figsize=(7, 4.5))
    ax = fig.add_subplot(projection='3d')
    pos=ax.voxels(xyzvalues, edgecolor='k',linewidth=1,shade=False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Voxel Map')
    plt.grid(False)
    plt.axis('off')
    plt.show()

# 空间为模型包围盒,获得体素坐标
def plotxyz_inaabb(obj_path,voxel_grid, voxel_size, scale):
    max_xyz, min_xyz = get_obj.get_aabb(obj_path, scale)
    min_xyz = min_xyz - voxel_size
    max_xyz = max_xyz + 2 * voxel_size
    num_xyz = list(map(int, (max_xyz-min_xyz)/voxel_size))
    min_xyz = min_xyz - np.array([(max_xyz[0]+min_xyz[0])/2, (max_xyz[1]+min_xyz[1])/2, 0])
    # print("num_xyz="+str(num_xyz))
    plot_area = np.zeros(num_xyz)
    # print(num_xyz)
    # print(len(voxel_grid_center))
    voxel_grid_center, _ = show_image.get_voxel_centerandcolor(voxel_grid)
    for i in range(len(voxel_grid_center)):
        exist_voxel = list(map(int, (voxel_grid_center[i] - min_xyz) / voxel_size))
        plot_area[exist_voxel[0], exist_voxel[1], exist_voxel[2]] = True
    # print(np.sum(plot_area==1))
    # voxle_plot_area(plot_area)
    return plot_area

def voxle_plot_inaabb(obj_path, exist_voxel, hidden_voxel, voxel_size, scale, camera, center):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    max_xyz, min_xyz = get_obj.get_aabb(obj_path, scale)
    min_xyz = min_xyz-voxel_size
    max_xyz = max_xyz+2*voxel_size
    num_xyz = list(map(int, (max_xyz-min_xyz)/voxel_size))
    plt.gca().set_box_aspect((num_xyz[0], num_xyz[1], num_xyz[2]))
    # ax.set_boxaspect()

    exist_voxel = np.bool_(plotxyz_inaabb(obj_path, exist_voxel, voxel_size, scale))
    hidden_voxel = np.bool_(plotxyz_inaabb(obj_path, hidden_voxel, voxel_size, scale))
    #print(np.sum(exist_voxel==True), np.sum(hidden_voxel==True))
    # print(exist_voxel.shape, hidden_voxel.shape)
    # combine the objects into a single boolean array
    voxelarray = hidden_voxel | exist_voxel
    #print(np.sum(voxelarray==True))

    # set the colors of each object
    colors = np.empty(voxelarray.shape, dtype=object)

    colors[hidden_voxel] = 'yellow'
    colors[exist_voxel] = 'blue'

    pos = ax.voxels(voxelarray,facecolors=colors, edgecolor='k',linewidth=1,shade=False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Voxel Map')
    ax.view_init(elev=18, azim=-35)  # 视角
    plt.grid(False)
    plt.axis('off')
    if 0:
        toward = np.array(center)-np.array(camera)
        ax.quiver(camera[0], camera[1],camera[2], toward[0], toward[1], toward[2], length=0.2)  # 在每一个x,y,z坐标绘制矢量方向为u,v,w的箭头
    plt.show()

# 绘制箭头
def plot_vertise(camerapos, targetpos):
    fig = plt.figure()
    ax3d = fig.add_subplot(projection='3d')  # 创建3d坐标系
    for i in range(len(camerapos)):
        toward = np.array(targetpos)-np.array(camerapos[i])
        ax3d.quiver(camerapos[i][0],camerapos[i][1],camerapos[i][2], toward[0], toward[1], toward[2], length=0.2)  # 在每一个x,y,z坐标绘制矢量方向为u,v,w的箭头
    plt.show()

# 绘制视点示意图
def plot_viewpoint(obj_path, scale, voxel_size, camerapos, targetpos, minxyz):
    _, voxel_grid = show_image.get_mould_voxel(obj_path, scale, voxel_size)
    fig = plt.figure(figsize=(7, 4.5))
    ax = fig.add_subplot(projection='3d')
    max_xyz, min_xyz = get_obj.get_aabb(obj_path, scale)
    num_xyz = list(map(int, (max_xyz-min_xyz)/voxel_size))
    print(num_xyz)
    plt.gca().set_box_aspect((num_xyz[0], num_xyz[1], num_xyz[2]-10))
    xyzvalues = plotxyz(voxel_grid, voxel_size)
    pos = ax.voxels(xyzvalues, edgecolor='k', linewidth=1, shade=False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Voxel Map')
    plt.grid(False)
    plt.axis('off')
    for i in range(len(camerapos)):
        # p = camerapos[i] - (num_xyz[0]/2 , num_xyz[1]/2, 0)
        p = camerapos[i]
        target = np.array(num_xyz)/2
        print(p)
        toward = target -p
        ax.quiver(p[0],p[1],p[2], toward[0], toward[1], toward[2], length=0.2 )  # 在每一个x,y,z坐标绘制矢量方向为u,v,w的箭头
    plt.show()


def test1():
    color_image1 = o3d.io.read_image('image/rgb1003.jpg')
    depth_image1 = o3d.io.read_image('image/depth1003.png')
    cloud = show_image.get_pointcloud_from_rgpd(color_image1, depth_image1)
    cloud = show_image.cloud_delet_background(cloud)
    voxel_size = 0.02
    scale = 0.01
    obj_path = 'mould_xifen/blade1_xifen.obj'
    voxel_grid = show_image.point_cloud_to_voxel(cloud, voxel_size, obj_path, scale)
    xyzvalues = plotxyz_inaabb(obj_path, voxel_grid, voxel_size, scale)
    # voxle_plot(voxel_grid, voxelsize)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    pos = ax.voxels(xyzvalues, edgecolor='k', linewidth=1, shade=False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Voxel Map')
    plt.grid(False)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    plot_vertise()

