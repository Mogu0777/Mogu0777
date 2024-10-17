import numpy
import open3d as o3d
import numpy as np
import math
import get_obj
import pybullet as p

def get_mould_voxel(obj_path, scale, voxel_size):
    obj_mould = get_obj.read_obj_vertices(obj_path)
    max_xyz, min_xyz = get_obj.get_aabb(obj_path, scale)
    mid = (max_xyz+min_xyz)/2
    vertices = np.array(obj_mould) * scale - [mid[0], mid[1],0]
    # vertices = np.array(obj_mould)*scale
    cloud = creat_point_cloud(vertices, np.zeros((len(vertices), 3)))
    _, min_xyz = get_obj.get_aabb(obj_path, scale)
    voxel_gird = point_cloud_to_voxel(cloud, voxel_size, obj_path, scale)
    return cloud, voxel_gird

def get_mould_voxel_excluding_bottom(obj_path, scale, voxel_size):
    obj_mould = get_obj.read_obj_vertices(obj_path)
    max_xyz, min_xyz = get_obj.get_aabb(obj_path, scale)
    mid = (max_xyz+min_xyz)/2
    vertices = np.array(obj_mould) * scale - [mid[0], mid[1],0]

    delete_index = []
    for i in range(vertices.shape[0]):
        if vertices[i][2] < min_xyz[2]+voxel_size/2:
            delete_index.append(i)
    vertices = np.delete(vertices, delete_index, axis=0)

    cloud = creat_point_cloud(vertices, np.zeros((len(vertices), 3)))
    _, min_xyz = get_obj.get_aabb(obj_path, scale)
    voxel_gird = point_cloud_to_voxel(cloud, voxel_size, obj_path, scale)
    return cloud, voxel_gird

def get_exist_voxelnum_excluding_bottom(obj_path, exist_voxel, scale, voxel_size):
    max_xyz, min_xyz = get_obj.get_aabb(obj_path, scale)
    center, _ = get_voxel_centerandcolor(exist_voxel)
    delete_index = []
    for i in range(center.shape[0]):
        if center[i][2]<min_xyz[2]+voxel_size/2:
            delete_index.append(i)
    center = np.delete(center, delete_index, axis=0)
    exist_voxelnum = len(center)
    return exist_voxelnum

# 将点云转化为体素网格
def point_cloud_to_voxel(point_cloud, voxel_size, obj_file_path, scale):
    # _, base_origin = get_obj.get_aabb(obj_file_path, scale)
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud.points)
    pcd.colors = o3d.utility.Vector3dVector(point_cloud.colors)
    pcd.normals = o3d.utility.Vector3dVector(point_cloud.normals)
    # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(pcd, voxel_size, [-0.48,-0.48,-0.48], [0.48,0.48,0.48])
    # origin_xyz = voxel_grid.origin
    # # print(origin_xyz, base_origin)
    # diff = np.array((origin_xyz - base_origin) / voxel_size, dtype=int)
    #
    # voxel_grid.origin = base_origin + diff * voxel_size
    # print(voxel_grid.origin)
    return voxel_grid

# 计算旋转变化矩阵
def compute_R_T_C(alpha, beta, gamma):
    cos_a = math.cos(alpha)
    sin_a = math.sin(alpha)
    cos_b = math.cos(beta)
    sin_b = math.sin(beta)
    cos_g = math.cos(gamma)
    sin_g = math.sin(gamma)
    R_T_C = np.array(
        [[cos_g * cos_b, -sin_g * cos_a + cos_g * sin_b * sin_a, sin_g * sin_a + cos_g * sin_b * cos_a],
         [sin_g * cos_b, cos_g * cos_a + sin_g * sin_b * sin_a, -cos_g * sin_a + sin_g * sin_b * cos_a],
         [-sin_b, cos_b * sin_a, cos_b * cos_a]])
    return R_T_C

# 将相机坐标系转化为世界坐标系
def trans(obj_path, scale,r, pcd, Azimuth_angle,Elevation_angle):
    max_xyz, min_xyz = get_obj.get_aabb(obj_path, scale)
    camera_points = np.asarray(pcd.points).T - np.array([0,0,r]).reshape(3,1)
    R_T_C = compute_R_T_C(Elevation_angle, math.pi, Azimuth_angle - math.pi / 2)
    point = np.dot(R_T_C, camera_points)
    new_point_cloud = creat_point_cloud(o3d.utility.Vector3dVector(point.T), pcd.colors)
    return new_point_cloud

# 获取体素中心信息和颜色信息
def get_voxel_centerandcolor(voxelgrid):
    voxel_all = voxelgrid.get_voxels()
    center = []
    color = []
    for i in voxel_all:
        center.append(voxelgrid.get_voxel_center_coordinate(i.grid_index))
        color.append(i.color)
    center = np.around(np.array(center), decimals=6)
    color = np.array(color)
    return center, color

# 将体素转化为体素数组
def voxel_to_voxelarrary(voxel_grid, labels):
    voxel_center, _ = get_voxel_centerandcolor(voxel_grid)
    labels_array = np.ones((voxel_center.shape[0], 1)) * labels
    # print(voxel_center.shape, labels_array.shape)
    if len(voxel_center)!=0:
        voxel_array = np.concatenate((voxel_center, labels_array), axis=1)
        voxel_array = np.around(voxel_array, decimals=6)
    else:
        voxel_array = np.array([[0,0,0,1]])
    return voxel_array

# 获得已知表面labs
def get_exist_labls(exist_pcd,obj_path, scale, voxel_size):
    voxel_exist = point_cloud_to_voxel(exist_pcd, voxel_size,obj_path, scale)
    # o3d.visualization.draw_geometries([voxel_exist])
    # print(voxel_exist)
    voxel_exist_arrary = voxel_to_voxelarrary(voxel_exist, labels=1)
    return voxel_exist_arrary

# 获得遮挡部分点云
def get_cloud_hidden(pointcloud, reduce_size, cameraPos, obj_path, scale):
    add_points = []
    pointcloud_reduce = o3d.geometry.PointCloud.voxel_down_sample(pointcloud, reduce_size/2)
    max_xyz, min_xyz = get_obj.get_aabb(obj_path, scale)
    for i in pointcloud_reduce.points:
        radiation = i - cameraPos
        radiation = radiation / math.sqrt(radiation[0] ** 2 + radiation[1] ** 2 + radiation[2] ** 2)
        n = 5
        while True:
            add_point = i + radiation * n * reduce_size
            if min_xyz[0] < add_point[0] < max_xyz[0] and min_xyz[1] < add_point[1] < max_xyz[1] and min_xyz[2] < \
                    add_point[2] < max_xyz[2]:
                add_points.append(add_point)
                n += 1
            else:
                break
    return add_points

# 获得空点云
def get_cloud_free(pointcloud, reduce_size, cameraPos, bounding_box_whd):
    add_points = []
    min_xyz = -np.array(bounding_box_whd)/2
    max_xyz = np.array(bounding_box_whd)/2
    pointcloud_reduce = o3d.geometry.PointCloud.voxel_down_sample(pointcloud, reduce_size/2)
    for i in pointcloud_reduce.points:
        radiation = i - cameraPos
        radiation = radiation / math.sqrt(radiation[0] ** 2 + radiation[1] ** 2 + radiation[2] ** 2)
        n = 1  #画图为0，训练为4
        while True:
            add_point = i - radiation * n * reduce_size
            if min_xyz[0] < add_point[0] < max_xyz[0] and min_xyz[1] < add_point[1] < max_xyz[1] and min_xyz[2] < \
                    add_point[2] < max_xyz[2]:
                add_points.append(add_point)
                n += 1
            else:
                break
    pcd_free = creat_point_cloud(add_points, np.zeros((len(add_points), 3)))
    return pcd_free
# 创建点云
def creat_point_cloud(point, color):
    pcd = o3d.geometry.PointCloud()  # 实例化一个pointcloud类
    pcd.points = o3d.utility.Vector3dVector(point)  # 给该类传入坐标数据，此时pcd.points已经是一个点云了
    pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd

# 根据点云创建体素observation
def get_voxelobs_frompcd(exist_pcd, free_pcd, voxel_size, obj_path, scale):
    voxel_exist = point_cloud_to_voxel(exist_pcd, voxel_size, obj_path, scale)
    voxel_free = point_cloud_to_voxel(free_pcd, voxel_size, obj_path, scale)

    # 删除重叠体素
    voxel_exist_center = get_voxel_centerandcolor(voxel_exist)[0].tolist()
    voxel_free_center = get_voxel_centerandcolor(voxel_free)[0].tolist()
    set1 = set(map(tuple, voxel_exist_center))
    set2 = set(map(tuple, voxel_free_center))
    intersection = list(map(list, set1.intersection(set2)))
    # print(intersection)
    for i in intersection:
        voxel_free_center.remove(i)
    # set2 = set(map(tuple, voxel_free_center))
    # intersection = list(map(list, set1.intersection(set2)))
    # print(intersection)
    voxel_free_pcd = creat_point_cloud(voxel_free_center, np.zeros((len(voxel_free_center), 3)))
    voxel_free = point_cloud_to_voxel(voxel_free_pcd, voxel_size, obj_path, scale)

    voxel_exist_arrary = voxel_to_voxelarrary(voxel_exist, labels=1)
    voxel_free_arrary = voxel_to_voxelarrary(voxel_free, labels=0)
    voxel_observation = np.concatenate((voxel_exist_arrary, voxel_free_arrary), axis=0)
    return voxel_observation

# 从RGBD图创建点云  p00=2.6033, p11=4.1653
def get_pointcloud_from_rgpd(color_image, depth_image, width=960, height=600, p00=3.124, p11=4.1653):
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image,
                                                                    convert_rgb_to_intensity=False)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.Kinect2DepthCameraDefault)
    # fx=P[0，0]*W/2;  fy=P[1,1]*H/2
    intrinsic.set_intrinsics(width=width, height=height, fx=p00 * width / 2, fy=p11 * height / 2, cx=width / 2,
                             cy=height / 2)
    # 将深度图转化为点云
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    # o3d.visualization.draw_geometries([point_cloud])
    return point_cloud

# 删除背景
def cloud_delet_background(cloud):
    point = np.asarray(cloud.points)
    color = np.asarray(cloud.colors)
    back_color = np.array([0.9,0.9,0.9])
    res = (color > back_color).all(1)
    point = np.delete(point, res, axis=0)
    color = np.delete(color, res, axis=0)
    new_cloud = creat_point_cloud(point, color)
    return new_cloud

def get_exist_obs(obj_path, scale, radius, color_image, depth_image, voxel_size, p_angle, e_angle):
    # 将深度图转化为点云
    point_cloud = get_pointcloud_from_rgpd(color_image, depth_image)
    # point_cloud = cloud_delet_background(point_cloud)
    # 将相机坐标系转化为世界坐标系
    pointcloud_world = trans(obj_path, scale, radius, point_cloud, p_angle, e_angle)
    exist_obeservation = get_exist_labls(pointcloud_world,obj_path, scale, voxel_size=voxel_size)
    return exist_obeservation

def show_mould(obj_path,scale, radius, bounding_box_whd, color_image, depth_image, cameraPos, voxel_size, p_angle, e_angle,
               dispaly=False):
    # 将深度图转化为点云
    point_cloud = get_pointcloud_from_rgpd(color_image, depth_image)
    # point_cloud = cloud_delet_background(point_cloud)
    # 将相机坐标系转化为世界坐标系
    pointcloud_world = trans(obj_path,scale,radius, point_cloud, p_angle, e_angle)
    # 创建空部分点云
    pcd_free = get_cloud_free(pointcloud_world, voxel_size, cameraPos, bounding_box_whd)
    new_pcd = pcd_free + pointcloud_world

    # 将点云转换为体素
    voxel_pcd = point_cloud_to_voxel(new_pcd, voxel_size, obj_path,scale)
    if dispaly == True:
        o3d.visualization.draw_geometries([voxel_pcd])  # 显示一下

    # 获取体素观测信息
    voxel_obeservation = get_voxelobs_frompcd(pointcloud_world, pcd_free, voxel_size, obj_path, scale)
    return new_pcd, voxel_pcd, voxel_obeservation

# 合并体素体素网格
def combine_voxel(obj_path, scale, voxela, cloudpoint, voxel_size):
    center, color = get_voxel_centerandcolor(voxela)
    point = np.concatenate((cloudpoint.points, center), axis=0)
    color = np.concatenate((cloudpoint.colors, color), axis=0)
    print(center.shape, point.shape)
    new_pcd = creat_point_cloud(point, color)
    o3d.visualization.draw_geometries([new_pcd])
    voxel_grid = point_cloud_to_voxel(new_pcd, voxel_size, obj_path, scale)
    o3d.visualization.draw_geometries([voxel_grid])  # 显示一下

# 绘制针对模型得到包围盒
def voxelgrid_reset(obj_minxyz, obj_maxxyz, voxel_size):
    obj_minxyz2 = obj_minxyz - np.array([(obj_maxxyz[0]+obj_minxyz[0])/2, (obj_maxxyz[1]+obj_minxyz[1])/2, 0])
    print("bound box minxyz:" + str(obj_minxyz2 - voxel_size))
    print("bound box maxxyz:"+str(obj_maxxyz+voxel_size))
    voxel_grid = o3d.geometry.VoxelGrid.create_dense(obj_minxyz2 - voxel_size, [0, 0, 0], voxel_size,
                                                     width=obj_maxxyz[0] - obj_minxyz[0] + voxel_size * 2,
                                                     height=obj_maxxyz[1] - obj_minxyz[1] + voxel_size * 2,
                                                     depth=obj_maxxyz[2] - obj_minxyz[2] + voxel_size * 2)
    voxel_grid_center, _ = get_voxel_centerandcolor(voxel_grid)
    # voxel_grid_arrary = voxel_to_voxelarrary(voxel_grid, labels=0)
    return voxel_grid_center, voxel_grid

# 绘制固定长宽高包围盒
def fixed_voxelgrid_reset(box_whd, voxel_size):
    print("bound box width height and depth:"+str(box_whd))
    box_minxyz = -1 * np.array(box_whd)/2
    voxel_grid = o3d.geometry.VoxelGrid.create_dense(box_minxyz, [0, 0, 0], voxel_size,
                                                     width=box_whd[0], height=box_whd[1], depth=box_whd[2])
    voxel_grid_center, _ = get_voxel_centerandcolor(voxel_grid)
    # voxel_grid_arrary = voxel_to_voxelarrary(voxel_grid, labels=0)
    return voxel_grid_center, voxel_grid

# 绘制坐标轴
def normal_xyz():
    x = []
    y = []
    z = []
    n = 50
    color_x = np.array([254,0,0]*n).reshape(n,3)
    color_y = np.array([0, 254, 0]*n).reshape(n,3)
    color_z = np.array([0, 0, 254]*n).reshape(n,3)
    for i in range(n):
        x.append([i*0.01,0,0])
        y.append([0, i * 0.01, 0])
        z.append([ 0, 0, i * 0.01])
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    point = np.concatenate((x,y,z),axis=0)
    color = np.concatenate((color_x,color_y,color_z),axis=0)
    cloud = creat_point_cloud(point,color)
    return cloud

# 绘制体素
def plot(voxel_grid_center, voxel_size):
    minxyz = np.min(voxel_grid_center, axis=0)
    maxxyz = np.max(voxel_grid_center, axis=0)
    num_xyz = list(map(int, (maxxyz-minxyz)/voxel_size+1))
    plot_area = np.zeros(num_xyz)
    print(num_xyz)
    for i in range(len(voxel_grid_center)):
        exist_voxel = list(map(int, (voxel_grid_center[i] - minxyz) / voxel_size))
        plot_area[exist_voxel[0], exist_voxel[1], exist_voxel[2]] = 1
    return plot_area

def test():
    stl_path = 'turbine_blade.STL'
    points = []
    f = open(stl_path)
    lines = f.readlines()
    prefix = 'vertex'
    num = 3
    for line in lines:
        # print (line)

        if line.startswith(prefix):

            values = line.strip().split()
            # print(values[1:4])
            if num % 3 == 0:
                points.append(values[1:4])
                num = 0
            num += 1
        # print(type(line))
    points = np.array(points)
    f.close()
    print(points.shape)

if __name__ == '__main__':
    # 测试包围盒
    obj_mould = './mould/impeller.obj'
    tatd = './mould/impeller_xifen.obj'
    max_xyz, min_xyz = get_obj.get_aabb(obj_mould, 0.002)
    print(get_obj.get_aabb(tatd, 0.002))
    voxel_size=0.02
    _, grid = voxelgrid_reset(min_xyz, max_xyz, voxel_size)

    a, m = get_mould_voxel(tatd, 0.002, voxel_size)
    o3d.visualization.draw_geometries([a])
    combine_voxel(obj_mould, 0.002, grid, a, voxel_size)


