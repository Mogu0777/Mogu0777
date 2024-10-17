from random import randint

import pybullet as p
import numpy as np
import cv2
import PIL.Image as Image
import open3d as o3d
import math
import get_obj
import show_image
import plot
import create_file

# 获得相机姿态
def get_cameraPosandcameraupPos(sphere_radius, Azimuth_angle, Elevation_angle, targetPos):
    cameraPos_x = sphere_radius*math.cos(Azimuth_angle)*math.sin(Elevation_angle)+targetPos[0]
    cameraPos_y = sphere_radius*math.sin(Azimuth_angle)*math.sin(Elevation_angle)+targetPos[1]
    cameraPos_z = sphere_radius*math.cos(Elevation_angle)+targetPos[2]
    cameraPos = [cameraPos_x, cameraPos_y, cameraPos_z]  # 相机位置
    if Elevation_angle == 0:
        cameraupPos = [1, 0, 0]
    else:
        toward = np.array(targetPos)-np.array(cameraPos)  # 相机朝向
        # line2 = np.array([toward[1], -toward[0], 0])  # 朝向向量与z轴组成平面的法向量
        line2 = np.cross(toward, [0, 0, 1])
        cameraupPos = np.cross(line2, toward)  # 计算相机顶端朝向
    # 获取摄像机姿态信息
    return cameraPos, cameraupPos

# 获得相机投影矩阵
def get_viewMatrixandprojection_matrix(cameraPos, targetPos, cameraupPos, fov, aspect, near, far):
    viewMatrix = p.computeViewMatrix(
        cameraEyePosition=cameraPos,
        cameraTargetPosition=targetPos,
        cameraUpVector=cameraupPos,
        physicsClientId=0
    )  # 计算视角矩阵
    viewMatrix_arrary = np.array(viewMatrix).reshape(4, 4).T
    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)  # 计算投影矩阵
    return viewMatrix, projection_matrix, viewMatrix_arrary

# 连接引擎
use_gui = True
if use_gui:
    physicsClient = p.connect(p.GUI)  # 返回一个物理服务器ID，如果连接不成功返回-1
else:
    physicsClient = p.connect(p.DIRECT)  # 在训练时采用这种方式连接

# 不展示GUI的套件
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, rgbBackground=[1, 1, 1])

# 添加资源路径
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# print(pybullet_data.getDataPath())

# planeUid = p.loadURDF("plane.urdf", useMaximalCoordinates=True)  # 加载一个地面
# trayUid = p.loadURDF("tray/blade.urdf", basePosition=[0, 0, 0])  # 加载一个箱子，设置初始位置为（0，0，0）


_, obj_mould, scale, model_urdf_path = create_file.random_get_mould(5, random=False)
print(obj_mould, scale, model_urdf_path)

voxel_size = 0.02

# obj_mould = './mould/blade_xifen.obj'
max_xyz, min_xyz = get_obj.get_aabb(obj_mould, scale)
print(max_xyz, min_xyz)

trayUid = p.loadURDF(model_urdf_path, basePosition=[-(max_xyz[0]+min_xyz[0])/2, -(max_xyz[1]+min_xyz[1])/2,
                                                         -(max_xyz[2]+min_xyz[2])/2])  # 加载一个箱子，设置初始位置为（0，0，0）
p.setGravity(0, 0, -10)

width = 960  # 图像宽度
height = 600  # 图像高度
fov = 27  # 相机视角
aspect = width/height  # 宽高比
near = 0.01  # 最近拍摄距离
far = 10  # 最远拍摄距离

Azimuth_angle1 = math.pi*7/4  # 方位角
Elevation_angle1 = 0  # 仰角

whl = max_xyz-min_xyz
sphere_radius = 0.72
targetPos = [0, 0, 0]  # 目标位置，与相机位置之间的向量构成相机朝向

print('targetPos:' + str(targetPos))
cameraPos, cameraupPos = get_cameraPosandcameraupPos(sphere_radius, Azimuth_angle1, Elevation_angle1, targetPos)
viewMatrix, projection_matrix, viewMatrix_arrary = get_viewMatrixandprojection_matrix(cameraPos, targetPos, cameraupPos,
                                                                                      fov, aspect, near,
                                                                                      far)

cameraPos1 = cameraPos
viewMatrix1 = viewMatrix_arrary
print('camerapos'+str(cameraPos1))
print(viewMatrix1)

w, h, rgb, depth, seg = p.getCameraImage(width, height, viewMatrix, projection_matrix,
                                         renderer=p.ER_BULLET_HARDWARE_OPENGL)
# angle =[[3.5884, 1.437], [0.004, 0.2618], [6.2832, 2.4435], [4.5019, 2.4435], [1.918, 2.4435]] #model1
# angle = [[6.2832,  1.1764], [0.7443,  2.0944], [1.0028,  0.2618], [2.1655,  2.0944], [3.6754,  0.7497], [4.5789,  2.0918], [4.1128,  1.4465]] #model2
# angle = [[5.0130, 1.2214], [0.0006, 0.9961], [3.9359, 1.0646], [1.1496, 1.2112], [2.4562, 1.1764]] #model3
# angle = [[1.6576, 0.2618], [4.6536, 0.2618], [0.6613, 1.5955], [1.0869, 2.7925], [5.5179, 2.7925]] #model4
# angle =[[0.818,1.361],[0.015,0.262],[0,2.269],[3.032,0.289],[1.631,2.269],[3.425,2.269],[4.99,2.269]] #model5
# angle =[[3.0816, 1.6459], [6.2709, 0.2619], [6.2779, 1.701], [6.0759, 2.2689], [2.0172, 2.2689], [4.3217, 2.2689]] #model6
angle = [[3.6832862,  0.02279442], [3.6832862,  0.02279442]]

# 由此便可以得到RGB图像、深度图像、分割标签图像
def take_picture():
    i = 0
    camerapos = []
    # 开始渲染
    while True:
        p.stepSimulation()
        keys = p.getKeyboardEvents()

        if ord("z") in keys and keys[ord("z")] & p.KEY_WAS_RELEASED:
            Azimuth_angle2 = angle[i][0]
            Elevation_angle2 = angle[i][1]
            # 随机改变相机位置
            # Azimuth_angle2 = random.random() * 2 * math.pi
            # Elevation_angle2 = random.random() * math.pi / 2
            cameraPos, cameraupPos = get_cameraPosandcameraupPos(sphere_radius, Azimuth_angle2, Elevation_angle2,
                                                                 targetPos)
            camerapos.append((cameraPos-min_xyz)/voxel_size)
            viewMatrix, projection_matrix, viewMatrix_arrary = get_viewMatrixandprojection_matrix(cameraPos, targetPos,
                                                                                                  cameraupPos, fov,
                                                                                                  aspect, near, far)
            images = p.getCameraImage(width, height, viewMatrix, projection_matrix,
                                      renderer=p.ER_BULLET_HARDWARE_OPENGL)

            rgbImg = cv2.cvtColor(images[2], cv2.COLOR_BGR2RGB)
            cv2.imwrite('image/rgb100'+str(i)+'.jpg', rgbImg)

            depImg = far*near/(far-(far-near)*images[3])
            depImg = np.asanyarray(depImg).astype(np.float32)*1000.
            depImg = (depImg.astype(np.uint16))
            print(type(depImg[0, 0]))
            depImg = Image.fromarray(depImg)
            depImg.save('image/depth100'+str(i)+'.png')
            cv2.imwrite('image/seg100'+str(i)+'.jpg', images[4])
            i = i+1

            if i == np.array(angle).shape[0]:
                break
    return camerapos

def show_one_exist_results(i, angle, obj_path, scale, r):
    color_image1 = o3d.io.read_image('image/rgb100'+str(i)+'.jpg')
    depth_image1 = o3d.io.read_image('image/depth100'+str(i)+'.png')
    cloud1 = show_image.get_pointcloud_from_rgpd(color_image1, depth_image1)
    cloud1 = show_image.cloud_delet_background(cloud1)
    cloud1_trast = show_image.trans(obj_path, scale, r, cloud1, angle[i][0], angle[i][1])
    return cloud1_trast

def show_one_existandhidden_results(i, angle, obj_path, voxel_size, scale, r):
    color_image1 = o3d.io.read_image('image/rgb100'+str(i)+'.jpg')
    depth_image1 = o3d.io.read_image('image/depth100'+str(i)+'.png')
    cloud1 = show_image.get_pointcloud_from_rgpd(color_image1, depth_image1)
    cloud1_trast = show_image.trans(obj_path, scale, r, cloud1, angle[i][0], angle[i][1])
    cameraPos, _ = get_cameraPosandcameraupPos(sphere_radius, angle[i][0], angle[i][1], targetPos)
    hidden_point = show_image.get_cloud_hidden(cloud1_trast, voxel_size, cameraPos, obj_path, scale)
    cloud1 = show_image.cloud_delet_background(cloud1_trast)
    return cloud1, hidden_point

def show_one_existandfree_results(i, angle, obj_path, voxel_size, scale, r):
    color_image1 = o3d.io.read_image('image/rgb100'+str(i)+'.jpg')
    depth_image1 = o3d.io.read_image('image/depth100'+str(i)+'.png')
    cloud1 = show_image.get_pointcloud_from_rgpd(color_image1, depth_image1)
    cloud1_trast = show_image.trans(obj_path, scale, r, cloud1, angle[i][0], angle[i][1])
    cameraPos, _ = get_cameraPosandcameraupPos(sphere_radius, angle[i][0], angle[i][1], targetPos)
    free_point = show_image.get_cloud_free(cloud1_trast, voxel_size, cameraPos, obj_path, scale)
    cloud1 = show_image.cloud_delet_background(cloud1_trast)
    return cloud1, free_point

def voxelgrid_reset_plot(obj_file_path, voxel_size):
    obj_maxxyz, obj_minxyz = get_obj.get_aabb(obj_file_path)
    voxel_grid = o3d.geometry.VoxelGrid.create_dense(obj_minxyz+voxel_size, [0, 0, 0], voxel_size,
                                                     width=obj_maxxyz[0]-obj_minxyz[0]-3*voxel_size,
                                                     height=obj_maxxyz[1]-obj_minxyz[1]-3*voxel_size,
                                                     depth=obj_maxxyz[2]-obj_minxyz[2])
    voxel_grid_center, _ = show_image.get_voxel_centerandcolor(voxel_grid)
    # voxel_grid_arrary = voxel_to_voxelarrary(voxel_grid, labels=0)
    return voxel_grid_center

# 以包围盒形式表示
def show_more_existandhidden_results(irange, angle, obj_path, voxel_size, scale, r):
    cloud1, _ = show_one_existandhidden_results(0, angle, obj_path, voxel_size, scale, r)
    hidden_voxel_center1 = voxelgrid_reset_plot(obj_path, voxel_size).tolist()
    for i in range(irange+1):
        print(i)
        cloud2, free_point2 = show_one_existandfree_results(i, angle, obj_path, voxel_size, scale, r)
        cloud1 = cloud1+cloud2
        free_point2 = show_image.point_cloud_to_voxel(free_point2, voxel_size, obj_path, scale)
        free_voxel_center2, _ = show_image.get_voxel_centerandcolor(free_point2)
        # 求遮挡部分交集
        hidden_voxel_center1 = set(map(tuple, hidden_voxel_center1))-set(map(tuple, free_voxel_center2.tolist()))
        hidden_point = show_image.creat_point_cloud(list(hidden_voxel_center1),
                                                    np.zeros((len(hidden_voxel_center1), 3)))

        exist_voxel = show_image.point_cloud_to_voxel(cloud1, voxel_size, obj_path, scale)
        hidden_voxel = show_image.point_cloud_to_voxel(hidden_point, voxel_size, obj_path, scale)
        plot.voxle_plot_inaabb(obj_mould, exist_voxel, hidden_voxel, voxel_size, scale)
    return cloud1, hidden_point

# 以模型表示
def show_more_exist_results(irange, angle, obj_path, voxel_size, scale, r, camera, center):
    cloud1, _ = show_one_existandhidden_results(0, angle, obj_path, voxel_size, scale, r)
    _, hidden_voxel = show_image.get_mould_voxel(obj_path, scale, voxel_size)
    hidden_voxel_center1, _ = show_image.get_voxel_centerandcolor(hidden_voxel)
    hidden_voxel_center1 = hidden_voxel_center1.tolist()

    mould_voxelnum = len(show_image.get_mould_voxel(obj_mould, scale, voxel_size)[1].get_voxels())
    mould_voxelnum_excluding_bottom = len(show_image.get_mould_voxel_excluding_bottom(obj_mould, scale, voxel_size)[1].get_voxels())
    for i in range(irange+1):
        print(i)
        cloud2, _ = show_one_existandfree_results(i, angle, obj_path, voxel_size, scale, r)
        cloud1 = cloud1+cloud2
        exist_voxel = show_image.point_cloud_to_voxel(cloud1, voxel_size, obj_path, scale)
        exist_voxel_center, _ = show_image.get_voxel_centerandcolor(exist_voxel)
        # 求遮挡部分交集
        hidden_voxel_center1 = set(map(tuple, hidden_voxel_center1))-set(map(tuple, exist_voxel_center.tolist()))
        hidden_point = show_image.creat_point_cloud(list(hidden_voxel_center1),
                                                    np.zeros((len(hidden_voxel_center1), 3)))

        exist_voxel = show_image.point_cloud_to_voxel(cloud1, voxel_size, obj_path, scale)
        hidden_voxel = show_image.point_cloud_to_voxel(hidden_point, voxel_size, obj_path, scale)
        exist_voxel_num = len(exist_voxel.get_voxels())
        exist_voxel_num_excluding_bottom = show_image.get_exist_voxelnum_excluding_bottom(obj_path, exist_voxel, scale, voxel_size)

        print('real_coverage=%.4f,  no_bottom_coverage=%.4f'%(exist_voxel_num/mould_voxelnum,
                                                              exist_voxel_num_excluding_bottom/mould_voxelnum_excluding_bottom))
        # show_image.combine_voxel(obj_path, scale, hidden_voxel, cloud1, voxel_size)
        # o3d.visualization.draw_geometries([exist_voxel])
        # o3d.visualization.draw_geometries([hidden_voxel])
        plot.voxle_plot_inaabb(obj_mould, exist_voxel, hidden_voxel, voxel_size, scale, camera[i], center)
    return cloud1, hidden_point

def show_all_results(angle, obj_path, scale, r):
    cloud_all = show_one_exist_results(0, angle, obj_path, scale, r)
    for i in range(1, np.array(angle).shape[0]):
        cloud = show_one_exist_results(i, angle, obj_path, scale, r)
        cloud_all += cloud
    # o3d.visualization.draw_geometries([cloud_all])
    voxel_grid1 = show_image.point_cloud_to_voxel(cloud_all, voxel_size, obj_path, scale)
    plot.voxle_plot(voxel_grid1, voxel_size)

if __name__ == '__main__':
    camerapose = take_picture()
    # plot.plot_vertise(camerapose, targetPos)
    # targetPos = (targetPos-min_xyz)/voxel_size
    #绘制点云
    exist_cloud = show_one_exist_results(0, angle, obj_mould, scale, sphere_radius)
    free_cloud = show_image.get_cloud_free(exist_cloud, voxel_size, camerapose[0], bounding_box_whd=[0.52,0.52,0.52])
    normal_xyz = show_image.normal_xyz()
    cloudd = exist_cloud+free_cloud + normal_xyz
    o3d.visualization.draw_geometries([cloudd])
    #绘制体素
    # cloud1_trast, hidden_point = show_more_exist_results(np.array(angle).shape[0]-1, angle, obj_mould, voxel_size,
    #                                                       scale, sphere_radius, camerapose, targetPos)
    ##绘制视点位置
    # plot.plot_viewpoint(obj_mould, scale, voxel_size, camerapose, targetPos, min_xyz)


    # cloud, voxel = show_image.get_mould_voxel_excluding_bottom(obj_mould, scale, voxel_size)
    # # o3d.visualization.draw_geometries([cloud])
    # obj_voxel_num = len(show_image.get_mould_voxel_excluding_bottom(obj_mould, scale, voxel_size)[1].get_voxels())
    # obj_voxel_num2 = len(show_image.get_mould_voxel(obj_mould, scale, voxel_size)[1].get_voxels())
    # print(obj_voxel_num, obj_voxel_num2)
