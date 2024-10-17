import math
import pybullet as p
import pybullet_data
import numpy as np
import cv2
import PIL.Image as Image
import open3d as o3d
import math
import get_obj

def state_to_angle(state, p_angle_level=12, e_angle_level=6):
    e_angle_n = math.ceil(state / p_angle_level)
    if e_angle_n == 0 or e_angle_n == 6:
        p_angle_n = 0
    else:
        p_angle_n = state - 1 - (e_angle_n - 1) * p_angle_level
    p_angle = p_angle_n / p_angle_level * math.pi * 2
    e_angle = e_angle_n / e_angle_level * math.pi

    return p_angle, e_angle

# 获得相机姿态
def get_cameraPosandcameraupPos(sphere_radius, Azimuth_angle, Elevation_angle, targetPos):
    cameraPos_x = sphere_radius * math.cos(Azimuth_angle) * math.sin(Elevation_angle) + targetPos[0]
    cameraPos_y = sphere_radius * math.sin(Azimuth_angle) * math.sin(Elevation_angle) + targetPos[1]
    cameraPos_z = sphere_radius * math.cos(Elevation_angle) + targetPos[2]
    cameraPos = [cameraPos_x, cameraPos_y, cameraPos_z]  # 相机位置
    if Elevation_angle==0:
        cameraupPos = [1,0,0]
    else:
        toward = np.array(targetPos) - np.array(cameraPos)  # 相机朝向
        # line2 = np.array([toward[1], -toward[0], 0])  # 朝向向量与z轴组成平面的法向量
        line2 = np.cross(toward, [0,0,1])
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

if __name__ == '__main__':
    # 连接引擎
    use_gui = True
    if use_gui:
        physicsClient = p.connect(p.GUI)  # 返回一个物理服务器ID，如果连接不成功返回-1
    else:
        physicsClient = p.connect(p.DIRECT)  # 在训练时采用这种方式连接

    # 不展示GUI的套件
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)

    # 添加资源路径
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    print(pybullet_data.getDataPath())

    # planeUid = p.loadURDF("plane.urdf", useMaximalCoordinates=True)  # 加载一个地面
    # trayUid = p.loadURDF("tray/blade.urdf", basePosition=[0, 0, 0])  # 加载一个箱子，设置初始位置为（0，0，0）
    trayUid = p.loadURDF("my_test/blade.urdf", basePosition=[0, 0, 0])  # 加载一个箱子，设置初始位置为（0，0，0）
    p.setGravity(0, 0, -10)

    width = 1080  # 图像宽度
    height = 720  # 图像高度

    fov = 50  # 相机视角
    aspect = width / height  # 宽高比
    near = 0.01  # 最近拍摄距离
    far = 10  # 最远拍摄距离

    sphere_radius = 2  # 球面半径
    Azimuth_angle1 = math.pi *7/ 4  # 方位角
    Elevation_angle1 = 0 # 仰角

    obj_mould = 'blade.obj'
    max_xyz, min_xyz = get_obj.get_aabb(obj_mould)
    targetPos = [0, 0, max_xyz[2]/2]  # 目标位置，与相机位置之间的向量构成相机朝向

    i = 0
    # 开始渲染
    while True:
        p.stepSimulation()
        keys = p.getKeyboardEvents()

        if ord("z") in keys and keys[ord("z")] & p.KEY_WAS_RELEASED:
            p_angle, e_angle = state_to_angle(i)
            cameraPos, cameraupPos = get_cameraPosandcameraupPos(sphere_radius, p_angle, e_angle, targetPos)
            viewMatrix, projection_matrix, viewMatrix_arrary = get_viewMatrixandprojection_matrix(cameraPos, targetPos,
                                                                                                  cameraupPos, fov,
                                                                                                  aspect, near, far)
            images = p.getCameraImage(width, height, viewMatrix, projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)

            rgbImg = cv2.cvtColor(images[2], cv2.COLOR_BGR2RGB)
            cv2.imwrite('image/rgb' + str(i) + '.jpg', rgbImg)

            depImg = far * near / (far - (far - near) * images[3])
            depImg = np.asanyarray(depImg).astype(np.float32) * 1000.
            depImg = (depImg.astype(np.uint16))

            depImg = Image.fromarray(depImg)
            depImg.save('image/depth' + str(i) + '.png')
            cv2.imwrite('image/seg' + str(i) + '.jpg', images[4])
            i = i + 1