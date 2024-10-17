import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces
import get_obs
import get_obj
import pybullet as p
import show_image
import torch
import create_file
import open3d as o3d


class camera:
    def __init__(self):
        self.p_angle = 0
        self.e_angle = 0

    def move(self, action):
        # print(action)
        self.p_angle = action[0]
        self.e_angle = action[1]

class nbv_env(gym.Env):
    def __init__(self):
        self.width = 640  # 图像宽度
        self.height = 480  # 图像高度
        self.fov = 27  # 相机视角
        self.aspect = self.width / self.height  # 宽高比
        self.near = 0.01  # 最近拍摄距离
        self.far = 10  # 最远拍摄距离
        self.voxel_size = 0.04
        self.desired_overlap = 0

        self.targetPos = [0,0,0]
        self.sphere_radius = 1.36 # 球面半径

        self.episode_num = 0
        self.episode_step = 0
        self.camera = camera()
        self.w1 = 20 #为30
        self.w2 = 20  #为0
        self.w3 = 0
        self.w4 = 1  #为8
        self.recover = 0
        self.repeat_rate = 0
        self.reward = 0
        self.scale_list = create_file.get_model_scale(for_train=True)

        self.box_whd = [0.96, 0.96, 0.96]
        self.state = show_image.fixed_voxelgrid_reset(self.box_whd, self.voxel_size)[0]
        print('bounding box voxels nums=%d' + str(self.state.shape))
        self.observation = [0.5] * self.state.shape[0]
        self.action_dim = 2
        # 约束为[15,110]
        self.action_space = spaces.Box(low=np.array([0, 0]),
                                       high=np.array([2 * math.pi, math.pi]), dtype=np.float32)
        self.obj_voxel_num = create_file.get_model_voxel_num(for_train=True)

        # 连接引擎
        use_gui = False
        if use_gui:
            self.physicsClient = p.connect(p.GUI)  # 返回一个物理服务器ID，如果连接不成功返回-1
        else:
            self.physicsClient = p.connect(p.DIRECT)  # 在训练时采用这种方式连接
        # 不展示GUI的套件
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, rgbBackground=[1, 1, 1])
        # 添加资源路径
        # p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # print(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)

        # self.obj_mould, self.obj_voxel_mould, self.scale, model_urdf_path = create_file.random_get_mould(5, random=False, for_train=True)
        # self.obj_voxel = len(
        #     show_image.get_mould_voxel(self.obj_voxel_mould, self.scale, self.voxel_size)[1].get_voxels())
        #
        # # 加载模型
        # self.obj_maxxyz, self.obj_minxyz = get_obj.get_aabb(self.obj_mould, self.scale)
        # self.trayUid = p.loadURDF(model_urdf_path, basePosition=[-(self.obj_maxxyz[0]+self.obj_minxyz[0])/2,
        #                                                          -(self.obj_maxxyz[1]+self.obj_minxyz[1])/2,
        #                                                          -(self.obj_maxxyz[2]+self.obj_minxyz[2])/2])

    def reset(self):
        self.observation = [0.5] * self.state.shape[0]
        observation = torch.tensor(self.observation)
        self.episode_step = 0
        self.recover = 0
        self.repeat_rate = 0
        self.reward = 0
        epi_step = 0

        self.obj_mould, self.obj_voxel, self.scale, model_urdf_path= create_file.random_get_ShapeNet_model(self.scale_list,self.obj_voxel_num , index=0, for_train=True, random=False)
        # self.obj_voxel = len(show_image.get_mould_voxel(self.obj_voxel_mould, self.scale, self.voxel_size)[1].get_voxels())
        # self.obj_voxel = len(
        #     o3d.geometry.VoxelGrid.create_from_point_cloud(self.obj_cloudpoint, self.voxel_size).get_voxels())

        if self.episode_num > 0:
            p.removeBody(self.trayUid)
        # 加载模型
        # print(model_urdf_path)
        self.obj_maxxyz, self.obj_minxyz = get_obj.get_aabb(self.obj_mould, self.scale)
        self.trayUid = p.loadURDF(model_urdf_path, basePosition=[-(self.obj_maxxyz[0]+self.obj_minxyz[0])/2,
                                                                    -(self.obj_maxxyz[1]+self.obj_minxyz[1])/2, -(self.obj_maxxyz[2]+self.obj_minxyz[2])/2])
        # print(p.getNumBodies(self.physicsClient))
        return observation, epi_step

    def step(self, action):
        done = False
        obs_center = np.array(self.state)
        obs_center = tuple(obs_center.tolist())
        obs_label = self.observation.copy()
        r_dat = 0
        self.camera.move(action)

        cur_obs_array = get_obs.get_voxel_obeservation(self.obj_mould, self.scale, self.sphere_radius, self.box_whd, self.camera.p_angle,
                                                       self.camera.e_angle, self.targetPos, self.voxel_size, self.width,
                                                       self.height, self.fov, self.aspect, self.near, self.far)
        cur_obs_center = np.array(cur_obs_array[:, :3])
        cur_obs_center = tuple(cur_obs_center.tolist())
        cur_obs_label = cur_obs_array[:, 3]

        # 更新obs
        unknown = np.sum(np.array(obs_label) == 0.5)
        if self.episode_step == 0:
            for i in range(cur_obs_array.shape[0]):
                if cur_obs_center[i]!=[0,0,0]:
                    index = obs_center.index(cur_obs_center[i])
                    # print( np.sum(np.array(cur_obs_label)==2))
                    obs_label[index] = cur_obs_label[i]
            r_dat = np.sum(np.array(obs_label) == 1)
            # print('r_dat'+str(np.sum(np.array(obs_label) == 2)))
        else:
            for i in range(self.state.shape[0]):
                if obs_label[i] == 0.5:
                    if obs_center[i] in cur_obs_center:
                        index = cur_obs_center.index(obs_center[i])
                        obs_label[i] = cur_obs_label[index]
                        if obs_label[i]==1:
                            r_dat += 1
                else:
                    pass
            self.repeat_rate = 1-r_dat/np.sum(cur_obs_label == 1)

        r_rec = unknown-np.sum(np.array(obs_label) == 0.5)
        obs_label = np.array(obs_label)
        r_dat = r_dat / len(obs_label)
        r_rec = r_rec / len(obs_label)
        self.recover = np.sum(obs_label == 1) / self.obj_voxel

        if self.episode_step == 0:
            self.reward = self.w1 * r_dat + self.w2 * r_rec - self.w3 * self.episode_step
        else:
            # self.reward = self.w1 * r_dat + self.w2 * r_rec - self.w3 * self.episode_step - self.w4 * abs(
            #     self.repeat_rate - self.desired_overlap)
            self.reward = self.w1*r_dat+self.w2*r_rec-self.w3*self.episode_step-self.w4*(3-self.recover)
        # print(np.sum(obs_label == 0),np.sum(obs_label == 1),np.sum(obs_label == 2))
        # print('reward information' + str([self.w1 * r_dat, r_rec, self.w4*abs(repeat_rate-0.5), reward]))

        self.episode_step += 1
        new_observation = obs_label
        self.observation = new_observation
        if self.episode_step > 10:
            print('episode_step:' + str(self.episode_step))
            self.episode_num += 1
            done = True

        return new_observation, self.reward, done, self.episode_step

    def render(self):
        print('recover:%.3f   repeat_rate:%.3f   reward:%.3f   p_angle:%.3f    e_angle:%.3f ' % (self.recover, self.repeat_rate,self.reward,
                                                                              self.camera.p_angle, self.camera.e_angle))
        return round(float(self.camera.p_angle),4), round(float(self.camera.e_angle),4) , round(float(self.recover),3)

    def show_image(self, x):
        pass
