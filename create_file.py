import os
import get_obj
import math
import open3d as o3d
from random import randint
import show_image

# path_list.remove('.DS_Store')  # macos中的文件管理文件，默认隐藏，这里可以忽略

def floor(x, n):
    return math.floor(x*10**n)/10**n

def random_get_mould(index=3, for_train=False, random=True):
    if for_train:
        model_dir_path = '../mould'
        model_xifen_dir_path = '../mould_xifen'
        urdf_dir_path = '../urdf_test'
    else:
        model_dir_path = './mould'
        model_xifen_dir_path = './mould_xifen'
        urdf_dir_path = './urdf_test'
    model_path_list = os.listdir(model_dir_path)
    model_xife_path_list = os.listdir(model_xifen_dir_path)
    urdf_path_list = os.listdir(urdf_dir_path)

    if random:
        index_num = randint(0, 5)
    else:
        index_num = index
    print("index=%d"%index_num)
    scale_list = get_model_scale(for_train)
    scale = scale_list[index_num]
    model_path = model_dir_path+'/'+model_path_list[index_num]
    model_urdf_path = urdf_dir_path+'/'+urdf_path_list[index_num]
    model_xifen_path = model_xifen_dir_path+'/'+model_xife_path_list[index_num]
    # print(model_path, model_urdf_path)
    return model_path, model_xifen_path, scale, model_urdf_path

def random_get_ShapeNet_model(scale_list, voxel_num_list, index=0,for_train=False, random=True):
    if for_train:
        model_dir_path = '../ShapeNet_model'
        model_xifen_dir_path = '../ShapeNet_cloudpoint'
        urdf_dir_path = '../ShapeNet_urdf'
    else:
        model_dir_path = './ShapeNet_model'
        model_xifen_dir_path = './ShapeNet_cloudpoint'
        urdf_dir_path = './ShapeNet_urdf'
    model_path_list = os.listdir(model_dir_path)
    model_xife_path_list = os.listdir(model_xifen_dir_path)
    urdf_path_list = os.listdir(urdf_dir_path)

    if random:
        index_num = randint(0, len(model_path_list)-1)
    else:
        index_num = index
    print("index=%d"%index_num)
    scale = scale_list[index_num]
    voxel_num = voxel_num_list[index_num]
    model_path = model_dir_path+'/'+model_path_list[index_num] + '/model.obj'
    model_urdf_path = urdf_dir_path+'/'+urdf_path_list[index_num]
    # print(model_path, model_urdf_path, scale)
    return model_path, voxel_num, scale, model_urdf_path

def get_model_voxel_num(for_train=False):
    if for_train:
        model_cloud_dir_path = '../ShapeNet_cloudpoint'
    else:
        model_cloud_dir_path = './ShapeNet_cloudpoint'
    model_cloudpoint_path_list = os.listdir(model_cloud_dir_path)
    # print(model_path_list)
    voxel_size = 0.04

    voxel_num_list = []
    for i in range(len(model_cloudpoint_path_list)):
        model_pcd_path = model_cloud_dir_path+'/'+model_cloudpoint_path_list[i]
        model_pcd = o3d.io.read_point_cloud(model_pcd_path)
        voxel_num_list.append(len(show_image.point_cloud_to_voxel(model_pcd, voxel_size, 0, 0).get_voxels()))
    return voxel_num_list

def get_model_cloudpoint():
    # 模型路径，支持后缀：stl/ply/obj/off/gltf/glb
    model_dir_path = './ShapeNet_model'
    model_path_list = os.listdir(model_dir_path)
    scale_list = get_model_scale()
    for i in range(len(model_path_list)):
        print('creating cloudpoint file:%d/%d'%(i+1, len(model_path_list)))
        # 读入网格模型
        mesh = o3d.io.read_triangle_mesh(model_dir_path+'/'+model_path_list[i]+'/model.obj')
        mesh.scale(scale_list[i], center=mesh.get_center())
        # 计算网格顶点
        mesh.compute_vertex_normals()
        # 可视化网格模型
        o3d.visualization.draw_geometries([mesh])
        # poisson_disk方法采样5000个点
        pcd = mesh.sample_points_poisson_disk(number_of_points=40000, init_factor=2)
        # 可视化点云模型
        o3d.visualization.draw_geometries([pcd])
        # 保存
        base_name = os.path.splitext(model_path_list[i])[0]
        o3d.io.write_point_cloud("./ShapeNet_cloudpoint/"+base_name+".pcd", pcd)
    print('all model_CloudPoint files have created!')

def get_model_scale(for_train=False):
    if for_train:
        model_dir_path = '../ShapeNet_model'
    else:
        model_dir_path = './ShapeNet_model'
    # if for_train:
    #     model_dir_path = '../mould'
    # else:
    #     model_dir_path = './mould'
    model_path_list = os.listdir(model_dir_path)
    # print(model_path_list)
    boxsize = 0.96-0.12

    scale_list = []
    for i in range(len(model_path_list)):
        max_xyz, min_xyz = get_obj.get_aabb(model_dir_path+'/'+model_path_list[i]+'/model.obj', scale=1)
        # max_xyz, min_xyz = get_obj.get_aabb(model_dir_path+'/'+model_path_list[i], scale=1)
        scale = floor(boxsize/max(max_xyz-min_xyz),4)
        scale_list.append(scale)
    return scale_list

def get_mould_urdf():
    model_dir_path = './ShapeNet_model'
    model_path_list = os.listdir(model_dir_path)

    scale_list = get_model_scale()
    ## 假设我要新建10个txt文件,这里我用一个for循环
    for i in range(len(model_path_list)):
        ##这里的./指代的是当前文件夹, %i表示文件的名称,w表示没有该文件就新建,有就覆盖.
        print('creating urdf file:%d/%d'%(i+1, len(model_path_list)))
        base_name = os.path.splitext(model_path_list[i])[0]
        f = open('./ShapeNet_urdf/%s'%base_name + '.urdf' ,"w")
        f.write("""<robot name="myobj">
      <link name="tray_base_link">
        <inertial>
          <origin rpy="0 0 0" xyz="0 0 0"/>
          <mass value="0"/>
          <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
        </inertial>
        <visual>
          <origin rpy="0 0 0" xyz="0 0 0"/>
          <geometry>
            <mesh filename=\"."""+model_dir_path+"/"+model_path_list[i] +"/model.obj\" "+ "scale=\"" +str(scale_list[i])+" "+ str(scale_list[i])+" "+str(scale_list[i])+ "\"/>\n"
          """      </geometry>
          <material name="tray_material">
            <color rgba="0.7 0.7 0.7 1"/>
          </material>
        </visual>
      </link>
    </robot>""")  # 写入文件，空
        f.close()  # 执行完结束
    print('all model_urdf files have created!')

if __name__ == "__main__":
    # get_mould_urdf()
    get_model_cloudpoint()