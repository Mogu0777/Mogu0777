import numpy as np
import open3d as o3d

def read_obj_vertices(file_path):
    vertices = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertex = line.strip().split(' ')[1:]
                vertex = [float(coord) for coord in vertex]
                vertices.append(vertex)
    return vertices

def get_aabb(obj_file_path, scale):
    obj_mould = read_obj_vertices(obj_file_path)
    vertices = np.array(obj_mould) * scale
    max_xyz = np.max(vertices, axis=0)
    min_xyz = np.min(vertices, axis=0)
    return max_xyz, min_xyz

def get_voxelgrid(obj_file_path, voxel_size, scale, show_mould=False):
    mesh = o3d.io.read_triangle_mesh(obj_file_path)
    mesh.scale(scale, center=mesh.get_center())
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size)
    if show_mould:
        import plot
        plot.voxle_plot(voxel_grid, voxel_size)
        # o3d.visualization.draw_geometries([voxel_grid])
    return voxel_grid

def get_camera_radius(obj_path, near, far):
    pass

if __name__ == '__main__':
    obj_file_path1 = 'mould_xifen/impeller_xifen.obj'
    obj_file_path2 = 'mould_xifen/blade_xifen.obj'
    scale = 0.0025
    print(get_aabb(obj_file_path1, scale))
    print(get_voxelgrid(obj_file_path1, 0.02,scale, show_mould=True))
    # print(get_aabb(obj_file_path2))
    # print(get_voxelgrid(obj_file_path2, 0.01, show_mould=True))
