import open3d as o3d
import numpy as np
from PIL import Image

# Dataset and camera parameters.
SCALE_FACTOR = 1
RGB_NUMBER_OF_SHADES = 256

fx = 481.20
fy = -480.00
cx = 319.50
cy = 239.50


def get_colors_vector(colors_image_path):
    colors_image = Image.open(colors_image_path)
    colors_matrix = np.array(colors_image) / RGB_NUMBER_OF_SHADES
    colors_vector = colors_matrix.reshape((colors_matrix.shape[0] * colors_matrix.shape[1], 3))
    return colors_vector


def get_points_vector(depth_image_path):
    depths_image = Image.open(depth_image_path)
    depths_matrix = np.array(depths_image)

    M, N = depths_matrix.shape
    V = np.repeat(np.arange(M), N)
    U = np.tile(np.arange(N), M)
    D = depths_matrix.reshape(depths_matrix.size)

    Z = D / SCALE_FACTOR
    X = (U - cx) * Z / fx
    Y = (V - cy) * Z / fy

    xyz_points_vector = np.column_stack((X, Y, Z))
    return xyz_points_vector


def visualize_pointcloud(points_vector, colors_vector):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_vector)
    pcd.colors = o3d.utility.Vector3dVector(colors_vector)

    pcd.transform([[-1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, -1]])
    o3d.visualization.draw_geometries([pcd])


def read_depth(depth_path):
    points_vector = get_points_vector(depth_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_vector)
    pcd.transform([[-1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, -1]])

    return pcd
