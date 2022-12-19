import numpy as np

__all__ = ['depth_to_pcd_array']

def depth_to_pcd_array(depth_image: np.array, fx, fy, cx, cy, scale=1.0):
    """Transform depth image array [height x width] to organized PCD points"""
    height, width = depth_image.shape
    points = np.zeros((width * height, 3))

    column_indices = np.tile(np.arange(width), (height, 1)).flatten()
    row_indices = np.transpose(np.tile(np.arange(height), (width, 1))).flatten()

    points[:, 2] = depth_image.flatten() / scale
    points[:, 0] = (column_indices - cx) * points[:, 2] / fx
    points[:, 1] = (row_indices - cy) * points[:, 2] / fy

    return points