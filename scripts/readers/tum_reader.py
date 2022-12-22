import cv2
import open3d as o3d
import numpy as np


class CameraIntrinsics:
    def __init__(self, width, height, fx, fy, cx, cy, factor):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.factor = factor
        if fx is not None and fy is not None and cx is not None and cy is not None:
            self.open3dIntrinsics = o3d.camera.PinholeCameraIntrinsic(
                width=width,
                height=height,
                fx=fx,  # X-axis focal length
                fy=fy,  # Y-axis focal length
                cx=cx,  # X-axis principle point
                cy=cy  # Y-axis principle point
            )


def depth_to_pcd_custom(
        depth_image: np.array,
        camera_intrinsics: CameraIntrinsics,
        initial_pcd_transform: list
) -> (o3d.geometry.PointCloud, np.array):
    points = np.zeros((camera_intrinsics.width * camera_intrinsics.height, 3))

    column_indices = np.tile(np.arange(camera_intrinsics.width), (camera_intrinsics.height, 1)).flatten()
    row_indices = np.transpose(np.tile(np.arange(camera_intrinsics.height), (camera_intrinsics.width, 1))).flatten()

    points[:, 2] = depth_image.flatten() / camera_intrinsics.factor
    points[:, 0] = (column_indices - camera_intrinsics.cx) * points[:, 2] / camera_intrinsics.fx
    points[:, 1] = (row_indices - camera_intrinsics.cy) * points[:, 2] / camera_intrinsics.fy

    zero_depth_indices = np.where(points[:, 2] == 0)[0]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.transform(initial_pcd_transform)

    return pcd, zero_depth_indices


def read_depth(depth_path):
    image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    # https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats#intrinsic_camera_calibration_of_the_kinect
    camera_intrinsics = CameraIntrinsics(
        width=640,
        height=480,
        fx=535.4,
        fy=539.2,
        cx=320.1,
        cy=247.6,
        factor=1.0
    )
    initial_pcd_transform = [[1, 0, 0, 0],
                             [0, -1, 0, 0],
                             [0, 0, -1, 0],
                             [0, 0, 0, 1]]
    pcd, _ = depth_to_pcd_custom(image,
                                 camera_intrinsics,
                                 initial_pcd_transform)

    return pcd
