from deplex.utils import DepthImage
import deplex

import open3d as o3d
import numpy as np

from pathlib import Path

data_dir = Path(__file__).parent.parent.parent.resolve() / "benchmark-artifact/data"
image_path = data_dir / Path("depth") / Path("000004415622.png")
config_path = data_dir / Path("config") / Path("TUM_fr3_long_val.ini")
intrinsics_path = data_dir / Path("config") / Path("intrinsics.K")
def process_cloud():
    config = deplex.Config(str(config_path))
    camera_intrinsic = np.genfromtxt(intrinsics_path)
    image = DepthImage(str(image_path))
    pcd_points = image.transform_to_pcd(camera_intrinsic)

    print("Image Height:", image.height, "Image Width:", image.width)
    print("Shape of PointCloud points:", pcd_points.shape)

    # Visualize PointCloud points using Open3D

    open3d_pcd = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(pcd_points)
    )

    o3d.visualization.draw_geometries([open3d_pcd])

    coarse_algorithm = deplex.PlaneExtractor(image_height=image.height, image_width=image.width, config=config)

    labels = coarse_algorithm.process(pcd_points)

    print(f"Number of found planes: {max(labels)}")
    print(np.unique(labels)) # Zero stands for non-planar point
    print("Labels shape:", labels.shape)

    color_set = np.random.rand(100 + 1, 3)
    pcd_colors = np.zeros(pcd_points.shape)

    non_planar_label = 0
    for color_id in np.unique(labels):
        if color_id != non_planar_label:
            pcd_colors[labels == color_id] = color_set[color_id]

    open3d_pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

    o3d.visualization.draw_geometries([open3d_pcd])

if __name__ == '__main__':
    process_cloud()