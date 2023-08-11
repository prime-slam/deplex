import open3d as o3d
import numpy as np

from deplex.utils import DepthImage

import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

data_dir = Path(__file__).parent.parent.parent.resolve() / "benchmark-artifact/data"
image_path = data_dir / Path("depth") / Path("000004415622.png")
config_path = data_dir / Path("config/cPlusPlus") / Path("TUM_fr3_long_val.ini")
intrinsics_path = data_dir / Path("config") / Path("intrinsics.K")

csv_file_labels = "" # File with marked planes
def alg():
    camera_intrinsic = np.genfromtxt(intrinsics_path, delimiter=',')

    image = DepthImage(image_path)
    pcd_points = image.transform_to_pcd(camera_intrinsic)

    print("Image Height:", image.height, "Image Width:", image.width)
    print("Shape of PointCloud points:", pcd_points.shape)

    # Visualize PointCloud points using Open3D

    open3d_pcd = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(pcd_points)
    )

    o3d.visualization.draw_geometries([open3d_pcd])

    labels = np.genfromtxt(csv_file_labels, delimiter=',')


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

def graphic():
    data = np.genfromtxt(Path(data_dir) / Path('process_sequence_50_snapshot.csv'), delimiter=',')

    # Создаем boxplot
    sns.boxplot(data=[data])

    plt.xticks([0], ['stable'])

    plt.ylabel("Время (мс.)")

    plt.show()
if __name__ == '__main__':
    graphic()