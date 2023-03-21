import deplex
import deplex.utils
import numpy as np

from pathlib import Path


def main():
    data_dir = Path(__file__).parent.parent.parent.resolve() / "data"
    image_path = data_dir / Path("tum") / Path("1341848230.910894.png")
    config_path = data_dir / Path("configs") / Path("TUM_fr3_long_val.ini")
    intrinsics_path = data_dir / Path("configs") / Path("TUM_fr3_long_val.K")

    # Load intrinsics from file or initialize as 3x3 numpy array
    intrinsics = np.loadtxt(intrinsics_path)
    # Read image
    image = deplex.utils.DepthImage(str(image_path))
    # Transform image to points using camera intrinsics
    points = image.transform_to_pcd(intrinsics)
    # Read config
    config = deplex.Config(str(config_path))

    algorithm = deplex.PlaneExtractor(image.height, image.width, config=config)
    labels = algorithm.process(points)
    print(f"Number of found planes: {max(labels)}")


if __name__ == '__main__':
    main()
