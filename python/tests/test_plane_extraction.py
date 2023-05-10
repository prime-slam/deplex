import pytest
import deplex

import open3d as o3d
import numpy as np
import utils

from pathlib import Path


class TestTUMExtraction:
    DATA_PATH = Path(__file__).parent.resolve() / "data"
    IMAGE_PATH = str(DATA_PATH / "tum/1341848230.910894.png")
    IMAGE_SHAPE = (480, 640)
    INTRINSICS_PATH = str(DATA_PATH / "tum/TUM_fr3_long_val.K")
    CONFIG_PATH = None

    @pytest.fixture
    def pcd_points(self):
        intrinsics = np.asarray(np.loadtxt(self.INTRINSICS_PATH), dtype=np.float64)
        fx = intrinsics[0][0]
        fy = intrinsics[1][1]
        cx = intrinsics[0][2]
        cy = intrinsics[1][2]

        image = o3d.io.read_image(self.IMAGE_PATH)
        points = utils.depth_to_pcd_array(np.asarray(image, dtype=np.float64), fx, fy, cx, cy)
        return points

    @pytest.fixture
    def config(self):
        if self.CONFIG_PATH:
            return deplex.Config(path=self.CONFIG_PATH)
        return None

    @pytest.fixture
    def algorithm(self, pcd_points, config):
        if config:
            return deplex.PlaneExtractor(image_height=self.IMAGE_SHAPE[0],
                                         image_width=self.IMAGE_SHAPE[1],
                                         config=config)
        return deplex.PlaneExtractor(image_height=self.IMAGE_SHAPE[0], image_width=self.IMAGE_SHAPE[1])

    def test_default_config_extraction(self, algorithm, pcd_points):
        labels = algorithm.process(pcd_points)
        number_of_planes = max(labels)
        assert number_of_planes == 34

    def test_labels_size(self, algorithm, pcd_points):
        labels = algorithm.process(pcd_points)
        expected_labels_size = pcd_points.shape[0]
        assert labels.size == expected_labels_size

    @pytest.mark.skip(reason="Fatal error")
    def test_empty_input(self, algorithm):
        pcd_points = np.empty(shape=(3, 3))
        labels = algorithm.process(pcd_points)

