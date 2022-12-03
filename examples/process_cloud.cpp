#include <deplex/plane_extractor.h>
#include <deplex/utils/image_to_cloud.h>

#include <filesystem>
#include <iostream>

int main() {
  std::filesystem::path data_dir =
      std::filesystem::current_path().parent_path().parent_path() / "data";
  std::filesystem::path image_path = data_dir / "tum/1341848230.910894.png";
  std::filesystem::path intrinsics_path =
      data_dir / "configs/TUM_fr3_long_val.K";
  std::filesystem::path config_path = data_dir / "configs/TUM_fr3_long_val.ini";

  constexpr int IMAGE_HEIGHT = 480, IMAGE_WIDTH = 640;

  deplex::config::Config config = deplex::config::Config(config_path);

  auto algorithm = deplex::PlaneExtractor(IMAGE_HEIGHT, IMAGE_WIDTH, config);
  Eigen::MatrixXf pcd_array =
      deplex::utils::readImage(image_path, intrinsics_path);

  auto labels = algorithm.process(pcd_array);
  std::cout << "Found planes: " << labels.maxCoeff() << '\n';

  return 0;
}