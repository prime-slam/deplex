#include <iostream>
#include <filesystem>
#include <gtest/gtest.h>

#include <deplex/plane_extractor.h>
#include <deplex/utils/depth_image.h>
#include <deplex/utils/eigen_io.h>

TEST(ReadImage, count_image) {
  const uint iterations = 500;
  const std::string IMAGE = "icl_nuim";

  std::__fs::filesystem::path data_dir = "/Users/denisovlev/Documents/CLionProjects/deplex/dataBenchmark";
  std::__fs::filesystem::path camera_path = data_dir / "configs" / "ICL_living_room.K";
  std::__fs::filesystem::path config_path = data_dir / "configs" / "ICL_living_room.ini";
  deplex::config::Config config (config_path.string());

  for (uint i = 0; i < iterations; i++) {
    std::__fs::filesystem::path image_path = data_dir / "icl_nuim" /  (std::to_string(i) + ".png");
    deplex::utils::DepthImage image(image_path.string());

    Eigen::Matrix3f intrinsics (deplex::utils::readIntrinsics(camera_path.string()));
    Eigen::MatrixXf pcd_array = image.toPointCloud(intrinsics);

    auto algorithm = deplex::PlaneExtractor(image.getHeight(), image.getWidth(), config);

    auto labels = algorithm.process(pcd_array);

    std::cout << image_path << ": " << labels.maxCoeff() << " planes found" << std::endl;
  }
}