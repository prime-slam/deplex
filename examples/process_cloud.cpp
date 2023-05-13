#include <deplex/plane_extractor.h>
#include <deplex/utils/depth_image.h>
#include <deplex/utils/eigen_io.h>

#include <chrono>
#include <filesystem>
#include <iostream>

int main(int argc, char* argv[]) {
  std::filesystem::path data_dir = std::filesystem::current_path().parent_path().parent_path() / "data";
  std::filesystem::path image_path = data_dir / "tum/1341848230.910894.png";
  std::filesystem::path intrinsics_path = data_dir / "configs/TUM_fr3_long_val.K";
  std::filesystem::path config_path = data_dir / "configs/TUM_fr3_long_val.ini";

  int NUMBER_OF_RUNS = (argc > 1 ? std::stoi(argv[1]) : 1);

  deplex::config::Config config = deplex::config::Config(config_path.string());
  Eigen::Matrix3f intrinsics(deplex::utils::readIntrinsics(intrinsics_path.string()));
  deplex::utils::DepthImage image(image_path.string());
  Eigen::MatrixXf pcd_array = image.toPointCloud(intrinsics);

  auto algorithm = deplex::PlaneExtractor(image.getHeight(), image.getWidth(), config);

  int found_planes = 0;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < NUMBER_OF_RUNS; ++i) {
    auto labels = algorithm.process(pcd_array);
    found_planes = labels.maxCoeff();
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / NUMBER_OF_RUNS;
  auto fps = 1e6l / elapsed_time;

  std::cout << "Found planes: " << found_planes << '\n';
  std::cout << "Elapsed time (mks): " << elapsed_time << '\n';
  std::cout << "FPS: " << fps << '\n';

  return 0;
}