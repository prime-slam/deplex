#include <deplex/plane_extractor.h>
#include <deplex/utils/eigen_io.h>
#include <deplex/utils/image.h>

#include <chrono>
#include <filesystem>
#include <iostream>

int main(int argc, char* argv[]) {
  std::filesystem::path data_dir = std::filesystem::current_path().parent_path().parent_path() / "data";
  std::filesystem::path image_path = data_dir / "tum/1341848230.910894.png";
  std::filesystem::path intrinsics_path = data_dir / "configs/TUM_fr3_long_val.K";
  std::filesystem::path config_path = data_dir / "configs/TUM_fr3_long_val.ini";

  constexpr int IMAGE_HEIGHT = 480, IMAGE_WIDTH = 640;
  int NUMBER_OF_RUNS = (argc > 1 ? std::stoi(argv[1]) : 1);

  deplex::config::Config config = deplex::config::Config(config_path.string());

  auto algorithm = deplex::PlaneExtractor(IMAGE_HEIGHT, IMAGE_WIDTH, config);

  Eigen::Matrix3f intrinsics(deplex::utils::readIntrinsics(intrinsics_path));
  Eigen::MatrixXf pcd_array = deplex::utils::Image(image_path).toPointCloud(intrinsics);
  int found_planes = 0;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < NUMBER_OF_RUNS; ++i) {
    auto labels = algorithm.process(pcd_array);
    found_planes = labels.maxCoeff();
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / NUMBER_OF_RUNS;

  std::cout << "Found planes: " << found_planes << '\n';
  std::cout << "Elapsed time (mks): " << elapsed_time << '\n';

  return 0;
}