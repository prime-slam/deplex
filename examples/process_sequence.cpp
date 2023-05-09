#include <deplex/plane_extractor.h>
#include <deplex/utils/depth_image.h>
#include <deplex/utils/eigen_io.h>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <numeric>

int main(int argc, char* argv[]) {
  std::filesystem::path data_dir = std::filesystem::current_path().parent_path().parent_path() / "data";
  std::filesystem::path intrinsics_path = data_dir / "configs/TUM_fr3_long_val.K";
  std::filesystem::path config_path = data_dir / "configs/TUM_fr3_long_val.ini";

  std::string dataset_path = (argc > 1 ? argv[1] : (data_dir / "tum").string());
  int MAX_NUMBER_OF_FRAMES = (argc > 2 ? std::stoi(argv[2]) : -1);

  deplex::config::Config config = deplex::config::Config(config_path.string());
  Eigen::Matrix3f intrinsics(deplex::utils::readIntrinsics(intrinsics_path.string()));

  // Sort data entries
  std::vector<std::filesystem::directory_entry> sorted_input_data;
  for (auto const& entry : std::filesystem::directory_iterator(dataset_path)) {
    if (entry.path().extension() == ".png") {
      sorted_input_data.push_back(entry);
    }
  }
  sort(sorted_input_data.begin(), sorted_input_data.end());

  std::vector<size_t> time_vector;
  for (auto const& entry : sorted_input_data) {
    auto START_TIME = std::chrono::high_resolution_clock::now();
    deplex::utils::DepthImage image(entry.path().string());
    auto algorithm = deplex::PlaneExtractor(image.getHeight(), image.getWidth(), config);
    auto labels = algorithm.process(image.toPointCloud(intrinsics));
    auto elapsed_time =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - START_TIME)
            .count();
    time_vector.push_back(elapsed_time);
    if (time_vector.size() == MAX_NUMBER_OF_FRAMES) {
      break;
    }
  }
  size_t elapsed_time_min = *std::min_element(time_vector.begin(), time_vector.end());
  size_t elapsed_time_max = *std::max_element(time_vector.begin(), time_vector.end());
  size_t elapsed_time_mean = std::accumulate(time_vector.begin(), time_vector.end(), 0) / time_vector.size();

  std::cout << "Elapsed time (min): " << elapsed_time_min << '\n';
  std::cout << "Elapsed time (max): " << elapsed_time_max << '\n';
  std::cout << "Elapsed time (mean): " << elapsed_time_mean << '\n';
  std::cout << "FPS (max): " << 1e6l / elapsed_time_min << '\n';
  std::cout << "FPS (min): " << 1e6l / elapsed_time_max << '\n';
  std::cout << "FPS (mean): " << 1e6l / elapsed_time_mean << '\n';

  return 0;
}