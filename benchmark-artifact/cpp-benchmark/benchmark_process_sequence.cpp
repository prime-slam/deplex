#include <deplex/plane_extractor.h>
#include <deplex/utils/depth_image.h>
#include <deplex/utils/eigen_io.h>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <numeric>

double calculateVariance(const Eigen::VectorXd& data, double mean) {
  double sum = 0;

  for (double x: data) {
    sum += (x - mean) * (x - mean);
  }

  sum /= double(data.size());

  return sum;
}

int main(int argc, char* argv[]) {
  std::filesystem::path data_dir = std::filesystem::current_path().parent_path().parent_path().parent_path() / "benchmark-artifact/data";
  std::filesystem::path image_path = data_dir / "depth/000004415622.png";
  std::filesystem::path intrinsics_path = data_dir / "config/cPlusPlus/intrinsics.K";
  std::filesystem::path config_path = data_dir / "config/TUM_fr3_long_val.ini";

  auto                  start_time      = std::chrono::steady_clock::now();
  auto                   end_time       = std::chrono::steady_clock::now();
  double                    time;

  int NUMBER_OF_RUNS = 10;
  int NUMBER_OF_SNAPSHOT = 50;

  Eigen::VectorXd test_duration = Eigen::VectorXd::Zero(NUMBER_OF_SNAPSHOT);

  std::string dataset_path = (argc > 1 ? argv[1] : (data_dir / "depth").string());

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

  Eigen::VectorXi labels;
  int found_planes;

  for (uint i = 0; i < NUMBER_OF_SNAPSHOT; ++i) {
    std::cout << "SNAPSHOT #" << i + 1;

    for (int t = 0; t < NUMBER_OF_RUNS; ++t) {
      start_time = std::chrono::steady_clock::now();

      deplex::utils::DepthImage image(sorted_input_data[i].path().string());
      auto algorithm = deplex::PlaneExtractor(image.getHeight(), image.getWidth(), config);
      labels = algorithm.process(image.toPointCloud(intrinsics));

      end_time = std::chrono::steady_clock::now();

      time = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end_time - start_time).count();
      test_duration[i] += time;
    }

    test_duration[i] /= NUMBER_OF_RUNS;

    found_planes = labels.maxCoeff();
    std::cout << ' ' << found_planes << " planes found" << std::endl;
  }
  deplex::utils::savePointCloudCSV(test_duration.cast<float>().transpose(),
                                   data_dir / ("process_sequence_" + std::to_string(NUMBER_OF_SNAPSHOT)
                                                                                + "_snapshot.csv"));

  double elapsed_time_min = *std::min_element(test_duration.begin(), test_duration.end());
  double elapsed_time_max = *std::max_element(test_duration.begin(), test_duration.end());
  double elapsed_time_mean = std::accumulate(test_duration.begin(), test_duration.end(), 0.0) / NUMBER_OF_SNAPSHOT;

  double dispersion = calculateVariance(test_duration, elapsed_time_mean);
  double standard_deviation = sqrt(dispersion);
  double standard_error = standard_deviation / sqrt(NUMBER_OF_SNAPSHOT);

  // 95% confidence interval
  double lower_bound = elapsed_time_mean - 1.96 * (standard_error);
  double upper_bound = elapsed_time_mean + 1.96 * (standard_error);

  std::cout << "\nDispersion: " << dispersion << std::endl;
  std::cout << "Standard deviation: " << standard_deviation << std::endl;
  std::cout << "Standard error: " << standard_error << std::endl;
  std::cout << "Confidence interval (95%): [" << lower_bound << "; " << upper_bound << "]\n\n";

  std::cout << "Elapsed time (min): " << elapsed_time_min << '\n';
  std::cout << "Elapsed time (max): " << elapsed_time_max << '\n';
  std::cout << "Elapsed time (mean): " << elapsed_time_mean << '\n';
  std::cout << "FPS (max): " << 1e6l / elapsed_time_min << '\n';
  std::cout << "FPS (min): " << 1e6l / elapsed_time_max << '\n';
  std::cout << "FPS (mean): " << 1e6l / elapsed_time_mean << '\n';

  return 0;
}