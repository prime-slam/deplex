#include <deplex/plane_extractor.h>
#include <deplex/utils/depth_image.h>
#include <deplex/utils/eigen_io.h>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <numeric>

double variance(const Eigen::VectorXd& data, double mean) {
  double sum = 0;

  for (auto x : data) {
    sum += (x - mean) * (x - mean);
  }

  sum /= static_cast<double>(data.size());
  return sum;
}

int main(int argc, char* argv[]) {
  std::filesystem::path data_dir =
      std::filesystem::current_path().parent_path().parent_path().parent_path() / "benchmark/data";
  std::filesystem::path image_path = data_dir / "depth/000004415622.png";
  std::filesystem::path intrinsics_path = data_dir / "config/intrinsics.K";
  std::filesystem::path config_path = data_dir / "config/TUM_fr3_long_val.ini";

  auto start_time = std::chrono::high_resolution_clock::now();
  auto end_time = std::chrono::high_resolution_clock::now();

  const int NUMBER_OF_RUNS = 10;
  const int NUMBER_OF_SNAPSHOT = 1;

  std::vector<Eigen::Vector3d> execution_time_stage(NUMBER_OF_SNAPSHOT, Eigen::VectorXd::Zero(3));

  std::string dataset_path = (argc > 1 ? argv[1] : (data_dir / "depth").string());

  deplex::config::Config config = deplex::config::Config(config_path.string());
  Eigen::Matrix3f intrinsics(deplex::utils::readIntrinsics(intrinsics_path.string()));
  deplex::utils::DepthImage image(image_path.string());

  // Sort data entries
  std::vector<std::filesystem::directory_entry> sorted_input_data;
  for (auto const& entry : std::filesystem::directory_iterator(dataset_path)) {
    if (entry.path().extension() == ".png") {
      sorted_input_data.push_back(entry);
    }
  }
  sort(sorted_input_data.begin(), sorted_input_data.end());

  Eigen::VectorXi labels;

  deplex::PlaneExtractor algorithm(image.getHeight(), image.getWidth(), config);
  std::cout << "Image Height: " << image.getHeight() << " Image Width: " << image.getWidth() << "\n\n";

  for (int t = 0; t < NUMBER_OF_RUNS; ++t) {
    std::cout << "LAUNCH #" << t + 1 << std::endl;

    for (Eigen::Index i = 0; i < NUMBER_OF_SNAPSHOT; ++i) {
      start_time = std::chrono::high_resolution_clock::now();
      image.reset(sorted_input_data[i].path().string());
      end_time = std::chrono::high_resolution_clock::now();
      execution_time_stage[i][0] +=
          static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count());

      start_time = std::chrono::high_resolution_clock::now();
      auto pcd_array = image.toPointCloud(intrinsics);
      end_time = std::chrono::high_resolution_clock::now();
      execution_time_stage[i][1] +=
          static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count());

      start_time = std::chrono::high_resolution_clock::now();
      labels = algorithm.process(pcd_array);
      end_time = std::chrono::high_resolution_clock::now();
      execution_time_stage[i][2] +=
          static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count());

      std::cout << "Snapshot #" << i + 1 << " Planes found: " << labels.maxCoeff() << std::endl;
    }
  }

  auto execution_time_segmentation_stage = algorithm.GetExecutionTime();

  for (auto& v : execution_time_segmentation_stage) {
    for (auto& stage : v) {
      stage /= NUMBER_OF_RUNS;
    }
  }

  for (auto& v : execution_time_stage) {
    for (auto& stage : v) {
      stage /= NUMBER_OF_RUNS;
    }
  }

  Eigen::VectorXd elements = Eigen::VectorXd::Zero(NUMBER_OF_SNAPSHOT);
  for (auto i = 0; i < NUMBER_OF_SNAPSHOT; ++i) {
    elements[i] = execution_time_stage[i][0];
  }

  deplex::utils::savePointCloudCSV(elements.cast<float>().transpose(),
                                   (data_dir / ("process_sequence_stage_read_image.csv")).string());

  for (auto i = 0; i < NUMBER_OF_SNAPSHOT; ++i) {
    elements[i] = execution_time_stage[i][1];
  }

  deplex::utils::savePointCloudCSV(elements.cast<float>().transpose(),
                                   (data_dir / ("process_sequence_stage_translate_image.csv")).string());

  for (auto i = 0; i < NUMBER_OF_SNAPSHOT; ++i) {
    elements[i] = execution_time_stage[i][2];
  }

  deplex::utils::savePointCloudCSV(elements.cast<float>().transpose(),
                                   (data_dir / ("process_sequence_stage_segmentation.csv")).string());

  for (auto i = 0; i < NUMBER_OF_SNAPSHOT; ++i) {
    elements[i] = execution_time_segmentation_stage[i][0];
  }

  deplex::utils::savePointCloudCSV(elements.cast<float>().transpose(),
                                   (data_dir / ("process_sequence_stage_segmentation_cell_grid.csv")).string());

  for (auto i = 0; i < NUMBER_OF_SNAPSHOT; ++i) {
    elements[i] = execution_time_segmentation_stage[i][1];
  }

  deplex::utils::savePointCloudCSV(elements.cast<float>().transpose(),
                                   (data_dir / ("process_sequence_stage_segmentation_region_growing.csv")).string());

  for (auto i = 0; i < NUMBER_OF_SNAPSHOT; ++i) {
    elements[i] = execution_time_segmentation_stage[i][2];
  }

  deplex::utils::savePointCloudCSV(elements.cast<float>().transpose(),
                                   (data_dir / ("process_sequence_stage_segmentation_merge_planes.csv")).string());

  for (auto i = 0; i < NUMBER_OF_SNAPSHOT; ++i) {
    elements[i] = execution_time_segmentation_stage[i][3];
  }

  deplex::utils::savePointCloudCSV(elements.cast<float>().transpose(),
                                   (data_dir / ("process_sequence_stage_segmentation_labels.csv")).string());

  Eigen::VectorXd total_time = Eigen::VectorXd::Zero(NUMBER_OF_SNAPSHOT);

  for (auto i = 0; i < NUMBER_OF_SNAPSHOT; ++i) {
    total_time[i] = execution_time_stage[i][0] + execution_time_stage[i][1] + execution_time_stage[i][2];
  }

  deplex::utils::savePointCloudCSV(total_time.cast<float>().transpose(),
                                   (data_dir / ("process_sequence_total_time.csv")).string());

  double elapsed_time_min = *std::min_element(total_time.begin(), total_time.end());
  double elapsed_time_max = *std::max_element(total_time.begin(), total_time.end());
  double elapsed_time_mean = std::accumulate(total_time.begin(), total_time.end(), 0.0) / NUMBER_OF_SNAPSHOT;

  double dispersion = variance(total_time, elapsed_time_mean);
  double standard_deviation = sqrt(dispersion);
  double standard_error = standard_deviation / sqrt(NUMBER_OF_SNAPSHOT);

  // 95% confidence interval
  const float t_value = 1.96;
  double lower_bound = elapsed_time_mean - t_value * standard_error;
  double upper_bound = elapsed_time_mean + t_value * standard_error;

  std::cout << "\nDispersion: " << dispersion << '\n';
  std::cout << "Standard deviation: " << standard_deviation << '\n';
  std::cout << "Standard error: " << standard_error << '\n';
  std::cout << "Confidence interval (95%): [" << lower_bound << "; " << upper_bound << "]\n\n";

  std::cout << "Elapsed time (ms.) (min): " << elapsed_time_min << '\n';
  std::cout << "Elapsed time (ms.) (max): " << elapsed_time_max << '\n';
  std::cout << "Elapsed time (ms.) (mean): " << elapsed_time_mean << '\n';
  std::cout << "FPS (max): " << 1000 / elapsed_time_min << '\n';
  std::cout << "FPS (min): " << 1000 / elapsed_time_max << '\n';
  std::cout << "FPS (mean): " << 1000 / elapsed_time_mean << '\n';

  return 0;
}