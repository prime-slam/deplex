/**
 * Copyright (c) 2022, Arthur Saliou, Anastasiia Kornilova
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "deplex/plane_extractor.h"

#include <algorithm>
#include <numeric>
#include <queue>
#if defined(DEBUG_DEPLEX) || defined(BENCHMARK_LOGGING)
#include <fstream>
#include <iostream>
#endif
#ifdef BENCHMARK_LOGGING
#include <chrono>
#endif

#include "cell_grid.h"
#include "normals_histogram.h"

#ifdef BENCHMARK_LOGGING
namespace {
template <typename T, typename Time>
inline size_t get_benchmark_time(Time start_time) {
  return std::chrono::duration_cast<T>(std::chrono::high_resolution_clock::now() - start_time).count();
}
}  // namespace
#endif

namespace deplex {
/**
 * Class with encapsulated PlaneExtractor logic (see PIMPL idiom)
 */
class PlaneExtractor::Impl {
 public:
  /**
   * Impl (PlaneExtractor) constructor.
   *
   * @param image_height Image height in pixels.
   * @param image_width Image width in pixels.
   * @param config Parameters of plane extraction algorithm.
   */
  Impl(int32_t image_height, int32_t image_width, config::Config config);

  /**
   * Extract planes from given image.
   *
   * @param pcd_array Points matrix [Nx3] of ORGANIZED point cloud
   * i.e. points that refer to organized image structure.
   * @returns 1D Array, where i-th value is plane number to which refers i-th point of point cloud.
   * 0-value label refers to non-planar segment.
   */
  Eigen::VectorXi process(Eigen::MatrixX3f const& pcd_array);

 private:
  config::Config config_;
  int32_t nr_horizontal_cells_;
  int32_t nr_vertical_cells_;
  int32_t image_height_;
  int32_t image_width_;
  Eigen::MatrixXi labels_map_;

  /**
   * Organize point cloud, so that points corresponding to one cell lie sequentially in memory.
   *
   * @param unorganized_data Points matrix [Nx3] with default Eigen alignment (RowMajor).
   * @returns Cell-wise organized points (RowMajor).
   */
  void cellContinuousOrganize(Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> const& unorganized_data,
                              Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>* organized_pcd);

  /**
   * Initialize histogram from planar cells of cell grid.
   *
   * @param cell_grid Cell Grid.
   * @returns Histogram of cells' normals.
   */
  NormalsHistogram initializeHistogram(CellGrid const& cell_grid);

  /**
   * Region Growing:
   * 1. Pick dominant cell;
   * 2. Perform growSeed operation;
   * 3. Push to cell segment.
   *
   * @param cell_grid Cell Grid.
   * @param hist Histogram of cells' normals.
   * @returns Vector of grown cell segments.
   */
  std::vector<CellSegment> createPlaneSegments(CellGrid const& cell_grid, NormalsHistogram hist);

  /**
   * Find labels of cells, which can be merged.
   *
   * @param plane_segments Vector of grown cell segments.
   * @returns Vector of merge labels. If merge[i] != i, than cell[i] can be merged with cell[merge[i]].
   */
  std::vector<int32_t> findMergedLabels(std::vector<CellSegment>* plane_segments);

  /**
   * Transform merge label information into 1D label array of image size.
   *
   * @param merge_labels Vector of merge labels.
   * @returns Flatten array of labels of size [image_width x image_height]
   */
  Eigen::VectorXi toImageLabels(std::vector<int32_t> const& merge_labels);

  /**
   * Clean all used data for sufficient sequential image computing.
   */
  void cleanArtifacts();

  /**
   * Seed growing via BFS.
   *
   * @param seed_id Start seed to grow from.
   * @param unassigned Vector of cell id's that don't belong to any cell yet.
   * @param activation_map Vector of activated cells after growSeed call.
   * @param cell_grid Cell Grid.
   */
  void growSeed(Eigen::Index seed_id, std::vector<bool> const& unassigned, std::vector<bool>* activation_map,
                CellGrid const& cell_grid) const;

  /**
   * Get vector of potentially mergeable cell components.
   *
   * @param nr_planes Number of planar components.
   * @returns Connectivity with neighbours map.
   */
  std::vector<std::vector<bool>> getConnectedComponents(size_t nr_planes) const;

#ifdef DEBUG_DEPLEX
  void planarCellsToLabels(std::vector<bool> const& planar_flags, std::string const& save_path);
#endif
};

PlaneExtractor::Impl::Impl(int32_t image_height, int32_t image_width, config::Config config)
    : config_(config),
      nr_horizontal_cells_(image_width / std::max(config.patch_size, 1)),
      nr_vertical_cells_(image_height / std::max(config.patch_size, 1)),
      image_height_(image_height),
      image_width_(image_width),
      labels_map_(Eigen::MatrixXi::Zero(nr_vertical_cells_, nr_horizontal_cells_)) {
  if (config.patch_size == 0) {
    throw std::runtime_error("Error! Invalid config parameter: patchSize(" + std::to_string(config.patch_size) +
                             "). patchSize has to be positive.");
  }
}

PlaneExtractor::~PlaneExtractor() = default;
PlaneExtractor::PlaneExtractor(PlaneExtractor&&) noexcept = default;
PlaneExtractor& PlaneExtractor::operator=(PlaneExtractor&& op) noexcept = default;

PlaneExtractor::PlaneExtractor(int32_t image_height, int32_t image_width, config::Config config)
    : impl_(new Impl(image_height, image_width, config)) {}

Eigen::VectorXi PlaneExtractor::process(Eigen::MatrixX3f const& pcd_array) { return impl_->process(pcd_array); }

Eigen::VectorXi PlaneExtractor::Impl::process(Eigen::MatrixX3f const& pcd_array) {
  if (pcd_array.rows() != image_width_ * image_height_) {
    std::string msg_points_size = std::to_string(pcd_array.rows());
    std::string msg_width = std::to_string(image_width_);
    std::string msg_height = std::to_string(image_height_);
    throw std::runtime_error("Error! Number of points doesn't match image shape: " + msg_points_size +
                             " != " + msg_height + " x " + msg_width);
  }
  // 0. Organize PCD to cell-continuous data
#ifdef BENCHMARK_LOGGING
  auto time_cell_continuous_organize = std::chrono::high_resolution_clock::now();
#endif
  Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> cell_continuous_points(pcd_array.rows(), pcd_array.cols());
  cellContinuousOrganize(pcd_array, &cell_continuous_points);

#ifdef BENCHMARK_LOGGING
  std::clog << "[BenchmarkLogging] Cell Continuous Organize: "
            << get_benchmark_time<decltype(std::chrono::microseconds())>(time_cell_continuous_organize) << '\n';
#endif
  // 1. Initialize cell grid (Planarity estimation)
#ifdef BENCHMARK_LOGGING
  auto time_init_cell_grid = std::chrono::high_resolution_clock::now();
#endif
  CellGrid cell_grid(cell_continuous_points, config_, nr_horizontal_cells_, nr_vertical_cells_);
#ifdef BENCHMARK_LOGGING
  std::clog << "[BenchmarkLogging] Cell Grid Initialization: "
            << get_benchmark_time<decltype(std::chrono::microseconds())>(time_init_cell_grid) << '\n';
#endif
#ifdef DEBUG_DEPLEX
  planarCellsToLabels(cell_grid.getPlanarMask(), "dbg_1_planar_cells.csv");
  std::clog << "[DebugInfo] Planar cell found: "
            << std::count(cell_grid.getPlanarMask().begin(), cell_grid.getPlanarMask().end(), true) << '\n';
#endif
  // 2. Find dominant cell normals
#ifdef BENCHMARK_LOGGING
  auto time_init_histogram = std::chrono::high_resolution_clock::now();
#endif
  NormalsHistogram hist = initializeHistogram(cell_grid);
#ifdef BENCHMARK_LOGGING
  std::clog << "[BenchmarkLogging] Histogram Initialization: "
            << get_benchmark_time<decltype(std::chrono::microseconds())>(time_init_histogram) << '\n';
#endif
  // 3. Region growing
#ifdef BENCHMARK_LOGGING
  auto time_region_growing = std::chrono::high_resolution_clock::now();
#endif
  auto plane_segments = createPlaneSegments(cell_grid, hist);
#ifdef BENCHMARK_LOGGING
  std::clog << "[BenchmarkLogging] Region Growing: "
            << get_benchmark_time<decltype(std::chrono::microseconds())>(time_region_growing) << '\n';
#endif
#ifdef DEBUG_DEPLEX
  std::clog << "[DebugInfo] Plane segments found: " << (plane_segments.empty() ? 0 : plane_segments.size() - 1) << '\n';
#endif
  if (plane_segments.empty()) {
    return Eigen::VectorXi::Zero(pcd_array.rows());
  }
  // 5. Merge planes
#ifdef BENCHMARK_LOGGING
  auto time_merge_planes = std::chrono::high_resolution_clock::now();
#endif
  std::vector<int32_t> merge_labels = findMergedLabels(&plane_segments);
#ifdef BENCHMARK_LOGGING
  std::clog << "[BenchmarkLogging] Merge Planes: "
            << get_benchmark_time<decltype(std::chrono::microseconds())>(time_merge_planes) << '\n';
#endif
#ifdef DEBUG_DEPLEX
  std::vector<int32_t> sorted_labels(merge_labels);
  std::sort(sorted_labels.begin(), sorted_labels.end());

  std::clog << "[DebugInfo] Planes number after merge: "
            << std::distance(sorted_labels.begin(), std::unique(sorted_labels.begin(), sorted_labels.end())) - 1
            << '\n';
#endif
#ifdef BENCHMARK_LOGGING
  auto time_labels_creation = std::chrono::high_resolution_clock::now();
#endif
  Eigen::VectorXi labels = toImageLabels(merge_labels);
#ifdef BENCHMARK_LOGGING
  std::clog << "[BenchmarkLogging] Labels creation: "
            << get_benchmark_time<decltype(std::chrono::microseconds())>(time_labels_creation) << '\n';
#endif
#ifdef DEBUG_DEPLEX
  std::ofstream of("dbg_3_labels.csv");
  of << labels.reshaped<Eigen::RowMajor>(image_height_, image_width_)
            .format(Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ",", "\n"));
#endif
  // 7. Cleanup
  cleanArtifacts();
  return labels;
}

void PlaneExtractor::Impl::cellContinuousOrganize(
    Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> const& unorganized_data,
    Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>* organized_pcd) {
  int32_t cell_width = config_.patch_size;
  int32_t cell_height = config_.patch_size;

#pragma omp parallel for default(none) shared(cell_width, cell_height, organized_pcd, unorganized_data)
  for (Eigen::Index cell_id = 0; cell_id < nr_vertical_cells_ * nr_horizontal_cells_; ++cell_id) {
    Eigen::Index outer_cell_stride = cell_width * cell_height * cell_id;
    for (Eigen::Index i = 0; i < cell_height; ++i) {
      Eigen::Index cell_row_stride = i * cell_width;
      organized_pcd->block(cell_row_stride + outer_cell_stride, 0, cell_width, 3) =
          unorganized_data.block(i * image_width_ + (cell_id / nr_horizontal_cells_ * image_width_ * cell_height) +
                                     (cell_id * cell_width) % image_width_,
                                 0, cell_height, 3);
    }
  }
}

NormalsHistogram PlaneExtractor::Impl::initializeHistogram(CellGrid const& cell_grid) {
  Eigen::MatrixX3f normals = Eigen::MatrixX3f::Zero(cell_grid.size(), 3);
  for (Eigen::Index i = 0; i < cell_grid.size(); ++i) {
    if (cell_grid.getPlanarMask()[i]) {
      normals.row(i) = cell_grid[i].getStat().getNormal();
    }
  }

  int nr_bins_per_coord = config_.histogram_bins_per_coord;
  return NormalsHistogram{nr_bins_per_coord, normals};
}

std::vector<CellSegment> PlaneExtractor::Impl::createPlaneSegments(CellGrid const& cell_grid, NormalsHistogram hist) {
  std::vector<CellSegment> plane_segments;
  std::vector<bool> unassigned_mask(cell_grid.getPlanarMask());
  auto remaining_planar_cells = static_cast<int32_t>(std::count(unassigned_mask.begin(), unassigned_mask.end(), true));

  while (remaining_planar_cells > 0) {
    // 1. Seeding
    std::vector<int32_t> seed_candidates = hist.getPointsFromMostFrequentBin();
    if (seed_candidates.size() < config_.min_region_growing_candidate_size) {
      return plane_segments;
    }
    // 2. Select seed with minimum MSE
    int32_t seed_id;
    double min_mse = INT_MAX;
    for (int32_t seed_candidate : seed_candidates) {
      if (cell_grid[seed_candidate].getStat().getMSE() < min_mse) {
        seed_id = seed_candidate;
        min_mse = cell_grid[seed_candidate].getStat().getMSE();
      }
    }
    // 3. Grow seed
    CellSegment plane_candidate(cell_grid[seed_id]);
    std::vector<bool> activation_map(unassigned_mask.size());
    growSeed(seed_id, unassigned_mask, &activation_map, cell_grid);

    // 4. Merge activated cells & remove from hist
    for (size_t i = 0; i < activation_map.size(); ++i) {
      if (activation_map[i]) {
        plane_candidate += cell_grid[i];
        hist.removePoint(static_cast<int32_t>(i));
        unassigned_mask[i] = false;
        --remaining_planar_cells;
      }
    }
    size_t nr_cells_activated = std::count(activation_map.begin(), activation_map.end(), true);

    if (nr_cells_activated < config_.min_region_growing_cells_activated) {
      continue;
    }

    plane_candidate.calculateStats();

    // 5. Model fitting
    if (plane_candidate.getStat().getScore() > config_.min_region_planarity_score) {
      plane_segments.push_back(plane_candidate);
      auto nr_curr_planes = static_cast<int32_t>(plane_segments.size());
      // Mark cells
      // TODO: Effective assigning by mask?
      int stacked_cell_id = 0;
      for (int32_t row_id = 0; row_id < nr_vertical_cells_; ++row_id) {
        for (int32_t col_id = 0; col_id < nr_horizontal_cells_; ++col_id) {
          if (activation_map[stacked_cell_id]) {
            labels_map_.row(row_id)[col_id] = nr_curr_planes;
          }
          ++stacked_cell_id;
        }
      }
    }
  }

  return plane_segments;
}

void PlaneExtractor::Impl::growSeed(Eigen::Index seed_id, std::vector<bool> const& unassigned,
                                    std::vector<bool>* activation_map, CellGrid const& cell_grid) const {
  if (!unassigned[seed_id] || activation_map->at(seed_id)) {
    return;
  }

  std::queue<Eigen::Index> seed_queue;
  seed_queue.push(seed_id);
  activation_map->at(seed_id) = true;

  while (!seed_queue.empty()) {
    Eigen::Index current_seed = seed_queue.front();
    seed_queue.pop();

    double d_current = cell_grid[current_seed].getStat().getD();
    Eigen::Vector3f normal_current = cell_grid[current_seed].getStat().getNormal();

    for (auto neighbour : cell_grid.getNeighbours(current_seed)) {
      if (!unassigned[neighbour] || activation_map->at(neighbour)) {
        continue;
      }
      Eigen::Vector3f normal_neighbour = cell_grid[neighbour].getStat().getNormal();
      Eigen::Vector3f mean_neighbour = cell_grid[neighbour].getStat().getMean();

      double cos_angle = normal_current.dot(normal_neighbour);
      double merge_dist = pow(normal_current.dot(mean_neighbour) + d_current, 2);
      if (cos_angle >= config_.min_cos_angle_merge && merge_dist <= cell_grid[neighbour].getMergeTolerance()) {
        activation_map->at(neighbour) = true;
        seed_queue.push(static_cast<Eigen::Index>(neighbour));
      }
    }
  }
}

std::vector<int32_t> PlaneExtractor::Impl::findMergedLabels(std::vector<CellSegment>* plane_segments) {
  size_t nr_planes = plane_segments->size();
  // Boolean matrix [nr_planes X nr_planes]
  auto planes_association_mx = getConnectedComponents(nr_planes);
  std::vector<int32_t> plane_merge_labels(nr_planes);
  std::iota(plane_merge_labels.begin(), plane_merge_labels.end(), 0);

  // Connect compatible planes
  for (size_t row_id = 0; row_id < nr_planes; ++row_id) {
    int32_t plane_id = plane_merge_labels[row_id];
    bool plane_expanded = false;
    for (size_t col_id = row_id + 1; col_id != planes_association_mx[row_id].size(); ++col_id) {
      if (planes_association_mx[row_id][col_id]) {
        double cos_angle =
            plane_segments->at(plane_id).getStat().getNormal().dot(plane_segments->at(col_id).getStat().getNormal());
        double distance =
            pow(plane_segments->at(plane_id).getStat().getNormal().dot(plane_segments->at(col_id).getStat().getMean()) +
                    plane_segments->at(plane_id).getStat().getD(),
                2);
        if (cos_angle > config_.min_cos_angle_merge && distance < config_.max_merge_dist) {
          plane_segments->at(plane_id) += plane_segments->at(col_id);
          plane_merge_labels[col_id] = plane_id;
          plane_expanded = true;
        } else {
          planes_association_mx[row_id][col_id] = false;
        }
      }
    }
    if (plane_expanded) plane_segments->at(plane_id).calculateStats();
  }

  return plane_merge_labels;
}

void PlaneExtractor::Impl::cleanArtifacts() { labels_map_.setZero(); }

std::vector<std::vector<bool>> PlaneExtractor::Impl::getConnectedComponents(size_t nr_planes) const {
  std::vector<std::vector<bool>> planes_assoc_matrix(nr_planes, std::vector<bool>(nr_planes, false));

  for (int32_t row_id = 0; row_id < labels_map_.rows() - 1; ++row_id) {
    auto row = labels_map_.row(row_id);
    auto next_row = labels_map_.row(row_id + 1);
    for (int32_t col_id = 0; col_id < labels_map_.cols() - 1; ++col_id) {
      auto plane_id = row[col_id];
      if (plane_id > 0) {
        if (row[col_id + 1] > 0 && plane_id != row[col_id + 1])
          planes_assoc_matrix[plane_id - 1][row[col_id + 1] - 1] = true;
        if (next_row[col_id] > 0 && plane_id != next_row[col_id])
          planes_assoc_matrix[plane_id - 1][next_row[col_id] - 1] = true;
      }
    }
  }
  for (int32_t row_id = 0; row_id < planes_assoc_matrix.size(); ++row_id) {
    for (int32_t col_id = 0; col_id < planes_assoc_matrix.size(); ++col_id) {
      planes_assoc_matrix[row_id][col_id] = planes_assoc_matrix[row_id][col_id] || planes_assoc_matrix[col_id][row_id];
    }
  }

  return planes_assoc_matrix;
}

Eigen::VectorXi PlaneExtractor::Impl::toImageLabels(std::vector<int32_t> const& merge_labels) {
  Eigen::MatrixXi labels(Eigen::MatrixXi::Zero(image_height_, image_width_));

  int32_t cell_width = config_.patch_size;
  int32_t cell_height = config_.patch_size;

  int32_t stacked_cell_id = 0;
  for (auto row = 0; row < labels_map_.rows(); ++row) {
    for (auto col = 0; col < labels_map_.cols(); ++col) {
      auto cell_row = stacked_cell_id / nr_horizontal_cells_;
      auto cell_col = stacked_cell_id % nr_horizontal_cells_;
      // Fill cell with label
      auto label_row = cell_row * cell_height;
      auto label_col = cell_col * cell_width;
      for (auto i = label_row; i < label_row + cell_height; ++i) {
        for (auto j = label_col; j < label_col + cell_width; ++j) {
          auto label = labels_map_.row(row)[col];
          labels.row(i)[j] = (label == 0 ? 0 : merge_labels[label - 1] + 1);
        }
      }
      ++stacked_cell_id;
    }
  }

  return labels.reshaped<Eigen::RowMajor>();
}

#ifdef DEBUG_DEPLEX

template <typename T>
void vectorToCSV(std::vector<std::vector<T>> const& data, std::string const& out_path, char sep = ',') {
  std::ofstream f_out(out_path);
  for (const auto& row : data) {
    for (auto value = row.begin(); value != row.end(); ++value) {
      if (value != row.begin()) {
        f_out << sep;
      }
      f_out << *value;
    }
    f_out << '\n';
  }
}

void PlaneExtractor::Impl::planarCellsToLabels(std::vector<bool> const& planar_flags, std::string const& save_path) {
  std::vector<std::vector<int32_t>> labels(image_height_, std::vector<int32_t>(image_width_, 0));

  int32_t cell_width = config_.patch_size;
  int32_t cell_height = config_.patch_size;

  for (auto cell_id = 0; cell_id < planar_flags.size(); ++cell_id) {
    if (planar_flags[cell_id]) {
      auto cell_row = cell_id / nr_horizontal_cells_;
      auto cell_col = cell_id % nr_horizontal_cells_;
      // Fill cell with label
      auto label_row = cell_row * cell_height;
      auto label_col = cell_col * cell_width;
      for (auto i = label_row; i < label_row + cell_height; ++i) {
        for (auto j = label_col; j < label_col + cell_width; ++j) {
          labels[i][j] = static_cast<int32_t>(cell_id);
        }
      }
    }
  }

  vectorToCSV(labels, save_path);
}
#endif

}  // namespace deplex