#include "deplex/plane_extractor.h"

#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>

#ifdef DEBUG_DEPLEX
#include <fstream>
#include <iostream>
#endif

#include "cell_segment.h"
#include "histogram.h"

#ifndef BITSET_SIZE
#define BITSET_SIZE 65536  // 2^16
#endif

namespace deplex {

typedef uchar label_t;

class PlaneExtractor::Impl {
 public:
  Impl(int32_t image_height, int32_t image_width, config::Config config = kDefaultConfig);

  Eigen::VectorXi process(Eigen::MatrixXf const& pcd_array);

 private:
  config::Config config_;
  int32_t nr_horizontal_cells_;
  int32_t nr_vertical_cells_;
  int32_t nr_total_cells_;
  int32_t nr_pts_per_cell_;
  int32_t image_height_;
  int32_t image_width_;
  std::vector<std::shared_ptr<CellSegment>> cell_grid_;
  cv::Mat_<int32_t> grid_plane_seg_map_;
  cv::Mat_<label_t> grid_plane_seg_map_eroded_;
  std::vector<label_t> seg_map_stacked_;
  void organizeByCell(Eigen::MatrixXf const& pcd_array, Eigen::MatrixXf* out);

  std::bitset<BITSET_SIZE> findPlanarCells(Eigen::MatrixXf const& pcd_array);

  Histogram initializeHistogram(std::bitset<BITSET_SIZE> const& planar_flags);

  std::vector<float> computeCellDistTols(Eigen::MatrixXf const& pcd_array,
                                         std::bitset<BITSET_SIZE> const& planar_flags);

  std::vector<std::shared_ptr<CellSegment>> createPlaneSegments(Histogram hist,
                                                                std::bitset<BITSET_SIZE> const& planar_flags,
                                                                std::vector<float> const& cell_dist_tols);

  std::vector<int32_t> mergePlanes(std::vector<std::shared_ptr<CellSegment>>& plane_segments);

  cv::Mat coarseToLabels(std::vector<int32_t> const& labels);

  void cleanArtifacts();

  void growSeed(int32_t x, int32_t y, int32_t prev_index, std::bitset<BITSET_SIZE> const& unassigned,
                std::bitset<BITSET_SIZE>* activation_map, std::vector<float> const& cell_dist_tols) const;

  std::vector<std::bitset<BITSET_SIZE>> getConnectedComponents(size_t nr_planes) const;

#ifdef DEBUG_DEPLEX
  void planarCellsToLabels(std::bitset<BITSET_SIZE> const& planar_flags, std::string const& save_path);

  void planeSegmentsMapToLabels(std::string const& save_path, cv::Mat_<int32_t> const& cell_map);

  void mergeSegmentsToLabels(std::string const& save_path, std::vector<int32_t> const& merge_labels);
#endif
};

const config::Config PlaneExtractor::kDefaultConfig{{{"patchSize", "12"},
                                                     {"histogramBinsPerCoord", "20"},
                                                     {"minCosAngleForMerge", "0.93"},
                                                     {"maxMergeDist", "500"},
                                                     {"minRegionGrowingCandidateSize", "5"},
                                                     {"minRegionGrowingCellsActivated", "4"},
                                                     {"minRegionPlanarityScore", "50"},
                                                     {"doRefinement", "true"},
                                                     {"refinementMultiplierCoeff", "15"},
                                                     {"depthSigmaCoeff", "1.425e-6"},
                                                     {"depthSigmaMargin", "10"},
                                                     {"minPtsPerCell", "3"},
                                                     {"depthDiscontinuityThreshold", "160"},
                                                     {"maxNumberDepthDiscontinuity", "1"}}};

PlaneExtractor::Impl::Impl(int32_t image_height, int32_t image_width, config::Config config)
    : config_(config),
      nr_horizontal_cells_(image_width / config.getInt("patchSize")),
      nr_vertical_cells_(image_height / config.getInt("patchSize")),
      nr_total_cells_(nr_horizontal_cells_ * nr_vertical_cells_),
      nr_pts_per_cell_(pow(config.getInt("patchSize"), 2)),
      image_height_(image_height),
      image_width_(image_width),
      cell_grid_(nr_total_cells_, nullptr),
      grid_plane_seg_map_(nr_vertical_cells_, nr_horizontal_cells_, 0),
      grid_plane_seg_map_eroded_(nr_vertical_cells_, nr_horizontal_cells_, uchar(0)),
      seg_map_stacked_(image_height * image_width, 0) {}

PlaneExtractor::~PlaneExtractor() = default;
PlaneExtractor::PlaneExtractor(PlaneExtractor&&) noexcept = default;
PlaneExtractor& PlaneExtractor::operator=(PlaneExtractor&& op) noexcept = default;

PlaneExtractor::PlaneExtractor(int32_t image_height, int32_t image_width, config::Config config)
    : impl_(new Impl(image_height, image_width, config)) {}

Eigen::VectorXi PlaneExtractor::process(Eigen::MatrixXf const& pcd_array) { return impl_->process(pcd_array); }

Eigen::VectorXi PlaneExtractor::Impl::process(Eigen::MatrixXf const& pcd_array) {
  // 0. Stack array by cell
  Eigen::MatrixXf organized_array(pcd_array.rows(), pcd_array.cols());
  organizeByCell(pcd_array, &organized_array);
  // 1. Planar cell fitting
  std::bitset<BITSET_SIZE> planar_flags = findPlanarCells(organized_array);
#ifdef DEBUG_DEPLEX
  planarCellsToLabels(planar_flags, "dbg_1_planar_cells.csv");
  std::clog << "[DebugInfo] Planar cell found: " << planar_flags.count() << '\n';
#endif
  // 2. Histogram initialization
  Histogram hist = initializeHistogram(planar_flags);
  // 3. Compute cell dist tols
  std::vector<float> cell_dist_tols = computeCellDistTols(organized_array, planar_flags);
  // 4. Region growing
  auto plane_segments = createPlaneSegments(hist, planar_flags, cell_dist_tols);
#ifdef DEBUG_DEPLEX
  planeSegmentsMapToLabels("dbg_2_plane_segments_raw.csv", grid_plane_seg_map_);
  std::clog << "[DebugInfo] Plane segments found: " << plane_segments.size()
            << '\n';
#endif
  // 5. Merge planes
  std::vector<int32_t> merge_labels = mergePlanes(plane_segments);
#ifdef DEBUG_DEPLEX
  mergeSegmentsToLabels("dbg_3_plane_segments_merged.csv", merge_labels);
  std::vector<int32_t> sorted_labels(merge_labels);
  std::sort(sorted_labels.begin(), sorted_labels.end());

  std::clog << "[DebugInfo] Planes number after merge: "
            << std::distance(sorted_labels.begin(), std::unique(sorted_labels.begin(), sorted_labels.end())) << '\n';
#endif
  cv::Mat_ labels(image_height_, image_width_, 0);
  labels = coarseToLabels(merge_labels);
#ifdef DEBUG_DEPLEX
  std::ofstream of("dbg_4_labels.csv");
  of << cv::format(labels, cv::Formatter::FMT_CSV);
#endif
  // 7. Cleanup
  cleanArtifacts();
  Eigen::MatrixXi eigen_labels;
  cv::cv2eigen(labels, eigen_labels);
  return eigen_labels.reshaped<Eigen::RowMajor>();
}

void PlaneExtractor::Impl::organizeByCell(Eigen::MatrixXf const& pcd_array, Eigen::MatrixXf* out) {
  int32_t patch_size = config_.getInt("patchSize");
  int32_t mxn = image_width_ * image_height_;
  int32_t mxn2 = 2 * mxn;

  int stacked_id = 0;
  for (int r = 0; r < image_height_; r++) {
    int cell_r = r / patch_size;
    int local_r = r % patch_size;
    for (int c = 0; c < image_width_; c++) {
      int cell_c = c / patch_size;
      int local_c = c % patch_size;
      auto shift = (cell_r * nr_horizontal_cells_ + cell_c) * patch_size * patch_size + local_r * patch_size + local_c;

      *(out->data() + shift) = *(pcd_array.data() + stacked_id);
      *(out->data() + mxn + shift) = *(pcd_array.data() + mxn + stacked_id);
      *(out->data() + mxn2 + shift) = *(pcd_array.data() + mxn2 + stacked_id);
      stacked_id++;
    }
  }
}

std::bitset<BITSET_SIZE> PlaneExtractor::Impl::findPlanarCells(Eigen::MatrixXf const& pcd_array) {
  std::bitset<BITSET_SIZE> planar_flags;
  int32_t cell_width = config_.getInt("patchSize");
  int32_t cell_height = config_.getInt("patchSize");
  int32_t stacked_cell_id = 0;
  for (Eigen::Index cell_r = 0; cell_r < nr_vertical_cells_; ++cell_r) {
    for (Eigen::Index cell_h = 0; cell_h < nr_horizontal_cells_; ++cell_h) {
      cell_grid_[stacked_cell_id] =
          std::make_shared<CellSegment>(stacked_cell_id, cell_width, cell_height, pcd_array, config_);
      planar_flags[stacked_cell_id] = cell_grid_[stacked_cell_id]->isPlanar();
      ++stacked_cell_id;
    }
  }
  return planar_flags;
}

Histogram PlaneExtractor::Impl::initializeHistogram(std::bitset<BITSET_SIZE> const& planar_flags) {
  Eigen::MatrixXd spherical_coord(nr_total_cells_, 2);
  for (size_t cell_id = planar_flags._Find_first(); cell_id != planar_flags.size();
       cell_id = planar_flags._Find_next(cell_id)) {
    Eigen::Vector3f cell_normal = cell_grid_[cell_id]->getStat().getNormal();
    double n_proj_norm = sqrt(cell_normal[0] * cell_normal[0] + cell_normal[1] * cell_normal[1]);
    spherical_coord(cell_id, 0) = acos(-cell_normal[2]);
    spherical_coord(cell_id, 1) = atan2(cell_normal[0] / n_proj_norm, cell_normal[1] / n_proj_norm);
  }
  int nr_bins_per_coord = config_.getInt("histogramBinsPerCoord");
  return Histogram{nr_bins_per_coord, spherical_coord, planar_flags};
}

std::vector<float> PlaneExtractor::Impl::computeCellDistTols(Eigen::MatrixXf const& pcd_array,
                                                             std::bitset<BITSET_SIZE> const& planar_flags) {
  std::vector<float> cell_dist_tols(nr_total_cells_, 0);
  double cos_angle_for_merge = config_.getFloat("minCosAngleForMerge");
  float sin_angle_for_merge = sqrt(1 - pow(cos_angle_for_merge, 2));
  // TODO: Put "minMergeDist" to config
  float min_merge_dist = 20.0f;
  float max_merge_dist = config_.getFloat("maxMergeDist");

  for (size_t cell_id = planar_flags._Find_first(); cell_id != planar_flags.size();
       cell_id = planar_flags._Find_next(cell_id)) {
    float cell_diameter = (pcd_array.block(cell_id * nr_pts_per_cell_ + nr_pts_per_cell_ - 1, 0, 1, 3) -
                           pcd_array.block(cell_id * nr_pts_per_cell_, 0, 1, 3))
                              .norm();
    float truncated_distance = std::min(std::max(cell_diameter * sin_angle_for_merge, min_merge_dist), max_merge_dist);
    cell_dist_tols[cell_id] = powf(truncated_distance, 2);
  }

  return cell_dist_tols;
}

std::vector<std::shared_ptr<CellSegment>> PlaneExtractor::Impl::createPlaneSegments(
    Histogram hist, std::bitset<BITSET_SIZE> const& planar_flags, std::vector<float> const& cell_dist_tols) {
  std::vector<std::shared_ptr<CellSegment>> plane_segments;
  std::bitset<BITSET_SIZE> unassigned_mask(planar_flags);
  auto remaining_planar_cells = static_cast<int32_t>(planar_flags.count());

  while (remaining_planar_cells > 0) {
    // 1. Seeding
    std::vector<int32_t> seed_candidates = hist.getPointsFromMostFrequentBin();
    if (seed_candidates.size() < config_.getInt("minRegionGrowingCandidateSize")) {
      return plane_segments;
    }
    // 2. Select seed with minimum MSE
    int32_t seed_id;
    double min_mse = INT_MAX;
    for (int32_t seed_candidate : seed_candidates) {
      if (cell_grid_[seed_candidate]->getStat().getMSE() < min_mse) {
        seed_id = seed_candidate;
        min_mse = cell_grid_[seed_candidate]->getStat().getMSE();
      }
    }
    // 3. Grow seed
    std::shared_ptr<CellSegment> new_segment = cell_grid_[seed_id];
    int32_t y = seed_id / nr_horizontal_cells_;
    int32_t x = seed_id % nr_horizontal_cells_;
    std::bitset<BITSET_SIZE> activation_map;
    growSeed(x, y, seed_id, unassigned_mask, &activation_map, cell_dist_tols);
    // 4. Merge activated cells & remove from hist
    for (size_t i = activation_map._Find_first(); i != activation_map.size(); i = activation_map._Find_next(i)) {
      *new_segment += *cell_grid_[i];
      hist.removePoint(static_cast<int32_t>(i));
      --remaining_planar_cells;
    }
    unassigned_mask &= (~activation_map);
    size_t nr_cells_activated = activation_map.count();

    if (nr_cells_activated < config_.getInt("minRegionGrowingCellsActivated")) {
      continue;
    }

    new_segment->calculateStats();

    // 5. Model fitting
    if (new_segment->getStat().getScore() > config_.getFloat("minRegionPlanarityScore")) {
      plane_segments.push_back(new_segment);
      auto nr_curr_planes = static_cast<int32_t>(plane_segments.size());
      // Mark cells
      int stacked_cell_id = 0;
      for (int32_t row_id = 0; row_id < nr_vertical_cells_; ++row_id) {
        auto row = grid_plane_seg_map_.ptr<int32_t>(row_id);
        for (int32_t col_id = 0; col_id < nr_horizontal_cells_; ++col_id) {
          if (activation_map[stacked_cell_id]) {
            row[col_id] = nr_curr_planes;
          }
          ++stacked_cell_id;
        }
      }
    }
  }

  return plane_segments;
}

std::vector<int32_t> PlaneExtractor::Impl::mergePlanes(std::vector<std::shared_ptr<CellSegment>>& plane_segments) {
  size_t nr_planes = plane_segments.size();
  // Boolean matrix [nr_planes X nr_planes]
  auto planes_association_mx = getConnectedComponents(nr_planes);
  std::vector<int32_t> plane_merge_labels(nr_planes);
  std::iota(plane_merge_labels.begin(), plane_merge_labels.end(), 0);

  // Connect compatible planes
  for (size_t row_id = 0; row_id < nr_planes; ++row_id) {
    int32_t plane_id = plane_merge_labels[row_id];
    bool plane_expanded = false;
    for (size_t col_id = planes_association_mx[row_id]._Find_next(row_id);
         col_id != planes_association_mx[row_id].size(); col_id = planes_association_mx[row_id]._Find_next(col_id)) {
      double cos_angle =
          plane_segments[plane_id]->getStat().getNormal().dot(plane_segments[col_id]->getStat().getNormal());
      double distance =
          pow(plane_segments[plane_id]->getStat().getNormal().dot(plane_segments[col_id]->getStat().getMean()) +
                  plane_segments[plane_id]->getStat().getD(),
              2);
      if (cos_angle > config_.getFloat("minCosAngleForMerge") && distance < config_.getFloat("maxMergeDist")) {
        (*plane_segments[plane_id]) += (*plane_segments[col_id]);
        plane_merge_labels[col_id] = plane_id;
        plane_expanded = true;
      } else {
        planes_association_mx[row_id][col_id] = false;
      }
    }
    if (plane_expanded) plane_segments[plane_id]->calculateStats();
  }

  return plane_merge_labels;
}

cv::Mat PlaneExtractor::Impl::coarseToLabels(std::vector<int32_t> const& labels) {
  cv::Mat_<int32_t> grid_plane_seg_map_merged;
  grid_plane_seg_map_.copyTo(grid_plane_seg_map_merged);

  for (int32_t i = 0; i < labels.size(); ++i) {
    if (labels[i] != i) {
      grid_plane_seg_map_merged.setTo(labels[i], grid_plane_seg_map_merged == i);
    }
  }

  return grid_plane_seg_map_merged;
}

void PlaneExtractor::Impl::cleanArtifacts() {
  cell_grid_.resize(nr_total_cells_, nullptr);
  grid_plane_seg_map_ = 0;
  grid_plane_seg_map_eroded_ = 0;
  seg_map_stacked_.resize(image_height_ * image_width_, 0);
}

void PlaneExtractor::Impl::growSeed(int32_t x, int32_t y, int32_t prev_index,
                                    std::bitset<BITSET_SIZE> const& unassigned,
                                    std::bitset<BITSET_SIZE>* activation_map,
                                    std::vector<float> const& cell_dist_tols) const {
  int32_t index = x + nr_horizontal_cells_ * y;
  if (index >= nr_total_cells_) throw std::out_of_range("growSeed: Index out of total cell number");
  if (!unassigned[index] || (*activation_map)[index]) {
    return;
  }

  double d_1 = cell_grid_[prev_index]->getStat().getD();
  Eigen::Vector3f normal_1 = cell_grid_[prev_index]->getStat().getNormal();
  Eigen::Vector3f normal_2 = cell_grid_[index]->getStat().getNormal();
  Eigen::Vector3f mean_2 = cell_grid_[index]->getStat().getMean();

  double cos_angle = normal_1.dot(normal_2);
  double merge_dist = pow(normal_1.dot(mean_2) + d_1, 2);
  if (cos_angle < config_.getFloat("minCosAngleForMerge") || merge_dist > cell_dist_tols[index]) {
    return;
  }

  activation_map->set(index);
  if (x > 0) growSeed(x - 1, y, index, unassigned, activation_map, cell_dist_tols);
  if (x < nr_horizontal_cells_ - 1) growSeed(x + 1, y, index, unassigned, activation_map, cell_dist_tols);
  if (y > 0)
    growSeed(x, y - 1, index, unassigned, activation_map, cell_dist_tols);
  if (y < nr_vertical_cells_ - 1)
    growSeed(x, y + 1, index, unassigned, activation_map, cell_dist_tols);
}

std::vector<std::bitset<BITSET_SIZE>> PlaneExtractor::Impl::getConnectedComponents(size_t nr_planes) const {
  std::vector<std::bitset<BITSET_SIZE>> planes_assoc_matrix(nr_planes);

  for (int32_t row_id = 0; row_id < grid_plane_seg_map_.rows - 1; ++row_id) {
    auto row = grid_plane_seg_map_.ptr<int>(row_id);
    auto next_row = grid_plane_seg_map_.ptr<int>(row_id + 1);
    for (int32_t col_id = 0; col_id < grid_plane_seg_map_.cols - 1; ++col_id) {
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
      planes_assoc_matrix[row_id][col_id] =
          planes_assoc_matrix[row_id][col_id] ||
          planes_assoc_matrix[col_id][row_id];
    }
  }

  return planes_assoc_matrix;
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

void PlaneExtractor::Impl::planarCellsToLabels(std::bitset<BITSET_SIZE> const& planar_flags,
                                               std::string const& save_path) {
  std::vector<std::vector<int32_t>> labels(image_height_, std::vector<int32_t>(image_width_, 0));

  int32_t cell_width = config_.getInt("patchSize");
  int32_t cell_height = config_.getInt("patchSize");

  for (auto cell_id = planar_flags._Find_first(); cell_id != planar_flags.size();
       cell_id = planar_flags._Find_next(cell_id)) {
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

  vectorToCSV(labels, save_path);
}

void PlaneExtractor::Impl::planeSegmentsMapToLabels(std::string const& save_path, cv::Mat_<int32_t> const& cell_map) {
  std::vector<std::vector<int32_t>> labels(image_height_, std::vector<int32_t>(image_width_, 0));

  int32_t cell_width = config_.getInt("patchSize");
  int32_t cell_height = config_.getInt("patchSize");

  int32_t stacked_cell_id = 0;
  for (auto row = 0; row < cell_map.rows; ++row) {
    for (auto col = 0; col < cell_map.cols; ++col) {
      auto cell_row = stacked_cell_id / nr_horizontal_cells_;
      auto cell_col = stacked_cell_id % nr_horizontal_cells_;
      // Fill cell with label
      auto label_row = cell_row * cell_height;
      auto label_col = cell_col * cell_width;
      for (auto i = label_row; i < label_row + cell_height; ++i) {
        for (auto j = label_col; j < label_col + cell_width; ++j) {
          labels[i][j] = cell_map[row][col];
        }
      }
      ++stacked_cell_id;
    }
  }

  vectorToCSV(labels, save_path);
}

void PlaneExtractor::Impl::mergeSegmentsToLabels(std::string const& save_path,
                                                 std::vector<int32_t> const& merge_labels) {
  cv::Mat_<int32_t> grid_plane_seg_map_merged;
  grid_plane_seg_map_.copyTo(grid_plane_seg_map_merged);

  for (int32_t i = 0; i < merge_labels.size(); ++i) {
    if (merge_labels[i] != i) {
      grid_plane_seg_map_merged.setTo(merge_labels[i], grid_plane_seg_map_merged == i);
    }
  }

  planeSegmentsMapToLabels(save_path, grid_plane_seg_map_merged);
}
#endif

}  // namespace deplex