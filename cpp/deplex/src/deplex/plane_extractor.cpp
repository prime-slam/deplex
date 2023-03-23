/**
 * Copyright 2022 prime-slam
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
#include <set>
#include <unordered_map>

#ifdef DEBUG_DEPLEX
#include <fstream>
#include <iostream>
#endif

#include <queue>

#include "cell_grid.h"
#include "normals_histogram.h"

namespace deplex {

class PlaneExtractor::Impl {
 public:
  Impl(int32_t image_height, int32_t image_width, config::Config config = kDefaultConfig);

  Eigen::VectorXi process(Eigen::MatrixXf const& pcd_array);

 private:
  config::Config config_;
  int32_t nr_horizontal_cells_;
  int32_t nr_vertical_cells_;
  int32_t image_height_;
  int32_t image_width_;
  Eigen::MatrixXi labels_map_;
  void organizeByCell(Eigen::MatrixXf const& pcd_array, Eigen::MatrixXf* out);

  NormalsHistogram initializeHistogram(CellGrid const& cell_grid);

  std::set<size_t> createPlaneSegments(CellGrid* cell_grid, NormalsHistogram* hist);

  void mergePlanes(std::set<size_t>* plane_labels, CellGrid* cell_Grid);

  std::unordered_map<size_t, std::set<size_t>> findPlaneNeighbours(std::set<size_t> const& plane_labels,
                                                                   CellGrid* cell_grid);

  Eigen::VectorXi toImageLabels(std::set<size_t> const& plane_labels, CellGrid* cell_grid);

  void cleanArtifacts();

  void growSeed(Eigen::Index seed_id, std::vector<bool> const& unassigned, std::vector<bool>* activation_map,
                CellGrid const& cell_grid) const;

#ifdef DEBUG_DEPLEX
  void planarCellsToLabels(std::vector<bool> const& planar_flags, std::string const& save_path);
#endif
};

const config::Config PlaneExtractor::kDefaultConfig{{{"patchSize", "12"},
                                                     {"histogramBinsPerCoord", "20"},
                                                     {"minCosAngleForMerge", "0.93"},
                                                     {"maxMergeDist", "500"},
                                                     {"minRegionGrowingCandidateSize", "5"},
                                                     {"minRegionGrowingCellsActivated", "4"},
                                                     {"minRegionPlanarityScore", "0.55"},
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
      image_height_(image_height),
      image_width_(image_width),
      labels_map_(Eigen::MatrixXi::Zero(nr_vertical_cells_, nr_horizontal_cells_)) {}

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
  // 1. Initialize cell grid (Planarity estimation)
  CellGrid cell_grid(organized_array, config_, nr_horizontal_cells_, nr_vertical_cells_);
  // 2. Find dominant cell normals
  NormalsHistogram hist = initializeHistogram(cell_grid);
  // 3. Region growing
  auto plane_labels = createPlaneSegments(&cell_grid, &hist);
  if (plane_labels.empty()) {
    return Eigen::VectorXi::Zero(pcd_array.rows());
  }
#ifdef DEBUG_DEPLEX
  std::clog << "Plane found: " << plane_labels.size() << '\n';
#endif
  // 4. Merge planes
  mergePlanes(&plane_labels, &cell_grid);
  // 5. Get labels map
  Eigen::VectorXi labels = toImageLabels(plane_labels, &cell_grid);
#ifdef DEBUG_DEPLEX
  std::ofstream of("dbg_labels.csv");
  of << labels.reshaped<Eigen::RowMajor>(image_height_, image_width_)
            .format(Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ",", "\n"));
#endif
  // 7. Cleanup
  cleanArtifacts();
  return labels;
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

NormalsHistogram PlaneExtractor::Impl::initializeHistogram(CellGrid const& cell_grid) {
  Eigen::MatrixXf normals = Eigen::MatrixXf::Zero(cell_grid.size(), 3);
  for (Eigen::Index i = 0; i < cell_grid.size(); ++i) {
    if (cell_grid.getPlanarMask()[i]) {
      normals.row(i) = cell_grid[i].getStat().getNormal();
    }
  }

  int nr_bins_per_coord = config_.getInt("histogramBinsPerCoord");
  return NormalsHistogram{nr_bins_per_coord, normals};
}

std::set<size_t> PlaneExtractor::Impl::createPlaneSegments(CellGrid* cell_grid, NormalsHistogram* hist) {
  std::set<size_t> plane_labels;
  std::vector<bool> unassigned_mask(cell_grid->getPlanarMask());
  auto remaining_planar_cells = static_cast<int32_t>(std::count(unassigned_mask.begin(), unassigned_mask.end(), true));

  while (remaining_planar_cells > 0) {
    // 1. Seeding
    std::vector<int32_t> seed_candidates = hist->getPointsFromMostFrequentBin();
    if (seed_candidates.size() < config_.getInt("minRegionGrowingCandidateSize")) {
      return plane_labels;
    }
    // 2. Select seed with minimum MSE
    int32_t seed_id;
    double min_mse = INT_MAX;
    for (int32_t seed_candidate : seed_candidates) {
      if ((*cell_grid)[seed_candidate].getStat().getMSE() < min_mse) {
        seed_id = seed_candidate;
        min_mse = (*cell_grid)[seed_candidate].getStat().getMSE();
      }
    }
    // 3. Grow seed
    CellSegment plane_candidate((*cell_grid)[seed_id]);
    std::vector<bool> activation_map(unassigned_mask.size());
    growSeed(seed_id, unassigned_mask, &activation_map, *cell_grid);

    // 4. Merge activated cells & remove from hist
    for (size_t i = 0; i < activation_map.size(); ++i) {
      if (activation_map[i]) {
        plane_candidate += (*cell_grid)[i];
        hist->removePoint(static_cast<int32_t>(i));
        unassigned_mask[i] = false;
        --remaining_planar_cells;
      }
    }
    size_t nr_cells_activated = std::count(activation_map.begin(), activation_map.end(), true);

    if (nr_cells_activated < config_.getInt("minRegionGrowingCellsActivated")) {
      continue;
    }

    plane_candidate.calculateStats();

    // 5. Model fitting
    if (plane_candidate.getStat().getScore() > config_.getFloat("minRegionPlanarityScore")) {
      plane_labels.insert(seed_id);
      // Mark cells
      for (size_t i = 0; i < activation_map.size(); ++i) {
        if (activation_map[i]) {
          cell_grid->uniteLabels(i, seed_id);
        }
      }
      cell_grid->updateCell(seed_id, std::move(plane_candidate));
    }
  }

  return plane_labels;
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
      if (cos_angle >= config_.getFloat("minCosAngleForMerge") &&
          merge_dist <= cell_grid[neighbour].getMergeTolerance()) {
        activation_map->at(neighbour) = true;
        seed_queue.push(static_cast<Eigen::Index>(neighbour));
      }
    }
  }
}

void PlaneExtractor::Impl::mergePlanes(std::set<size_t>* plane_labels, CellGrid* cell_grid) {
  // 1. Find neighbours
  // Map<label, Set<label>> - set of neighbouring labels (planes)
  auto neighbour_labels = findPlaneNeighbours(*plane_labels, cell_grid);
  // 2. Initialize min-MSE priority queue
  auto mse_cmp = [&cell_grid](size_t a, size_t b) {
    return (*cell_grid)[cell_grid->findLabel(a)].getStat().getMSE() <
           (*cell_grid)[cell_grid->findLabel(b)].getStat().getMSE();
  };
  std::set<size_t, decltype(mse_cmp)> min_mse_queue(mse_cmp);
  for (auto label : *plane_labels) {
    min_mse_queue.insert(label);
  }

  while (!min_mse_queue.empty()) {
    auto min_plane_label = *min_mse_queue.begin();
    min_mse_queue.erase(min_mse_queue.begin());
    auto min_mse = std::numeric_limits<double>::max();
    CellSegment min_mse_candidate;
    size_t min_mse_candidate_label = min_plane_label;
    // Find neighbour : merge(neighbour, min_plane).MSE() -> min
    // TODO: Or merge with all neighbours?
    for (auto neighbour : neighbour_labels[min_plane_label]) {
      CellSegment merge_candidate = (*cell_grid)[min_plane_label];
      merge_candidate += (*cell_grid)[neighbour];
      merge_candidate.calculateStats();
      auto candidate_mse = merge_candidate.getStat().getMSE();
      auto candidate_score = merge_candidate.getStat().getScore();
      // TODO: Replace Const * minRegionPlanarityScore with mergeScore parameter
      if (candidate_mse < min_mse && candidate_score < 1.4 * config_.getFloat("minRegionPlanarityScore")) {
        min_mse = candidate_mse;
        min_mse_candidate = std::move(merge_candidate);
        min_mse_candidate_label = neighbour;
      }
    }
    if (min_mse_candidate_label != min_plane_label) {
      cell_grid->uniteLabels(min_mse_candidate_label, min_plane_label);
      // If plane(a) merged with plane(b), then neighbours(a) = neighbours(b) \ a
      // TODO: (?) Assuming isNeighbour(x, z) <- isNeighbour(x, y) && isNeighbour(y, z)
      for (auto const& candidate_neighbour : neighbour_labels[min_mse_candidate_label]) {
        neighbour_labels[min_plane_label].insert(cell_grid->findLabel(candidate_neighbour));
      }
      neighbour_labels[min_plane_label].erase(min_plane_label);
      neighbour_labels[min_plane_label].erase(min_mse_candidate_label);
      neighbour_labels.erase(min_mse_candidate_label);
      // Put merged plane to plane_labels and back to Queue
      cell_grid->updateCell(min_plane_label, min_mse_candidate);
      plane_labels->insert(min_plane_label);
      min_mse_queue.insert(min_plane_label);
#ifdef DEBUG_DEPLEX
      std::clog << "- Merged " << min_mse_candidate_label << " and " << min_plane_label << '\n';
#endif
    }
  }
}

std::unordered_map<size_t, std::set<size_t>> PlaneExtractor::Impl::findPlaneNeighbours(
    std::set<size_t> const& plane_labels, CellGrid* cell_grid) {
  std::unordered_map<size_t, std::set<size_t>> neighbours;
  // Iterate over each grown planar segments and find its DSU-neighbours
  // TODO: Iterate over each planar segment? (Not every planar segment is present in plane_labels)
  for (size_t cell_id = 0; cell_id < cell_grid->size(); ++cell_id) {
    auto plane_segment_label = cell_grid->findLabel(cell_id);
    auto plane_segment_it = plane_labels.find(plane_segment_label);
    // If cell_id is part of one of plane_labels components
    if (plane_segment_it != plane_labels.end()) {
      for (auto neighbour_id : cell_grid->getNeighbours(cell_id)) {
        auto neighbour_label = cell_grid->findLabel(neighbour_id);
        // If cell-neighbour is not within the same component && cell-neighbour is mergeable
        if (neighbour_label != plane_segment_label &&
            (*cell_grid)[plane_segment_label].areNeighbours3D((*cell_grid)[neighbour_label])) {
          neighbours[plane_segment_label].insert(neighbour_label);
          neighbours[neighbour_label].insert(plane_segment_label);
        }
      }
    }
  }

  return neighbours;
}

void PlaneExtractor::Impl::cleanArtifacts() { labels_map_.setZero(); }

Eigen::VectorXi PlaneExtractor::Impl::toImageLabels(std::set<size_t> const& plane_labels, CellGrid* cell_grid) {
  Eigen::MatrixXi labels(Eigen::MatrixXi::Zero(image_height_, image_width_));

  int32_t cell_width = config_.getInt("patchSize");
  int32_t cell_height = config_.getInt("patchSize");

  int32_t stacked_cell_id = 0;
  for (auto row = 0; row < nr_vertical_cells_; ++row) {
    for (auto col = 0; col < nr_horizontal_cells_; ++col) {
      auto cell_label = cell_grid->findLabel(stacked_cell_id);
      if (plane_labels.find(cell_label) != plane_labels.end()) {
        auto cell_row = stacked_cell_id / nr_horizontal_cells_;
        auto cell_col = stacked_cell_id % nr_horizontal_cells_;
        // Fill cell with label
        auto label_row = cell_row * cell_height;
        auto label_col = cell_col * cell_width;
        for (auto i = label_row; i < label_row + cell_height; ++i) {
          for (auto j = label_col; j < label_col + cell_width; ++j) {
            labels.row(i)[j] = cell_label;
          }
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

  int32_t cell_width = config_.getInt("patchSize");
  int32_t cell_height = config_.getInt("patchSize");

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