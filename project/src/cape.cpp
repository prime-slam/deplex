#include "CAPE/cape.h"
#include <numeric>

#ifdef DEBUG_CAPE
#include <iostream>
#include <fstream>
#endif

namespace cape {
CAPE::CAPE(int32_t image_height, int32_t image_width, config::Config config)
    : _config(config),
      _nr_horizontal_cells(image_width / config.getInt("patchSize")),
      _nr_vertical_cells(image_height / config.getInt("patchSize")),
      _nr_total_cells(_nr_horizontal_cells * _nr_vertical_cells),
      _nr_pts_per_cell(pow(config.getInt("patchSize"), 2)),
      _image_height(image_height),
      _image_width(image_width),
      _cell_grid(_nr_total_cells, nullptr),
      _grid_plane_seg_map(_nr_vertical_cells, _nr_horizontal_cells, 0) {}

void CAPE::process(Eigen::MatrixXf const& pcd_array) {
  // 1. Planar cell fitting
  std::bitset<BITSET_SIZE> planar_flags = findPlanarCells(pcd_array);
#ifdef DEBUG_CAPE
  planarCellsToLabels(planar_flags, "dbg_1_planar_cells.csv");
  std::clog << "[DebugInfo] Planar cell found: " << planar_flags.count()
            << '\n';
#endif
  // 2. Histogram initialization
  Histogram hist = initializeHistogram(planar_flags);
  // 3. Compute cell dist tols
  std::vector<float> cell_dist_tols =
      computeCellDistTols(pcd_array, planar_flags);
  // 4. Region growing
  auto plane_segments = createPlaneSegments(hist, planar_flags, cell_dist_tols);
#ifdef DEBUG_CAPE
  planeSegmentsMapToLabels("dbg_2_plane_segments_raw.csv");
  std::clog << "[DebugInfo] Plane segments found: " << plane_segments.size()
            << '\n';
#endif
  // 5. Merge planes
  std::vector<int32_t> merge_labels = mergePlanes(plane_segments);
#ifdef DEBUG_CAPE
  std::vector<int32_t> sorted_labels(merge_labels);
  std::sort(sorted_labels.begin(), sorted_labels.end());

  std::clog << "[DebugInfo] Planes number after merge: "
            << std::distance(
                   sorted_labels.begin(),
                   std::unique(sorted_labels.begin(), sorted_labels.end()))
            << '\n';
#endif
  // 6. Refinement (optional)
  if (_config.getBool("doRefinement")) {
    refinePlanes();
  }
}

std::bitset<BITSET_SIZE> CAPE::findPlanarCells(
    Eigen::MatrixXf const& pcd_array) {
  std::bitset<BITSET_SIZE> planar_flags;
  int32_t cell_width = _config.getInt("patchSize");
  int32_t cell_height = _config.getInt("patchSize");
  int32_t stacked_cell_id = 0;
  for (Eigen::Index cell_r = 0; cell_r < _nr_vertical_cells; ++cell_r) {
    for (Eigen::Index cell_h = 0; cell_h < _nr_horizontal_cells; ++cell_h) {
      _cell_grid[stacked_cell_id] = std::make_shared<PlaneSeg>(
          stacked_cell_id, cell_width, cell_height, pcd_array, _config);
      planar_flags[stacked_cell_id] = _cell_grid[stacked_cell_id]->isPlanar();
      ++stacked_cell_id;
    }
  }
  return planar_flags;
}

Histogram CAPE::initializeHistogram(
    std::bitset<BITSET_SIZE> const& planar_flags) {
  Eigen::MatrixXd spherical_coord(_nr_total_cells, 2);
  for (size_t cell_id = planar_flags._Find_first();
       cell_id != planar_flags.size();
       cell_id = planar_flags._Find_next(cell_id)) {
    Eigen::Vector3d cell_normal = _cell_grid[cell_id]->getNormal();
    double n_proj_norm =
        sqrt(cell_normal[0] * cell_normal[0] + cell_normal[1] * cell_normal[1]);
    spherical_coord(cell_id, 0) = acos(-cell_normal[2]);
    spherical_coord(cell_id, 1) =
        atan2(cell_normal[0] / n_proj_norm, cell_normal[1] / n_proj_norm);
  }
  int nr_bins_per_coord = _config.getInt("histogramBinsPerCoord");
  return Histogram{nr_bins_per_coord, spherical_coord, planar_flags};
}

std::vector<float> CAPE::computeCellDistTols(
    Eigen::MatrixXf const& pcd_array,
    std::bitset<BITSET_SIZE> const& planar_flags) {
  std::vector<float> cell_dist_tols(_nr_total_cells, 0);
  double cos_angle_for_merge = _config.getFloat("minCosAngleForMerge");
  float sin_angle_for_merge = sqrt(1 - pow(cos_angle_for_merge, 2));
  // TODO: Put "minMergeDist" to config
  float min_merge_dist = 20.0f;
  float max_merge_dist = _config.getFloat("maxMergeDist");

  for (size_t cell_id = planar_flags._Find_first();
       cell_id != planar_flags.size();
       cell_id = planar_flags._Find_next(cell_id)) {
    float cell_diameter =
        (pcd_array.block(cell_id * _nr_pts_per_cell + _nr_pts_per_cell - 1, 0,
                         1, 3) -
         pcd_array.block(cell_id * _nr_pts_per_cell, 0, 1, 3))
            .norm();
    float truncated_distance =
        std::min(std::max(cell_diameter * sin_angle_for_merge, min_merge_dist),
                 max_merge_dist);
    cell_dist_tols[cell_id] = powf(truncated_distance, 2);
  }

  return cell_dist_tols;
}

std::vector<std::shared_ptr<PlaneSeg>> CAPE::createPlaneSegments(
    Histogram hist, std::bitset<BITSET_SIZE> const& planar_flags,
    std::vector<float> const& cell_dist_tols) {
  std::vector<std::shared_ptr<PlaneSeg>> plane_segments;
  std::bitset<BITSET_SIZE> unassigned_mask(planar_flags);
  auto remaining_planar_cells = static_cast<int32_t>(planar_flags.count());

  while (remaining_planar_cells > 0) {
    // 1. Seeding
    std::vector<int32_t> seed_candidates = hist.getPointsFromMostFrequentBin();
    if (seed_candidates.size() <
        _config.getInt("minRegionGrowingCandidateSize")) {
      return plane_segments;
    }
    // 2. Select seed with minimum MSE
    int32_t seed_id;
    double min_mse = INT_MAX;
    for (int32_t seed_candidate : seed_candidates) {
      if (_cell_grid[seed_candidate]->getMSE() < min_mse) {
        seed_id = seed_candidate;
        min_mse = _cell_grid[seed_candidate]->getMSE();
      }
    }
    // 3. Grow seed
    std::shared_ptr<PlaneSeg> new_segment = _cell_grid[seed_id];
    int32_t y = seed_id / _nr_horizontal_cells;
    int32_t x = seed_id % _nr_horizontal_cells;
    std::bitset<BITSET_SIZE> activation_map;
    growSeed(x, y, seed_id, unassigned_mask, &activation_map, cell_dist_tols);
    // 4. Merge activated cells & remove from hist
    for (size_t i = activation_map._Find_first(); i != activation_map.size();
         i = activation_map._Find_next(i)) {
      *new_segment += *_cell_grid[i];
      hist.removePoint(static_cast<int32_t>(i));
      --remaining_planar_cells;
    }
    unassigned_mask &= (~activation_map);
    size_t nr_cells_activated = activation_map.count();

    if (nr_cells_activated < _config.getInt("minRegionGrowingCellsActivated")) {
      continue;
    }

    new_segment->calculateStats();

    // 5. Model fitting
    if (new_segment->getScore() > _config.getFloat("minRegionPlanarityScore")) {
      plane_segments.push_back(new_segment);
      auto nr_curr_planes = static_cast<int32_t>(plane_segments.size());
      // Mark cells
      int stacked_cell_id = 0;
      for (int32_t row_id = 0; row_id < _nr_vertical_cells; ++row_id){
        auto row = _grid_plane_seg_map.ptr<int32_t>(row_id);
        for (int32_t col_id = 0; col_id < _nr_horizontal_cells; ++col_id){
          if (activation_map[stacked_cell_id]){
            row[col_id] = nr_curr_planes;
          }
          ++stacked_cell_id;
        }
      }
    }
  }

  return plane_segments;
}

std::vector<int32_t> CAPE::mergePlanes(
    std::vector<std::shared_ptr<PlaneSeg>>& plane_segments) {
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
         col_id != planes_association_mx[row_id].size();
         col_id = planes_association_mx[row_id]._Find_next(col_id)) {
      double cos_angle = plane_segments[plane_id]->getNormal().dot(
          plane_segments[col_id]->getNormal());
      double distance = pow(plane_segments[plane_id]->getNormal().dot(
                                plane_segments[col_id]->getMean()) +
                                plane_segments[plane_id]->getD(),
                            2);
      if (cos_angle > _config.getFloat("minCosAngleForMerge") &&
          distance < _config.getFloat("maxMergeDist")) {
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

void CAPE::refinePlanes() {
  std::cerr << "Refinement not yet implemented\n";
}

void CAPE::growSeed(int32_t x, int32_t y, int32_t prev_index,
                    std::bitset<BITSET_SIZE> const& unassigned,
                    std::bitset<BITSET_SIZE>* activation_map,
                    std::vector<float> const& cell_dist_tols) const {
  int32_t index = x + _nr_horizontal_cells * y;
  if (index >= _nr_total_cells)
    throw std::out_of_range("growSeed: Index out of total cell number");
  if (!unassigned[index] || (*activation_map)[index]) {
    return;
  }

  double d_1 = _cell_grid[prev_index]->getD();
  Eigen::Vector3d normal_1 = _cell_grid[prev_index]->getNormal();
  Eigen::Vector3d normal_2 = _cell_grid[index]->getNormal();
  Eigen::Vector3d mean_2 = _cell_grid[index]->getMean();

  double cos_angle = normal_1.dot(normal_2);
  double merge_dist = pow(normal_1.dot(mean_2) + d_1, 2);
  if (cos_angle < _config.getFloat("minCosAngleForMerge") ||
      merge_dist > cell_dist_tols[index]) {
    return;
  }

  activation_map->set(index);
  if (x > 0)
    growSeed(x - 1, y, index, unassigned, activation_map, cell_dist_tols);
  if (x < _nr_horizontal_cells - 1)
    growSeed(x + 1, y, index, unassigned, activation_map, cell_dist_tols);
  if (y > 0)
    growSeed(x, y - 1, index, unassigned, activation_map, cell_dist_tols);
  if (y < _nr_vertical_cells - 1)
    growSeed(x, y + 1, index, unassigned, activation_map, cell_dist_tols);
}

std::vector<std::bitset<BITSET_SIZE>> CAPE::getConnectedComponents(
    size_t nr_planes) const {
  std::vector<std::bitset<BITSET_SIZE>> planes_assoc_matrix(nr_planes);

  for (int32_t row_id = 0; row_id < _grid_plane_seg_map.rows - 1; ++row_id) {
    auto row = _grid_plane_seg_map.ptr<int>(row_id);
    auto next_row = _grid_plane_seg_map.ptr<int>(row_id + 1);
    for (int32_t col_id = 0; col_id < _grid_plane_seg_map.cols - 1; ++col_id) {
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

#ifdef DEBUG_CAPE

template<typename T>
void vectorToCSV(std::vector<std::vector<T>> const& data, std::string const& out_path, char sep=','){
  std::ofstream f_out(out_path);
  for (const auto & row : data){
    for (auto value = row.begin(); value != row.end(); ++value){
      if (value != row.begin()){
        f_out << sep;
      }
      f_out << *value;
    }
    f_out << '\n';
  }
}

void CAPE::planarCellsToLabels(std::bitset<BITSET_SIZE> const& planar_flags,
                               std::string const& save_path) {
  std::vector<std::vector<int32_t>> labels(
      _image_height, std::vector<int32_t>(_image_width, 0));

  int32_t cell_width = _config.getInt("patchSize");
  int32_t cell_height = _config.getInt("patchSize");

  for (auto cell_id = planar_flags._Find_first();
       cell_id != planar_flags.size();
       cell_id = planar_flags._Find_next(cell_id)) {
    auto cell_row = cell_id / _nr_horizontal_cells;
    auto cell_col = cell_id % _nr_horizontal_cells;
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

void CAPE::planeSegmentsMapToLabels(std::string const& save_path) {
  std::vector<std::vector<int32_t>> labels(
      _image_height, std::vector<int32_t>(_image_width, 0));

  int32_t cell_width = _config.getInt("patchSize");
  int32_t cell_height = _config.getInt("patchSize");

  int32_t stacked_cell_id = 0;
  for (auto row = 0; row < _grid_plane_seg_map.rows; ++row) {
    for (auto col = 0; col < _grid_plane_seg_map.cols; ++col) {
      auto cell_row = stacked_cell_id / _nr_horizontal_cells;
      auto cell_col = stacked_cell_id % _nr_horizontal_cells;
      // Fill cell with label
      auto label_row = cell_row * cell_height;
      auto label_col = cell_col * cell_width;
      for (auto i = label_row; i < label_row + cell_height; ++i) {
        for (auto j = label_col; j < label_col + cell_width; ++j) {
          labels[i][j] = _grid_plane_seg_map[row][col];
        }
      }
      ++stacked_cell_id;
    }
  }

  vectorToCSV(labels, save_path);
}

#endif

}  // namespace cape
