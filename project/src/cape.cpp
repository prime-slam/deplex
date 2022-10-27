#include "CAPE/cape.h"

namespace cape {
CAPE::CAPE(int32_t image_height, int32_t image_width, config::Config config)
    : _nr_horizontal_cells(image_width / config.getInt("patchSize")),
      _nr_vertical_cells(image_height / config.getInt("patchSize")) {
  _nr_total_cells = _nr_vertical_cells * _nr_horizontal_cells;
}

void CAPE::process(Eigen::MatrixXf const& pcd_array) {
  // 1. Planar cell fitting
  std::bitset<BITSET_SIZE> planar_flags = findPlanarCells(pcd_array);
  // 2. Histogram initialization
  Histogram hist = initializeHistogram(planar_flags);
  // 3. Compute cell dist tols
  std::vector<float> cell_dist_tols =
      computeCellDistTols(pcd_array, planar_flags);
  // 4. Region growing
  std::vector<PlaneSeg> plane_segments =
      CAPE::createPlaneSegments(hist, planar_flags, cell_dist_tols);
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

std::vector<PlaneSeg> CAPE::createPlaneSegments(
    Histogram hist, std::bitset<BITSET_SIZE> const& planar_flags,
    std::vector<float> const& cell_dist_tols) {
  std::vector<PlaneSeg> plane_segments;
  std::bitset<BITSET_SIZE> unassigned_mask(planar_flags);
  auto nr_planar_cells = static_cast<int32_t>(planar_flags.count());

  while (nr_planar_cells > 0) {
    // 1. Seeding
    std::vector<int32_t> seed_candidates = hist.getPointsFromMostFrequentBin();
    if (seed_candidates.size() <
        _config.getInt("minRegionGrowingCandidateSize")) {
      return plane_segments;
    }
    // 2. Select seed with minimum MSE
    auto ptr_min_mse_seed = std::min_element(
        seed_candidates.begin(), seed_candidates.end(),
        [this](int32_t lhs, int32_t rhs) {
          return _cell_grid[lhs]->getMSE() < _cell_grid[rhs]->getMSE();
        });
    int32_t seed_id = std::distance(seed_candidates.begin(), ptr_min_mse_seed);
    // 3. Grow seed
    std::shared_ptr<PlaneSeg> new_segment = _cell_grid[seed_id];
    int32_t y = seed_id / _nr_horizontal_cells;
    int32_t x = seed_id / _nr_vertical_cells;
    std::bitset<BITSET_SIZE> activation_map;
    growSeed(x, y, seed_id, unassigned_mask, &activation_map, cell_dist_tols);
    // 4. Merge activated cells & remove from hist
  }
}

void CAPE::growSeed(int32_t x, int32_t y, int32_t prev_index,
                    std::bitset<BITSET_SIZE> const& unassigned,
                    std::bitset<BITSET_SIZE>* activation_map,
                    std::vector<float> const& cell_dist_tols) {
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

}  // namespace cape
