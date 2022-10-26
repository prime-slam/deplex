#include "CAPE/cape.h"

namespace cape {
CAPE::CAPE(int32_t image_height, int32_t image_width, config::Config config)
    : _nr_horizontal_cells(image_width / config.getInt("patchSize")),
      _nr_vertical_cells(image_height / config.getInt("patchSize")) {
  _nr_total_cells = _nr_vertical_cells * _nr_horizontal_cells;
}

void CAPE::process(Eigen::MatrixXf const& pcd_array) {}

std::bitset<BITSET_SIZE> CAPE::findPlanarCells(
    Eigen::MatrixXf const& pcd_array) {
  std::bitset<BITSET_SIZE> planar_flags;
  int32_t stacked_cell_id = 0;
  for (Eigen::Index cell_r = 0; cell_r < _nr_vertical_cells; ++cell_r) {
    for (Eigen::Index cell_h = 0; cell_h < _nr_horizontal_cells; ++cell_h) {
      _cell_grid[stacked_cell_id] =
          std::make_shared<PlaneSeg>(stacked_cell_id, pcd_array, _config);
      planar_flags[stacked_cell_id] = _cell_grid[stacked_cell_id]->isPlanar();
      ++stacked_cell_id;
    }
  }
  return planar_flags;
}
}  // namespace cape
