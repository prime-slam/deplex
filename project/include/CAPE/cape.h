#pragma once

#include "planeseg.hpp"

#include <Eigen/Core>
#include <bitset>
#include <vector>
#include <memory>

#ifndef BITSET_SIZE
#define BITSET_SIZE 65536  // 2^16
#endif

namespace cape {
class CAPE {
 public:
  CAPE(int32_t image_height, int32_t image_width, config::Config config);
  void process(Eigen::MatrixXf const& pcd_array);
 private:
  config::Config _config;
  int32_t _nr_horizontal_cells;
  int32_t _nr_vertical_cells;
  int32_t _nr_total_cells;
  std::vector<std::shared_ptr<PlaneSeg>> _cell_grid;
  std::bitset<BITSET_SIZE> findPlanarCells(Eigen::MatrixXf const& pcd_array);
};
}  // namespace cape
