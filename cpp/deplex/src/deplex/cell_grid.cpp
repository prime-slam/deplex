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
#include "cell_grid.h"

namespace deplex {
CellGrid::CellGrid(Eigen::MatrixXf const& points, config::Config const& config, int32_t number_horizontal_cells,
                   int32_t number_vertical_cells)
    : cell_width_(config.getInt("patchSize")),
      cell_height_(config.getInt("patchSize")),
      number_horizontal_cells_(number_horizontal_cells),
      number_vertical_cells_(number_vertical_cells),
      cell_grid_(number_vertical_cells),
      planar_mask_(number_horizontal_cells_ * number_vertical_cells_) {
  int32_t stacked_cell_id = 0;
  for (Eigen::Index cell_row = 0; cell_row < number_vertical_cells; ++cell_row) {
    cell_grid_[cell_row].reserve(number_horizontal_cells);
    for (Eigen::Index cell_col = 0; cell_col < number_horizontal_cells; ++cell_col) {
      Eigen::Index offset = stacked_cell_id * cell_width_ * cell_height_;
      Eigen::MatrixXf cell_points = points.block(offset, 0, cell_width_ * cell_height_, 3);
      cell_grid_[cell_row].emplace_back(cell_points, config);
      planar_mask_[stacked_cell_id] = cell_grid_[cell_row][cell_col].isPlanar();
      ++stacked_cell_id;
    }
  }
}

CellSegment const& CellGrid::getByCoordinates(size_t x, size_t y) const { return cell_grid_[x][y]; }

CellSegment const& CellGrid::operator[](size_t cell_id) const {
  return cell_grid_[cell_id / number_horizontal_cells_][cell_id % number_horizontal_cells_];
}

std::vector<bool> const& CellGrid::getPlanarMask() const { return planar_mask_; }

std::vector<size_t> CellGrid::getNeighbours(size_t cell_id) const {
  std::vector<size_t> neighbours;
  size_t x = cell_id / number_horizontal_cells_;
  size_t y = cell_id % number_horizontal_cells_;
  if (x >= 1) neighbours.push_back(cell_id - number_horizontal_cells_);
  if (x + 1 < number_vertical_cells_) neighbours.push_back(cell_id + number_horizontal_cells_);
  if (y >= 1) neighbours.push_back(cell_id - 1);
  if (y + 1 < number_horizontal_cells_) neighbours.push_back(cell_id + 1);

  return neighbours;
}

size_t CellGrid::size() const { return planar_mask_.size(); }

}  // namespace deplex