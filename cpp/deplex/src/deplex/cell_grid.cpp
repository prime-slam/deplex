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
      parent_(number_vertical_cells * number_horizontal_cells),
      component_size_(number_vertical_cells * number_horizontal_cells, 1),
      planar_mask_(number_vertical_cells * number_horizontal_cells) {
  cell_grid_.reserve(planar_mask_.size());
  int32_t stacked_cell_id = 0;
  for (Eigen::Index cell_row = 0; cell_row < number_vertical_cells; ++cell_row) {
    for (Eigen::Index cell_col = 0; cell_col < number_horizontal_cells; ++cell_col) {
      Eigen::Index offset = stacked_cell_id * cell_width_ * cell_height_;
      Eigen::MatrixXf cell_points = points.block(offset, 0, cell_width_ * cell_height_, 3);
      parent_[stacked_cell_id] = stacked_cell_id;
      cell_grid_.emplace_back(cell_points, config);
      planar_mask_[stacked_cell_id] = cell_grid_[stacked_cell_id].isPlanar();
      ++stacked_cell_id;
    }
  }
}

CellSegment const& CellGrid::operator[](size_t cell_id) const { return cell_grid_[cell_id]; }

size_t CellGrid::findLabel(size_t cell_id) {
  return (parent_[cell_id] == cell_id) ? cell_id : parent_[cell_id] = findLabel(parent_[cell_id]);
}

void CellGrid::uniteLabels(size_t a, size_t b) {
  a = findLabel(a);
  b = findLabel(b);
  if (component_size_[a] > component_size_[b]) std::swap(a, b);
  component_size_[b] += component_size_[a];
  parent_[a] = b;
}

void CellGrid::updateCell(size_t cell_id, CellSegment new_cell) {
  planar_mask_[cell_id] = new_cell.isPlanar();
  cell_grid_[cell_id] = std::move(new_cell);
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