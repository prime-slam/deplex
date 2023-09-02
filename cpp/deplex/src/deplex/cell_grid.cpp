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
#include "cell_grid.h"

#include <utility>

namespace deplex {
CellGrid::CellGrid(Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> const& points, config::Config const& config,
                   int32_t number_horizontal_cells, int32_t number_vertical_cells)
    : cell_width_(config.patch_size),
      cell_height_(config.patch_size),
      number_horizontal_cells_(number_horizontal_cells),
      number_vertical_cells_(number_vertical_cells),
      parent_(number_vertical_cells * number_horizontal_cells),
      component_size_(number_vertical_cells * number_horizontal_cells, 1),
      planar_mask_(number_vertical_cells * number_horizontal_cells) {
  cell_grid_.reserve(planar_mask_.size());
  Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> cell_continuous_points(points.rows(), points.cols());
  cellContinuousOrganize(points, &cell_continuous_points);

  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>> cell_points(cell_continuous_points.data(),
                                                                                         cell_width_ * cell_height_, 3);
#pragma omp parallel for default(none) shared(cell_continuous_points, config) firstprivate(cell_points)
  for (Eigen::Index cell_id = 0; cell_id < number_horizontal_cells_ * number_vertical_cells_; ++cell_id) {
    Eigen::Index offset = cell_id * cell_height_ * cell_width_ * 3;
    new (&cell_points) decltype(cell_points)(cell_continuous_points.data() + offset, cell_width_ * cell_height_, 3);
    cell_grid_[cell_id] = CellSegment(cell_points, config);
    parent_[cell_id] = cell_id;
    planar_mask_[cell_id] = cell_grid_[cell_id].isPlanar();
  }
}

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

CellSegment const& CellGrid::operator[](size_t cell_id) const { return cell_grid_[cell_id]; }

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

void CellGrid::cellContinuousOrganize(Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> const& unorganized_data,
                                      Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>* organized_pcd) {
  int32_t image_width = number_horizontal_cells_ * cell_width_;
#pragma omp parallel for default(none) shared(cell_width_, cell_height_, organized_pcd, unorganized_data, image_width)
  for (Eigen::Index cell_id = 0; cell_id < number_horizontal_cells_ * number_vertical_cells_; ++cell_id) {
    Eigen::Index outer_cell_stride = cell_width_ * cell_height_ * cell_id;
    for (Eigen::Index i = 0; i < cell_height_; ++i) {
      Eigen::Index cell_row_stride = i * cell_width_;
      organized_pcd->block(cell_row_stride + outer_cell_stride, 0, cell_width_, 3) =
          unorganized_data.block(i * image_width + (cell_id / number_horizontal_cells_ * image_width * cell_height_) +
                                     (cell_id * cell_width_) % image_width,
                                 0, cell_height_, 3);
    }
  }
}

}  // namespace deplex