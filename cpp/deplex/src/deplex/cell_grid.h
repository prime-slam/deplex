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
#pragma once

#include <Eigen/Core>

#include "cell_segment.h"

namespace deplex {
class CellGrid {
 public:
  CellGrid(Eigen::MatrixXf const& points, config::Config const& config, int32_t number_horizontal_cells,
           int32_t number_vertical_cells);

  size_t findLabel(size_t v);

  void uniteLabels(size_t a, size_t b);

  void updateCell(size_t cell_id, CellSegment new_cell);

  CellSegment const& operator[](size_t cell_id) const;

  std::vector<bool> const& getPlanarMask() const;

  std::vector<size_t> getNeighbours(size_t cell_id) const;

  size_t size() const;

 private:
  int32_t cell_width_;
  int32_t cell_height_;
  int32_t number_horizontal_cells_;
  int32_t number_vertical_cells_;
  std::vector<size_t> parent_;
  std::vector<int> component_size_;
  std::vector<CellSegment> cell_grid_;
  std::vector<bool> planar_mask_;
};
}  // namespace deplex