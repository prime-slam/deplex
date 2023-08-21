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
#pragma once

#include <Eigen/Core>

#include "cell_segment.h"

#include "CTPL/ctpl_stl.h"

namespace deplex {
/**
 * Class to store and work with cell-related data
 */
class CellGrid {
 public:
  /**
   * CellGrid constructor.
   *
   * @param points Point cloud points (row-major).
   * @param config Plane extractor config.
   * @param number_horizontal_cells Total number of horizontal cells.
   * @param number_vertical_cells Total number of vertical cells.
   */
  CellGrid(Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> const& points, config::Config const& config,
           int32_t number_horizontal_cells, int32_t number_vertical_cells);

  CellGrid(Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> const& points, config::Config const& config,
           int32_t number_horizontal_cells, int32_t number_vertical_cells, ctpl::thread_pool& pool);

  /**
   * Get cell's label.
   *
   * @note TODO: EXPERIMENTAL DSU REWORK, IN PROGRESS.
   * @param cell_id Cell id.
   * @returns Cell's label.
   */
  size_t findLabel(size_t cell_id);

  /**
   * DSU Unite operation.
   *
   * @note TODO: EXPERIMENTAL DSU REWORK, IN PROGRESS.
   * @param a First set to unite.
   * @param b Second set to unite.
   */
  void uniteLabels(size_t a, size_t b);

  /**
   * Update cell by id
   *
   * @note TODO: EXPERIMENTAL DSU REWORK, IN PROGRESS.
   * @param cell_id Cell id.
   * @param new_cell New cell to push into cell id.
   */
  void updateCell(size_t cell_id, CellSegment new_cell);

  CellSegment const& operator[](size_t cell_id) const;

  /**
   * Return flat boolean mask of size of total cells. Boolean value corresponds to cell's planarity.
   *
   * @returns Boolean mask. 1 - cell[i] is planar, 0 - cell[i] is not planar
   */
  std::vector<bool> const& getPlanarMask() const;

  /**
   * Get cell 2D-neighbours indices
   *
   * @returns Vector of neighbours indices (Maximum 4 neighbours)
   */
  std::vector<size_t> getNeighbours(size_t cell_id) const;

  /**
   * Number of total cells
   *
   * @returns Number of total cells
   */
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

  /**
   * Organize point cloud, so that points corresponding to one cell lie sequentially in memory.
   *
   * @param unorganized_data Points matrix [Nx3] with default Eigen alignment (RowMajor).
   * @returns Cell-wise organized points (RowMajor).
   */
  void cellContinuousOrganize(Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> const& unorganized_data,
                              Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>* organized_pcd);
};
}  // namespace deplex