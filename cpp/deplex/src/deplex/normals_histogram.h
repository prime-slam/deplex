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

#include <vector>

#include <Eigen/Core>

namespace deplex {
/**
 * Normals Histogram class.
 * Get dominant normal direction in spherical coordinates.
 */
class NormalsHistogram {
 public:
  /**
   * NormalsHistogram constructor.
   * Build histogram in spherical coordinates from cell's normals.
   *
   * @param nr_bins_per_coord Config parameter, granularity of spherical space.
   * @param normals Cell's normals.
   */
  NormalsHistogram(int32_t nr_bins_per_coord, Eigen::MatrixX3f const& normals);

  /**
   * Get the id's of cells whose normals lie in the dominant direction.
   *
   * @returns Vector of dominant cells indices.
   */
  std::vector<int32_t> getPointsFromMostFrequentBin() const;

  /**
   * Remove point from histogram.
   *
   * @note Thus, we need to recalculate most frequent bin in corresponding method.
   */
  void removePoint(int32_t point_id);

 private:
  std::vector<int32_t> bins_;
  std::vector<int32_t> hist_;
  int32_t nr_bins_per_coord_;
  int32_t nr_points_;
};
}  // namespace deplex