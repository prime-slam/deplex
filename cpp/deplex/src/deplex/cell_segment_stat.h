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

namespace deplex {
/**
 * Cell Segment Statistics.
 * MSE, Planarity score, PCA etc.
 */
class CellSegmentStat {
 public:
  CellSegmentStat();

  /**
   * CellSegmentStat constructor.
   * Compute cell's variance, eigenvalues (PCA), cell's normal etc
   *
   * @param cell_points Cell points block.
   */
  explicit CellSegmentStat(Eigen::MatrixX3f const& cell_points);

  /**
   * Merge two cell stats together
   *
   * @param other another CellSegmentStat.
   * @returns new CellSegmentStat with merged stats.
   */
  CellSegmentStat& operator+=(CellSegmentStat const& other);

  Eigen::Vector3f const& getNormal() const;

  Eigen::Vector3f const& getMean() const;

  float getScore() const;

  float getMSE() const;

  float getD() const;

  /**
   * Principal Component Analysis.
   * Compute cell's variance, eigenvalues (PCA), cell's normal etc
   */
  void fitPlane();

 private:
  float d_;
  float score_;
  float mse_;
  int32_t nr_pts_;
  Eigen::Vector3f coord_sum_;
  Eigen::Matrix3f variance_;
  Eigen::Vector3f mean_;
  Eigen::Vector3f normal_;
};

}  // namespace deplex