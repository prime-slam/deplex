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
#include "cell_segment.h"

namespace deplex {
CellSegment::CellSegment(Eigen::MatrixXf const& cell_points, config::Config const& config) : is_planar_(false) {
  size_t valid_pts_threshold = cell_points.size() / config.getInt("minPtsPerCell");
  int32_t cell_width = config.getInt("patchSize");
  int32_t cell_height = config.getInt("patchSize");

  bool is_valid =
      hasValidPoints(cell_points, valid_pts_threshold) &&
      isDepthContinuous(cell_points, cell_width, cell_height, config.getFloat("depthDiscontinuityThreshold"),
                        config.getInt("maxNumberDepthDiscontinuity"));
  if (!is_valid) return;
  stats_ = CellSegmentStat(cell_points.cast<double>());
  is_planar_ = hasSmallPlaneError(config.getFloat("depthSigmaCoeff"), config.getFloat("depthSigmaMargin"));
  // TODO: add minMergeDist to config
  merge_tolerance_ = calculateMergeTolerance(cell_points, config.getFloat("minCosAngleForMerge"), 20.0,
                                             config.getFloat("maxMergeDist"));
}

CellSegment& CellSegment::operator+=(CellSegment const& other) {
  stats_ += other.stats_;
  return *this;
}

CellSegmentStat const& CellSegment::getStat() const { return stats_; };

bool CellSegment::isPlanar() const { return is_planar_; }

float CellSegment::getMergeTolerance() const { return merge_tolerance_; }

void CellSegment::calculateStats() { stats_.fitPlane(); }

bool CellSegment::hasValidPoints(Eigen::MatrixXf const& cell_points, size_t valid_pts_threshold) const {
  Eigen::Index valid_pts = (cell_points.col(2).array() > 0).count();
  return valid_pts >= valid_pts_threshold;
}

bool CellSegment::isHorizontalContinuous(Eigen::MatrixXf const& cell_z, float depth_disc_threshold,
                                         int32_t max_number_depth_disc) const {
  Eigen::Index middle = cell_z.rows() / 2;
  float prev_depth = cell_z(middle, 0);
  int32_t disc_count = 0;
  for (Eigen::Index col = 0; col < cell_z.cols(); ++col) {
    float curr_depth = cell_z(middle, col);
    if (curr_depth > 0 && fabsf(curr_depth - prev_depth) < depth_disc_threshold) {
      prev_depth = curr_depth;
    } else if (curr_depth > 0)
      ++disc_count;
  }

  return disc_count < max_number_depth_disc;
}

bool CellSegment::isVerticalContinuous(Eigen::MatrixXf const& cell_z, float depth_disc_threshold,
                                       int32_t max_number_depth_disc) const {
  Eigen::Index middle = cell_z.cols() / 2;
  float prev_depth = cell_z(0, middle);
  int32_t disc_count = 0;
  for (Eigen::Index row = 0; row < cell_z.rows(); ++row) {
    float curr_depth = cell_z(row, middle);
    if (curr_depth > 0 && fabsf(curr_depth - prev_depth) < depth_disc_threshold) {
      prev_depth = curr_depth;
    } else if (curr_depth > 0)
      ++disc_count;
  }

  return disc_count < max_number_depth_disc;
}

bool CellSegment::isDepthContinuous(Eigen::MatrixXf const& cell_points, int32_t cell_width, int32_t cell_height,
                                    float depth_disc_threshold, int32_t max_number_depth_disc) const {
  Eigen::MatrixXf cell_z = cell_points.col(2).reshaped(cell_width, cell_height);

  return isHorizontalContinuous(cell_z, depth_disc_threshold, max_number_depth_disc) &&
         isVerticalContinuous(cell_z, depth_disc_threshold, max_number_depth_disc);
}

bool CellSegment::hasSmallPlaneError(float depth_sigma_coeff, float depth_sigma_margin) const {
  float planar_threshold = depth_sigma_coeff * powf(stats_.getMean()[2], 2) + depth_sigma_margin;
  return stats_.getMSE() <= pow(planar_threshold, 2);
}

float CellSegment::calculateMergeTolerance(Eigen::MatrixXf const& cell_points, float cos_angle, float min_merge_dist,
                                           float max_merge_dist) const {
  float sin_angle_for_merge = sqrtf(1 - powf(cos_angle, 2));
  float cell_diameter = (cell_points.row(0) - cell_points.row(cell_points.rows() - 1)).norm();
  float truncated_distance = std::min(std::max(cell_diameter * sin_angle_for_merge, min_merge_dist), max_merge_dist);
  return powf(truncated_distance, 2);
}

}  // namespace deplex