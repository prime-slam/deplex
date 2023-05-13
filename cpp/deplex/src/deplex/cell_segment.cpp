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
#include "cell_segment.h"

namespace deplex {
CellSegment::CellSegment() : stats_(), is_planar_(false), merge_tolerance_(), min_merge_cos_(), max_merge_dist_() {}

CellSegment::CellSegment(Eigen::MatrixX3f const& cell_points, config::Config const& config)
    : is_planar_(false), min_merge_cos_(config.min_cos_angle_merge), max_merge_dist_(config.max_merge_dist) {
  size_t valid_pts_threshold = cell_points.size() / config.min_pts_per_cell;
  int32_t cell_width = config.patch_size;
  int32_t cell_height = config.patch_size;

  bool is_valid = hasValidPoints(cell_points, valid_pts_threshold) &&
                  isDepthContinuous(cell_points, cell_width, cell_height, config.depth_discontinuity_threshold,
                                    config.max_number_depth_discontinuity);
  if (!is_valid) return;
  stats_ = CellSegmentStat(cell_points);
  is_planar_ = hasSmallPlaneError(config.depth_sigma_coeff, config.depth_sigma_margin);
  // TODO: add minMergeDist to config
  merge_tolerance_ = calculateMergeTolerance(cell_points, config.min_cos_angle_merge, 20.0, config.max_merge_dist);
}

CellSegment& CellSegment::operator+=(CellSegment const& other) {
  stats_ += other.stats_;
  return *this;
}

bool CellSegment::areNeighbours3D(CellSegment const& other) const {
  if (!this->is_planar_ || !other.is_planar_) return false;
  auto cos_angle = this->getStat().getNormal().dot(other.getStat().getNormal());
  auto distance = pow(this->getStat().getNormal().dot(other.getStat().getMean()) + this->getStat().getD(), 2);
  return cos_angle >= min_merge_cos_ && distance <= max_merge_dist_;
}

CellSegmentStat const& CellSegment::getStat() const { return stats_; };

bool CellSegment::isPlanar() const { return is_planar_; }

float CellSegment::getMergeTolerance() const { return merge_tolerance_; }

void CellSegment::calculateStats() { stats_.fitPlane(); }

bool CellSegment::hasValidPoints(Eigen::MatrixX3f const& cell_points, size_t valid_pts_threshold) const {
  Eigen::Index valid_pts = (cell_points.col(2).array() > 0).count();
  return valid_pts >= valid_pts_threshold;
}

bool CellSegment::isHorizontalContinuous(Eigen::MatrixX3f const& cell_points, int32_t cell_width, int32_t cell_height,
                                         float depth_disc_threshold, int32_t max_number_depth_disc) const {
  Eigen::Index middle = cell_width * cell_height / 2;
  float prev_depth = cell_points.col(2)(middle);
  int32_t disc_count = 0;
  for (Eigen::Index i = middle; i < middle + cell_width; ++i) {
    float curr_depth = cell_points.col(2)(i);
    if (curr_depth > 0 && fabsf(curr_depth - prev_depth) < depth_disc_threshold) {
      prev_depth = curr_depth;
    } else if (curr_depth > 0)
      ++disc_count;
  }

  return disc_count < max_number_depth_disc;
}

bool CellSegment::isVerticalContinuous(Eigen::MatrixX3f const& cell_points, int32_t cell_width,
                                       float depth_disc_threshold, int32_t max_number_depth_disc) const {
  float prev_depth = cell_points.col(2)(cell_width / 2);
  int32_t disc_count = 0;
  for (Eigen::Index i = cell_width / 2; i < cell_points.rows(); i += cell_width) {
    float curr_depth = cell_points.col(2)(i);
    if (curr_depth > 0 && fabsf(curr_depth - prev_depth) < depth_disc_threshold) {
      prev_depth = curr_depth;
    } else if (curr_depth > 0)
      ++disc_count;
  }

  return disc_count < max_number_depth_disc;
}

bool CellSegment::isDepthContinuous(Eigen::MatrixX3f const& cell_points, int32_t cell_width, int32_t cell_height,
                                    float depth_disc_threshold, int32_t max_number_depth_disc) const {
  return isHorizontalContinuous(cell_points, cell_width, cell_height, depth_disc_threshold, max_number_depth_disc) &&
         isVerticalContinuous(cell_points, cell_width, depth_disc_threshold, max_number_depth_disc);
}

bool CellSegment::hasSmallPlaneError(float depth_sigma_coeff, float depth_sigma_margin) const {
  float planar_threshold = depth_sigma_coeff * powf(stats_.getMean()[2], 2) + depth_sigma_margin;
  return stats_.getMSE() <= pow(planar_threshold, 2);
}

float CellSegment::calculateMergeTolerance(Eigen::MatrixX3f const& cell_points, float cos_angle, float min_merge_dist,
                                           float max_merge_dist) const {
  float sin_angle_for_merge = sqrtf(1 - powf(cos_angle, 2));
  float cell_diameter = (cell_points.row(0) - cell_points.row(cell_points.rows() - 1)).norm();
  float truncated_distance = std::min(std::max(cell_diameter * sin_angle_for_merge, min_merge_dist), max_merge_dist);
  return powf(truncated_distance, 2);
}

}  // namespace deplex