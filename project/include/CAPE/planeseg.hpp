#pragma once

#include "config.hpp"

#include <Eigen/Dense>
#include <memory>

namespace cape {
class PlaneSeg {
 public:
  PlaneSeg(Eigen::MatrixXf const& pcd_array);
  bool isPlanar() const;

 private:

  std::unique_ptr<Eigen::MatrixXf const> const _ptr_pcd_array;
  std::unique_ptr<config::Config const> const _config;
  int32_t _nr_pts_per_cell;
  int32_t _cell_width;
  int32_t _cell_height;
  int32_t _offset;
  bool isValidPoints() const;
  bool isDepthContinuous() const;
  bool isValidMSE() const;
  bool _isHorizontalContinuous(Eigen::MatrixXf const& cell_z) const;
  bool _isVerticalContinuous(Eigen::MatrixXf const& cell_z) const;
};

bool PlaneSeg::isPlanar() const {
  return isValidPoints() && isDepthContinuous() && isValidMSE();
}

bool PlaneSeg::isValidPoints() const {
  Eigen::VectorXf cell_z =
      _ptr_pcd_array->block(_offset, 2, _nr_pts_per_cell, 1);

  Eigen::Index valid_pts_threshold =
      _nr_pts_per_cell / _config->getInt("minPtsPerCell");
  Eigen::Index valid_pts = (cell_z.array() > 0).count();
  return valid_pts >= valid_pts_threshold;
}

bool PlaneSeg::_isHorizontalContinuous(Eigen::MatrixXf const& cell_z) const {
  float depth_disc_threshold = _config->getFloat("depthDiscontinuityThreshold");
  Eigen::Index middle = cell_z.rows() / 2;
  float prev_depth = cell_z(middle, 0);
  int32_t disc_count = 0;
  for (Eigen::Index col = 0; col < cell_z.cols(); ++col) {
    float curr_depth = cell_z(middle, col);
    if (curr_depth > 0 &&
        fabsf(curr_depth - prev_depth) < depth_disc_threshold) {
      prev_depth = curr_depth;
    } else if (curr_depth > 0)
      ++disc_count;
  }

  return disc_count < _config->getInt("maxNumberDepthDiscontinuity");
}

bool PlaneSeg::_isVerticalContinuous(Eigen::MatrixXf const& cell_z) const {
  float depth_disc_threshold = _config->getFloat("depthDiscontinuityThreshold");
  Eigen::Index middle = cell_z.cols() / 2;
  float prev_depth = cell_z(0, middle);
  int32_t disc_count = 0;
  for (Eigen::Index row = 0; row < cell_z.rows(); ++row) {
    float curr_depth = cell_z(row, middle);
    if (curr_depth > 0 &&
        fabsf(curr_depth - prev_depth) < depth_disc_threshold) {
      prev_depth = curr_depth;
    } else if (curr_depth > 0)
      ++disc_count;
  }

  return disc_count < _config->getInt("maxNumberDepthDiscontinuity");
}

bool PlaneSeg::isDepthContinuous() const {
  Eigen::MatrixXf cell_z =
      _ptr_pcd_array->block(_offset, 2, _nr_pts_per_cell, 1)
          .reshaped(_cell_height, _cell_width);

  return _isHorizontalContinuous(cell_z) && _isVerticalContinuous(cell_z);
}

}  // namespace cape