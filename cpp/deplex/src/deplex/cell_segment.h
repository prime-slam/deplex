#pragma once

#include <cstdint>

#include "cell_segment_stat.h"
#include "deplex/config.h"

namespace deplex {
class CellSegment {
 public:
  CellSegment(Eigen::MatrixXf const& cell_points, config::Config const& config);

  CellSegment& operator+=(CellSegment const& other);

  CellSegmentStat const& getStat() const;

  void calculateStats();

  bool isPlanar() const;

 private:
  CellSegmentStat stats_;
  bool is_planar_;

  bool hasValidPoints(Eigen::MatrixXf const& cell_points, size_t valid_pts_threshold) const;

  bool isDepthContinuous(Eigen::MatrixXf const& cell_points, int32_t cell_width, int32_t cell_height,
                         float depth_disc_threshold, int32_t max_number_depth_disc) const;

  bool isHorizontalContinuous(Eigen::MatrixXf const& cell_z, float depth_disc_threshold,
                              int32_t max_number_depth_disc) const;

  bool isVerticalContinuous(Eigen::MatrixXf const& cell_z, float depth_disc_threshold,
                            int32_t max_number_depth_disc) const;

  bool hasSmallPlaneError(float depth_sigma_coeff, float depth_sigma_margin) const;
};
}  // namespace deplex