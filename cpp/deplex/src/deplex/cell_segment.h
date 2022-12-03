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
  config::Config const* const config_;
  bool is_planar_;

  bool isValidPoints(Eigen::MatrixXf const& cell_points) const;

  bool isDepthContinuous(Eigen::MatrixXf const& cell_points) const;

  bool isHorizontalContinuous(Eigen::MatrixXf const& cell_z) const;

  bool isVerticalContinuous(Eigen::MatrixXf const& cell_z) const;

  bool isFittingMSE() const;
};
}  // namespace deplex