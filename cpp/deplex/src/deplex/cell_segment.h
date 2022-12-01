#pragma once

#include <cstdint>

#include "cell_segment_stat.h"
#include "deplex/config.h"

namespace deplex {
class CellSegment {
 public:
  CellSegment(int32_t cell_id, int32_t cell_width, int32_t cell_height, Eigen::MatrixXf const& pcd_array,
              config::Config const& config);

  CellSegment(CellSegment const& other) = default;

  CellSegment& operator=(CellSegment const& other) = default;

  CellSegment& operator+=(CellSegment const& other);

  CellSegmentStat const& getStat() const;

  void calculateStats();

  bool isPlanar();

 private:
  CellSegmentStat stats_;
  Eigen::MatrixXf const* const ptr_pcd_array_;
  config::Config const* const config_;
  int32_t nr_pts_per_cell_;
  int32_t cell_width_;
  int32_t cell_height_;
  int32_t offset_;

  bool isValidPoints() const;

  bool isDepthContinuous() const;

  bool isHorizontalContinuous(Eigen::MatrixXf const& cell_z) const;

  bool isVerticalContinuous(Eigen::MatrixXf const& cell_z) const;
};
}  // namespace deplex