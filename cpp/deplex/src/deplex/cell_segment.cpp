#include "cell_segment.h"

#include <Eigen/Eigenvalues>

namespace deplex {
CellSegment& CellSegment::operator+=(CellSegment const& other) {
  stats_ += other.stats_;
  return *this;
}

CellSegmentStat const& CellSegment::getStat() const { return stats_; };

CellSegment::CellSegment(int32_t cell_id, int32_t cell_width, int32_t cell_height, Eigen::MatrixXf const& pcd_array,
                         config::Config const& config)
    : ptr_pcd_array_(&pcd_array),
      config_(&config),
      nr_pts_per_cell_(cell_width * cell_height),
      cell_width_(cell_width),
      cell_height_(cell_height),
      offset_(cell_id * cell_width * cell_height) {}

bool CellSegment::isPlanar() {
  if (isValidPoints() && isDepthContinuous()) {
    initStats();
    float depth_sigma_coeff = config_->getFloat("depthSigmaCoeff");
    float depth_sigma_margin = config_->getFloat("depthSigmaMargin");
    float planar_threshold = depth_sigma_coeff * pow(stats_.getMean()[2], 2) + depth_sigma_margin;
    return stats_.getMSE() <= pow(planar_threshold, 2);
  }
  return false;
}

bool CellSegment::isValidPoints() const {
  Eigen::VectorXf cell_z = ptr_pcd_array_->block(offset_, 2, nr_pts_per_cell_, 1);

  Eigen::Index valid_pts_threshold = nr_pts_per_cell_ / config_->getInt("minPtsPerCell");
  Eigen::Index valid_pts = (cell_z.array() > 0).count();
  return valid_pts >= valid_pts_threshold;
}

bool CellSegment::isHorizontalContinuous(Eigen::MatrixXf const& cell_z) const {
  float depth_disc_threshold = config_->getFloat("depthDiscontinuityThreshold");
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

  return disc_count < config_->getInt("maxNumberDepthDiscontinuity");
}

bool CellSegment::isVerticalContinuous(Eigen::MatrixXf const& cell_z) const {
  float depth_disc_threshold = config_->getFloat("depthDiscontinuityThreshold");
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

  return disc_count < config_->getInt("maxNumberDepthDiscontinuity");
}

bool CellSegment::isDepthContinuous() const {
  Eigen::MatrixXf cell_z = ptr_pcd_array_->block(offset_, 2, nr_pts_per_cell_, 1).reshaped(cell_height_, cell_width_);

  return isHorizontalContinuous(cell_z) && isVerticalContinuous(cell_z);
}

void CellSegment::initStats() {
  Eigen::VectorXf cell_x = ptr_pcd_array_->block(offset_, 0, nr_pts_per_cell_, 1);
  Eigen::VectorXf cell_y = ptr_pcd_array_->block(offset_, 1, nr_pts_per_cell_, 1);
  Eigen::VectorXf cell_z = ptr_pcd_array_->block(offset_, 2, nr_pts_per_cell_, 1);

  stats_ = CellSegmentStat(cell_x, cell_y, cell_z);
}

void CellSegment::calculateStats() { stats_.fitPlane(); }
}  // namespace deplex