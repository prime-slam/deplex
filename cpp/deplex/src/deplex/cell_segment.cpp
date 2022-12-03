#include "cell_segment.h"

#include <Eigen/Eigenvalues>

namespace deplex {
CellSegment::CellSegment(Eigen::MatrixXf const& cell_points, config::Config const& config)
    : config_(&config), is_planar_(false) {
  bool is_valid = isValidPoints(cell_points) && isDepthContinuous(cell_points);
  if (!is_valid) return;
  stats_ = CellSegmentStat(cell_points);
  is_planar_ = isFittingMSE();
}

CellSegment& CellSegment::operator+=(CellSegment const& other) {
  stats_ += other.stats_;
  return *this;
}

CellSegmentStat const& CellSegment::getStat() const { return stats_; };

bool CellSegment::isPlanar() const { return is_planar_; }

void CellSegment::calculateStats() { stats_.fitPlane(); }

bool CellSegment::isValidPoints(Eigen::MatrixXf const& cell_points) const {
  Eigen::Index valid_pts_threshold = cell_points.size() / config_->getInt("minPtsPerCell");
  Eigen::Index valid_pts = (cell_points.col(2).array() > 0).count();
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

bool CellSegment::isDepthContinuous(Eigen::MatrixXf const& cell_points) const {
  auto cell_height = config_->getInt("patchSize");
  auto cell_width = config_->getInt("patchSize");
  Eigen::MatrixXf cell_z = cell_points.col(2).reshaped(cell_height, cell_width);

  return isHorizontalContinuous(cell_z) && isVerticalContinuous(cell_z);
}

bool CellSegment::isFittingMSE() const {
  if (stats_.getMSE() < 0) return false;
  float depth_sigma_coeff = config_->getFloat("depthSigmaCoeff");
  float depth_sigma_margin = config_->getFloat("depthSigmaMargin");
  float planar_threshold = depth_sigma_coeff * pow(stats_.getMean()[2], 2) + depth_sigma_margin;
  return stats_.getMSE() <= pow(planar_threshold, 2);
}
}  // namespace deplex