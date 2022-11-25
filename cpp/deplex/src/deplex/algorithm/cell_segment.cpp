#include "deplex/algorithm/cell_segment.h"

#include <Eigen/Eigenvalues>

namespace deplex {
CellSegment& CellSegment::operator+=(CellSegment const& other) {
  stats_.x_ += other.stats_.x_;
  stats_.y_ += other.stats_.y_;
  stats_.z_ += other.stats_.z_;
  stats_.xx_ += other.stats_.xx_;
  stats_.yy_ += other.stats_.yy_;
  stats_.zz_ += other.stats_.zz_;
  stats_.xy_ += other.stats_.xy_;
  stats_.xz_ += other.stats_.xz_;
  stats_.yz_ += other.stats_.yz_;
  stats_.nr_pts_ += other.stats_.nr_pts_;
  return *this;
}

CellSegment::Stats::Stats() : mse_(-1) {}

CellSegment::Stats::Stats(Eigen::VectorXf const& X, Eigen::VectorXf const& Y,
                          Eigen::VectorXf const& Z)
    : x_(X.sum()),
      y_(Y.sum()),
      z_(Z.sum()),
      xx_(X.dot(X)),
      yy_(Y.dot(Y)),
      zz_(Z.dot(Z)),
      xy_(X.dot(Y)),
      xz_(X.dot(Z)),
      yz_(Y.dot(Z)),
      nr_pts_(X.size()) {
  makePCA();
}

void CellSegment::Stats::makePCA() {
  mean_ = Eigen::Vector3d(x_, y_, z_) / nr_pts_;

  Eigen::Matrix3d cov{{xx_ - x_ * x_ / nr_pts_, xy_ - x_ * y_ / nr_pts_,
                       xz_ - x_ * z_ / nr_pts_},
                      {0.0, yy_ - y_ * y_ / nr_pts_, yz_ - y_ * z_ / nr_pts_},
                      {0.0, 0.0, zz_ - z_ * z_ / nr_pts_}};

  cov(1, 0) = cov(0, 1);
  cov(2, 0) = cov(0, 2);
  cov(2, 1) = cov(1, 2);

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
  Eigen::VectorXd v = es.eigenvectors().col(0);

  d_ = -mean_.dot(v);
  // Enforce normal orientation
  normal_ = (d_ > 0 ? v : -v);
  d_ = (d_ > 0 ? d_ : -d_);

  mse_ = es.eigenvalues()[0] / nr_pts_;
  score_ = es.eigenvalues()[1] / es.eigenvalues()[0];
}

CellSegment::CellSegment(int32_t cell_id, int32_t cell_width,
                         int32_t cell_height, Eigen::MatrixXf const& pcd_array,
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
    float planar_threshold =
        depth_sigma_coeff * pow(stats_.mean_[2], 2) + depth_sigma_margin;
    return stats_.mse_ <= pow(planar_threshold, 2);
  }
  return false;
}

bool CellSegment::isValidPoints() const {
  Eigen::VectorXf cell_z =
      ptr_pcd_array_->block(offset_, 2, nr_pts_per_cell_, 1);

  Eigen::Index valid_pts_threshold =
      nr_pts_per_cell_ / config_->getInt("minPtsPerCell");
  Eigen::Index valid_pts = (cell_z.array() > 0).count();
  return valid_pts >= valid_pts_threshold;
}

bool CellSegment::_isHorizontalContinuous(Eigen::MatrixXf const& cell_z) const {
  float depth_disc_threshold = config_->getFloat("depthDiscontinuityThreshold");
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

  return disc_count < config_->getInt("maxNumberDepthDiscontinuity");
}

bool CellSegment::_isVerticalContinuous(Eigen::MatrixXf const& cell_z) const {
  float depth_disc_threshold = config_->getFloat("depthDiscontinuityThreshold");
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

  return disc_count < config_->getInt("maxNumberDepthDiscontinuity");
}

bool CellSegment::isDepthContinuous() const {
  Eigen::MatrixXf cell_z =
      ptr_pcd_array_->block(offset_, 2, nr_pts_per_cell_, 1)
          .reshaped(cell_height_, cell_width_);

  return _isHorizontalContinuous(cell_z) && _isVerticalContinuous(cell_z);
}

void CellSegment::initStats() {
  Eigen::VectorXf cell_x =
      ptr_pcd_array_->block(offset_, 0, nr_pts_per_cell_, 1);
  Eigen::VectorXf cell_y =
      ptr_pcd_array_->block(offset_, 1, nr_pts_per_cell_, 1);
  Eigen::VectorXf cell_z =
      ptr_pcd_array_->block(offset_, 2, nr_pts_per_cell_, 1);

  stats_ = Stats(cell_x, cell_y, cell_z);
}

void CellSegment::calculateStats() { stats_.makePCA(); }
}  // namespace deplex