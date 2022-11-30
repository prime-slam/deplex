#include "cell_segment_stat.h"

#include <Eigen/Eigenvalues>

namespace deplex {
CellSegmentStat::CellSegmentStat() : mse_(-1) {}

CellSegmentStat::CellSegmentStat(Eigen::VectorXf const& X, Eigen::VectorXf const& Y, Eigen::VectorXf const& Z)
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
  fitPlane();
}

Eigen::Vector3d const& CellSegmentStat::getNormal() const { return normal_; };

Eigen::Vector3d const& CellSegmentStat::getMean() const { return mean_; };

double CellSegmentStat::getScore() const { return score_; };

double CellSegmentStat::getMSE() const { return mse_; };

double CellSegmentStat::getD() const { return d_; };

CellSegmentStat& CellSegmentStat::operator+=(CellSegmentStat const& other) {
  x_ += other.x_;
  y_ += other.y_;
  z_ += other.z_;
  xx_ += other.xx_;
  yy_ += other.yy_;
  zz_ += other.zz_;
  xy_ += other.xy_;
  xz_ += other.xz_;
  yz_ += other.yz_;
  nr_pts_ += other.nr_pts_;
  return *this;
}

void CellSegmentStat::fitPlane() {
  mean_ = Eigen::Vector3d(x_, y_, z_) / nr_pts_;

  Eigen::Matrix3d cov{{xx_ - x_ * x_ / nr_pts_, xy_ - x_ * y_ / nr_pts_, xz_ - x_ * z_ / nr_pts_},
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
}  // namespace deplex
