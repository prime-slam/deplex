#include "cell_segment_stat.h"

#include <limits>

#include <Eigen/Eigenvalues>

namespace deplex {
CellSegmentStat::CellSegmentStat() : nr_pts_(0), mse_(std::numeric_limits<float>::max()), score_(0) {}

CellSegmentStat::CellSegmentStat(Eigen::MatrixXd const& points)
    : nr_pts_(points.rows()),
      coord_sum_(points.colwise().sum()),
      variance_(points.transpose() * points),
      mean_(coord_sum_.cast<float>() / nr_pts_) {
  fitPlane();
}

CellSegmentStat& CellSegmentStat::operator+=(CellSegmentStat const& other) {
  nr_pts_ += other.nr_pts_;
  coord_sum_ += other.coord_sum_;
  variance_ += other.variance_;
  mean_ = coord_sum_.cast<float>() / nr_pts_;
  return *this;
}

Eigen::Vector3f const& CellSegmentStat::getNormal() const { return normal_; };

Eigen::Vector3f const& CellSegmentStat::getMean() const { return mean_; };

float CellSegmentStat::getScore() const { return score_; };

float CellSegmentStat::getMSE() const { return mse_; };

float CellSegmentStat::getD() const { return d_; };

void CellSegmentStat::fitPlane() {
  Eigen::Matrix3d cov = variance_ - coord_sum_ * coord_sum_.transpose() / nr_pts_;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);

  Eigen::Index max_es_ind = 0, min_es_ind = 0;
  es.eigenvalues().maxCoeff(&max_es_ind);
  es.eigenvalues().minCoeff(&min_es_ind);

  Eigen::VectorXf v = es.eigenvectors().col(min_es_ind).cast<float>();

  d_ = static_cast<float>(-mean_.cast<float>().dot(v));
  // Enforce normal orientation
  normal_ = (d_ > 0 ? v : -v);
  d_ = (d_ > 0 ? d_ : -d_);

  mse_ = es.eigenvalues()[min_es_ind] / nr_pts_;
  score_ = es.eigenvalues()[max_es_ind] / (es.eigenvalues().sum());
}
}  // namespace deplex
