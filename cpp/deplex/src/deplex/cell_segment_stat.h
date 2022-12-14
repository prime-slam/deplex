#pragma once

#include <Eigen/Core>

namespace deplex {
class CellSegmentStat {
 public:
  CellSegmentStat();

  explicit CellSegmentStat(Eigen::MatrixXd const& points);

  CellSegmentStat& operator+=(CellSegmentStat const& other);

  Eigen::Vector3f const& getNormal() const;

  Eigen::Vector3f const& getMean() const;

  float getScore() const;

  float getMSE() const;

  float getD() const;

  void fitPlane();

 private:
  float d_;
  float score_;
  float mse_;
  int32_t nr_pts_;
  Eigen::Vector3d coord_sum_;
  Eigen::Matrix3d variance_;
  Eigen::Vector3f mean_;
  Eigen::Vector3f normal_;
};

}  // namespace deplex