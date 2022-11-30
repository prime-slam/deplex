#pragma once

#include <Eigen/Core>

namespace deplex {
class CellSegmentStat {
 public:
  CellSegmentStat();

  CellSegmentStat(Eigen::VectorXf const& X, Eigen::VectorXf const& Y, Eigen::VectorXf const& Z);

  CellSegmentStat& operator+=(CellSegmentStat const& other);

  Eigen::Vector3d const& getNormal() const;

  Eigen::Vector3d const& getMean() const;

  double getScore() const;

  double getMSE() const;

  double getD() const;

  void fitPlane();

 private:
  Eigen::Vector3d normal_;
  Eigen::Vector3d mean_;
  double d_;
  double score_;
  double mse_;
  float x_, y_, z_, xx_, yy_, zz_, xy_, xz_, yz_;
  int32_t nr_pts_;
};

}  // namespace deplex