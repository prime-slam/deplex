#pragma once

#include <Eigen/Core>

namespace deplex {
class CellSegmentStat {
 public:
  CellSegmentStat();

  explicit CellSegmentStat(Eigen::MatrixXf const& points);

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
  Eigen::Vector3f coord_sum_;
  Eigen::Matrix3f variance_;
  Eigen::Vector3f mean_;
  Eigen::Vector3f normal_;
};

}  // namespace deplex