#pragma once

#include <Eigen/Core>
#include <cstdint>

#include "deplex/algorithm/config.h"

namespace deplex {
class CellSegment {
 public:
  CellSegment(int32_t cell_id, int32_t cell_width, int32_t cell_height,
              Eigen::MatrixXf const& pcd_array, config::Config const& config);

  CellSegment(CellSegment const& other) = default;

  CellSegment& operator=(CellSegment const& other) = default;

  CellSegment& operator+=(CellSegment const& other);

  void calculateStats();

  bool isPlanar();

  Eigen::Vector3d const& getNormal() const { return stats_.normal_; }

  Eigen::Vector3d const& getMean() const { return stats_.mean_; }

  double getScore() const { return stats_.score_; }

  double getMSE() const { return stats_.mse_; }

  double getD() const { return stats_.d_; }

 private:
  struct Stats {
    friend class CellSegment;

   public:
    Stats();

    Stats(Eigen::VectorXf const& X, Eigen::VectorXf const& Y, Eigen::VectorXf const& Z);

   private:
    Eigen::Vector3d normal_;
    Eigen::Vector3d mean_;
    double d_;
    double score_;
    double mse_;
    float x_, y_, z_, xx_, yy_, zz_, xy_, xz_, yz_;
    int32_t nr_pts_;

    void makePCA();
  } stats_;
  Eigen::MatrixXf const* const ptr_pcd_array_;
  config::Config const* const config_;
  int32_t nr_pts_per_cell_;
  int32_t cell_width_;
  int32_t cell_height_;
  int32_t offset_;

  bool isValidPoints() const;

  bool isDepthContinuous() const;

  bool _isHorizontalContinuous(Eigen::MatrixXf const& cell_z) const;

  bool _isVerticalContinuous(Eigen::MatrixXf const& cell_z) const;

  void initStats();
};
}  // namespace deplex