#pragma once

#include <Eigen/Core>
#include <cstdint>

#include "algorithm/config.h"

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
  Eigen::Vector3d const& getNormal() const { return _stats._normal; }
  Eigen::Vector3d const& getMean() const { return _stats._mean; }
  double getScore() const { return _stats._score; }
  double getMSE() const { return _stats._mse; }
  double getD() const { return _stats._d; }

 private:
  struct Stats {
    friend class CellSegment;

   public:
    Stats();
    Stats(Eigen::VectorXf const& X, Eigen::VectorXf const& Y,
          Eigen::VectorXf const& Z);

   private:
    void makePCA();
    Eigen::Vector3d _normal;
    Eigen::Vector3d _mean;
    double _d;
    double _score;
    double _mse;
    float _x, _y, _z, _xx, _yy, _zz, _xy, _xz, _yz;
    int32_t _nr_pts;
  } _stats;
  Eigen::MatrixXf const* const _ptr_pcd_array;
  config::Config const* const _config;
  int32_t _nr_pts_per_cell;
  int32_t _cell_width;
  int32_t _cell_height;
  int32_t _offset;
  bool isValidPoints() const;
  bool isDepthContinuous() const;
  bool _isHorizontalContinuous(Eigen::MatrixXf const& cell_z) const;
  bool _isVerticalContinuous(Eigen::MatrixXf const& cell_z) const;
  void initStats();
};
}  // namespace deplex