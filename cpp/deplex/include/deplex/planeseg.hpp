#pragma once

#include "config.hpp"

#include <Eigen/Dense>

namespace deplex {
class PlaneSeg {
 public:
  PlaneSeg(int32_t cell_id, int32_t cell_width, int32_t cell_height,
           Eigen::MatrixXf const& pcd_array, config::Config const& config);

  PlaneSeg(PlaneSeg const& other) = default;
  PlaneSeg& operator=(PlaneSeg const& other) = default;

  PlaneSeg& operator+=(PlaneSeg const& other);

  void calculateStats();
  bool isPlanar();
  Eigen::Vector3d const& getNormal() const { return _stats._normal; }
  Eigen::Vector3d const& getMean() const { return _stats._mean; }
  double getScore() const { return _stats._score; }
  double getMSE() const { return _stats._mse; }
  double getD() const { return _stats._d; }

 private:
  struct Stats {
    friend class PlaneSeg;

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

PlaneSeg& PlaneSeg::operator+=(PlaneSeg const& other) {
  _stats._x += other._stats._x;
  _stats._y += other._stats._y;
  _stats._z += other._stats._z;
  _stats._xx += other._stats._xx;
  _stats._yy += other._stats._yy;
  _stats._zz += other._stats._zz;
  _stats._xy += other._stats._xy;
  _stats._xz += other._stats._xz;
  _stats._yz += other._stats._yz;
  _stats._nr_pts += other._stats._nr_pts;
  return *this;
}

PlaneSeg::Stats::Stats() : _mse(-1) {}

PlaneSeg::Stats::Stats(Eigen::VectorXf const& X, Eigen::VectorXf const& Y,
                       Eigen::VectorXf const& Z)
    : _x(X.sum()),
      _y(Y.sum()),
      _z(Z.sum()),
      _xx(X.dot(X)),
      _yy(Y.dot(Y)),
      _zz(Z.dot(Z)),
      _xy(X.dot(Y)),
      _xz(X.dot(Z)),
      _yz(Y.dot(Z)),
      _nr_pts(X.size()) {
  makePCA();
}

void PlaneSeg::Stats::makePCA() {
  _mean = Eigen::Vector3d(_x, _y, _z) / _nr_pts;

  Eigen::Matrix3d cov{{_xx - _x * _x / _nr_pts, _xy - _x * _y / _nr_pts,
                       _xz - _x * _z / _nr_pts},
                      {0.0, _yy - _y * _y / _nr_pts, _yz - _y * _z / _nr_pts},
                      {0.0, 0.0, _zz - _z * _z / _nr_pts}};

  cov(1, 0) = cov(0, 1);
  cov(2, 0) = cov(0, 2);
  cov(2, 1) = cov(1, 2);

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
  Eigen::VectorXd v = es.eigenvectors().col(0);

  _d = -_mean.dot(v);
  // Enforce normal orientation
  _normal = (_d > 0 ? v : -v);
  _d = (_d > 0 ? _d : -_d);

  _mse = es.eigenvalues()[0] / _nr_pts;
  _score = es.eigenvalues()[1] / es.eigenvalues()[0];
}

PlaneSeg::PlaneSeg(int32_t cell_id, int32_t cell_width, int32_t cell_height,
                   Eigen::MatrixXf const& pcd_array,
                   config::Config const& config)
    : _ptr_pcd_array(&pcd_array),
      _config(&config),
      _nr_pts_per_cell(cell_width * cell_height),
      _cell_width(cell_width),
      _cell_height(cell_height),
      _offset(cell_id * cell_width * cell_height) {}

bool PlaneSeg::isPlanar() {
  if (isValidPoints() && isDepthContinuous()) {
    initStats();
    float depth_sigma_coeff = _config->getFloat("depthSigmaCoeff");
    float depth_sigma_margin = _config->getFloat("depthSigmaMargin");
    float planar_threshold =
        depth_sigma_coeff * pow(_stats._mean[2], 2) + depth_sigma_margin;
    return _stats._mse <= pow(planar_threshold, 2);
  }
  return false;
}

bool PlaneSeg::isValidPoints() const {
  Eigen::VectorXf cell_z =
      _ptr_pcd_array->block(_offset, 2, _nr_pts_per_cell, 1);

  Eigen::Index valid_pts_threshold =
      _nr_pts_per_cell / _config->getInt("minPtsPerCell");
  Eigen::Index valid_pts = (cell_z.array() > 0).count();
  return valid_pts >= valid_pts_threshold;
}

bool PlaneSeg::_isHorizontalContinuous(Eigen::MatrixXf const& cell_z) const {
  float depth_disc_threshold = _config->getFloat("depthDiscontinuityThreshold");
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

  return disc_count < _config->getInt("maxNumberDepthDiscontinuity");
}

bool PlaneSeg::_isVerticalContinuous(Eigen::MatrixXf const& cell_z) const {
  float depth_disc_threshold = _config->getFloat("depthDiscontinuityThreshold");
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

  return disc_count < _config->getInt("maxNumberDepthDiscontinuity");
}

bool PlaneSeg::isDepthContinuous() const {
  Eigen::MatrixXf cell_z =
      _ptr_pcd_array->block(_offset, 2, _nr_pts_per_cell, 1)
          .reshaped(_cell_height, _cell_width);

  return _isHorizontalContinuous(cell_z) && _isVerticalContinuous(cell_z);
}

void PlaneSeg::initStats() {
  Eigen::VectorXf cell_x =
      _ptr_pcd_array->block(_offset, 0, _nr_pts_per_cell, 1);
  Eigen::VectorXf cell_y =
      _ptr_pcd_array->block(_offset, 1, _nr_pts_per_cell, 1);
  Eigen::VectorXf cell_z =
      _ptr_pcd_array->block(_offset, 2, _nr_pts_per_cell, 1);

  _stats = Stats(cell_x, cell_y, cell_z);
}

void PlaneSeg::calculateStats() { _stats.makePCA(); }

}  // namespace deplex