/**
 * Copyright (c) 2022, Arthur Saliou, Anastasiia Kornilova
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "cell_segment_stat.h"

#include <algorithm>
#include <limits>
#include <vector>

extern "C" {
#include <dsyevh3.h>
}

namespace deplex {
CellSegmentStat::CellSegmentStat() : nr_pts_(0), mse_(std::numeric_limits<float>::max()), score_(0) {}

CellSegmentStat::CellSegmentStat(Eigen::MatrixX3f const& points)
    : nr_pts_(points.rows()),
      coord_sum_(points.colwise().sum()),
      variance_(points.transpose() * points),
      mean_(coord_sum_ / nr_pts_) {
  fitPlane();
}

CellSegmentStat& CellSegmentStat::operator+=(CellSegmentStat const& other) {
  nr_pts_ += other.nr_pts_;
  coord_sum_ += other.coord_sum_;
  variance_ += other.variance_;
  mean_ = coord_sum_ / nr_pts_;
  return *this;
}

Eigen::Vector3f const& CellSegmentStat::getNormal() const { return normal_; };

Eigen::Vector3f const& CellSegmentStat::getMean() const { return mean_; };

float CellSegmentStat::getScore() const { return score_; };

float CellSegmentStat::getMSE() const { return mse_; };

float CellSegmentStat::getD() const { return d_; };

void CellSegmentStat::fitPlane() {
  Eigen::Matrix3f cov = variance_ - coord_sum_ * coord_sum_.transpose() / nr_pts_;
  double tmp_cov[3][3];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      tmp_cov[i][j] = cov.data()[3 * j + i];
    }
  }
  double eigenvectors[3][3];
  double eigenvalues[3];
  dsyevh3(tmp_cov, eigenvectors, eigenvalues);

  Eigen::Index min_es_ind = std::distance(eigenvalues, std::min_element(eigenvalues, eigenvalues + 3));
  Eigen::Index max_es_ind = std::distance(eigenvalues, std::max_element(eigenvalues, eigenvalues + 3));
  Eigen::Vector3f v(3);
  for (int i = 0; i < 3; ++i) {
    v[i] = static_cast<float>(eigenvectors[i][min_es_ind]);
  }

  d_ = -mean_.dot(v);
  // Enforce normal orientation
  normal_ = (d_ > 0 ? v : -v);
  d_ = (d_ > 0 ? d_ : -d_);

  mse_ = static_cast<float>(eigenvalues[min_es_ind] / nr_pts_);
  score_ = static_cast<float>(eigenvalues[max_es_ind] / (Eigen::Map<Eigen::Vector3d>(eigenvalues, 3).sum()));
}
}  // namespace deplex
