/**
 * Copyright 2022 prime-slam
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
#include "normals_histogram.h"

#include <cmath>

namespace deplex {
NormalsHistogram::NormalsHistogram(int32_t nr_bins_per_coord, Eigen::MatrixX3f const& normals)
    : nr_bins_per_coord_(nr_bins_per_coord),
      nr_points_(static_cast<int32_t>(normals.rows())),
      hist_(nr_bins_per_coord * nr_bins_per_coord, 0),
      bins_(static_cast<int32_t>(normals.rows()), -1) {
  // Set limits
  // Polar angle [0 pi]
  double min_X(0), max_X(M_PI);
  // Azimuth angle [-pi pi]
  double min_Y(-M_PI), max_Y(M_PI);

  for (Eigen::Index i = 0; i < normals.rows(); ++i) {
    if (!normals.row(i).isZero()) {
      Eigen::Vector3f normal = normals.row(i);
      double n_proj_norm = sqrt(pow(normal[0], 2) + pow(normal[1], 2));
      double polar_angle = acos(-normal[2]);
      double azimuth_angle = atan2(normal[0] / n_proj_norm, normal[1] / n_proj_norm);

      auto X_q = static_cast<int32_t>((nr_bins_per_coord_ - 1) * (polar_angle - min_X) / (max_X - min_X));
      int32_t Y_q = 0;
      if (X_q > 0) {
        Y_q = static_cast<int32_t>((nr_bins_per_coord_ - 1) * (azimuth_angle - min_Y) / (max_Y - min_Y));
      }
      int32_t bin = Y_q * nr_bins_per_coord_ + X_q;
      bins_[i] = bin;
      ++hist_[bin];
    }
  }
}

std::vector<int32_t> NormalsHistogram::getPointsFromMostFrequentBin() const {
  std::vector<int32_t> point_ids;

  auto most_frequent_bin = std::max_element(hist_.begin(), hist_.end());
  int32_t max_nr_occurrences = *most_frequent_bin;
  size_t max_bin_id = std::distance(hist_.begin(), most_frequent_bin);

  if (max_nr_occurrences > 0) {
    for (int32_t i = 0; i < nr_points_; ++i) {
      if (bins_[i] == max_bin_id) {
        point_ids.push_back(i);
      }
    }
  }

  return point_ids;
}

void NormalsHistogram::removePoint(int32_t point_id) {
  --hist_[bins_[point_id]];
  bins_[point_id] = -1;
}
}  // namespace deplex