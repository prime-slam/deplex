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
#include "deplex/utils/eigen_io.h"

#include <fstream>
#include <vector>

namespace deplex {
namespace utils {
Eigen::MatrixXf readPointCloudCSV(std::string const& path, char delimiter) {
  std::vector<float> points;

  std::ifstream file(path);
  std::string matrix_row;
  std::string matrix_entry;
  while (getline(file, matrix_row)) {
    std::stringstream ss(matrix_row);
    while (getline(ss, matrix_entry, delimiter)) {
      points.push_back(std::stof(matrix_entry));
    }
  }
  if (points.size() % 3 != 0) {
    throw std::runtime_error("Error reading file: Invalid points shape");
  }
  return Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>>(points.data(), points.size() / 3, 3);
}

void savePointCloudCSV(Eigen::MatrixXf const& pcd_points, std::string const& path) {
  std::ofstream file(path);
  file << pcd_points.format(CSVFormat);
}
}  // namespace utils
}  // namespace deplex