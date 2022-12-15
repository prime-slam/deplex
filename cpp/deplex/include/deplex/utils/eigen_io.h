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
#pragma once

#include <Eigen/Core>

namespace deplex {
namespace utils {
const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
/**
 * Read point cloud points from file
 *
 * @param path Path to input file with cloud points [Nx3].
 * @param delimiter Symbol by which values in files are separated.
 * @returns Matrix[Nx3] with points.
 */
Eigen::MatrixXf readPointCloudCSV(std::string const& path, char delimiter = ',');

/**
 * Write point cloud points to file
 *
 * @param pcd_points Point cloud points [Nx3]
 * @param path Path to output file.
 */
void savePointCloudCSV(Eigen::MatrixXf const& pcd_points, std::string const& path);
}  // namespace utils
}  // namespace deplex::utils