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

namespace deplex::utils {
/**
  * Read image to organized point cloud points.
  *
  * @param image_path Path to RGB-D image.
  * @param K Camera intrinsics matrix.
  * @returns Organized point cloud points.
 */
Eigen::MatrixXf readImage(std::string const& image_path, Eigen::Matrix3f const& K);

/**
  * Read image to organized point cloud points.
  *
  * @param image_path Path to RGB-D image.
  * @param intrinsics_path Path to file with intrinsics matrix.
  * @returns Organized point cloud points.
 */
Eigen::MatrixXf readImage(std::string const& image_path, std::string const& intrinsics_path);
}  // namespace deplex::utils