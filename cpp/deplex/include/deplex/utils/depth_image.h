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
#pragma once

#include <Eigen/Core>
#include <memory>
#include <string>

namespace deplex {
namespace utils {
class DepthImage {
 public:
  DepthImage();

  /**
   * DepthImage constructor.
   *
   * @param image_path Input path to a depth-image in .png format
   */
  DepthImage(std::string const& image_path);

  int32_t getWidth() const;

  int32_t getHeight() const;

  /**
   * Map 2D depth-image to a 3D organized Point Cloud
   *
   * @param intrinsics Matrix[3x3] camera intrinsics matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
   * @returns Point Cloud Matrix[Nx3] of points (row-major storage)
   */
  Eigen::MatrixX3f toPointCloud(Eigen::Matrix3f const& intrinsics) const;

  void reset(std::string const& image_path);

 private:
  std::unique_ptr<unsigned short> image_;
  int32_t width_;
  int32_t height_;

  Eigen::VectorXf column_indices_;
  Eigen::VectorXf row_indices_;
};
}  // namespace utils
}  // namespace deplex