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

#include <memory>

#include <Eigen/Core>

#include "deplex/config.h"

namespace deplex {
/**
 * Algorithm for plane extraction from RGB-D data.
 */
class PlaneExtractor {
 public:
  /**
   * PlaneExtractor constructor.
   *
   * @param image_height Image height in pixels.
   * @param image_width Image width in pixels.
   * @param config Parameters of plane extraction algorithm.
   */
  PlaneExtractor(int32_t image_height, int32_t image_width, config::Config config = config::Config());
  ~PlaneExtractor();

  /**
   * Extract planes from given image.
   *
   * @param pcd_array Points matrix [Nx3] of ORGANIZED point cloud
   * i.e. points that refer to organized image structure.
   * @returns 1D Array, where i-th value is plane number to which refers i-th point of point cloud.
   * 0-value label refers to non-planar segment.
   */
  Eigen::VectorXi process(Eigen::MatrixX3f const& pcd_array);

  std::vector<Eigen::Vector3d> GetExecutionTime();

  PlaneExtractor(PlaneExtractor&& op) noexcept;
  PlaneExtractor& operator=(PlaneExtractor&& op) noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace deplex