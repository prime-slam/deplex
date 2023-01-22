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
#include <memory>
#include <string>

namespace deplex {
namespace utils {
class Image {
 public:
  Image();
  Image(std::string const& image_path);

  int32_t getWidth() const;

  int32_t getHeight() const;

  Eigen::MatrixXf toPointCloud(Eigen::Matrix3f const& intrinsics) const;

 private:
  std::unique_ptr<unsigned short> image_;
  int32_t width_;
  int32_t height_;
};
}  // namespace utils
}  // namespace deplex