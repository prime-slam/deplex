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
#include "deplex/utils/image.h"

#include <fstream>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_NO_FAILURE_STRINGS
#include "stb_image/stb_image.h"

namespace deplex {
namespace utils {
Image::Image() : image_(nullptr), width_(0), height_(0) {}

Image::Image(std::string const& image_path) : width_(0), height_(0) {
  int32_t actual_channels = 0;
  auto data_ptr = stbi_load_16(image_path.c_str(), &width_, &height_, &actual_channels, STBI_grey);
  if (data_ptr == nullptr) {
    throw std::runtime_error("Error: Couldn't read image " + image_path);
  }
  image_.reset(data_ptr);
}

int32_t Image::getWidth() const { return width_; }

int32_t Image::getHeight() const { return height_; }

Eigen::MatrixXf Image::toPointCloud(Eigen::Matrix3f const& intrinsics) const {
  Eigen::VectorXf column_indices = Eigen::VectorXf::LinSpaced(width_, 0.0, width_ - 1).replicate(height_, 1);
  Eigen::VectorXf row_indices =
      Eigen::VectorXf::LinSpaced(height_, 0.0, height_ - 1).replicate(1, width_).reshaped<Eigen::RowMajor>();

  Eigen::MatrixXf pcd_points(width_ * height_, 3);

  float fx = intrinsics.row(0)[0];
  float fy = intrinsics.row(1)[1];
  float cx = intrinsics.row(0)[2];
  float cy = intrinsics.row(1)[2];

  typedef std::remove_reference<decltype(*image_)>::type t_image;
  pcd_points.col(2) = Eigen::Map<Eigen::Vector<t_image, Eigen::Dynamic>>(image_.get(), width_ * height_).cast<float>();
  pcd_points.col(0) = (column_indices.array() - cx) * pcd_points.col(2).array() / fx;
  pcd_points.col(1) = (row_indices.array() - cy) * pcd_points.col(2).array() / fy;

  return pcd_points;
}

}  // namespace utils
}  // namespace deplex