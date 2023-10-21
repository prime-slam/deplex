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
#include "deplex/utils/depth_image.h"

#include <fstream>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_NO_FAILURE_STRINGS
#include "stb_image/stb_image.h"

namespace deplex {
namespace utils {
DepthImage::DepthImage() : image_(nullptr), width_(0), height_(0) {}

DepthImage::DepthImage(std::string const& image_path) : width_(0), height_(0) {
  int32_t actual_channels = 0;
  auto data_ptr = stbi_load_16(image_path.c_str(), &width_, &height_, &actual_channels, STBI_grey);
  if (data_ptr == nullptr) {
    throw std::runtime_error("Error: Couldn't read image " + image_path);
  }
  image_.reset(data_ptr);

  column_indices_ = Eigen::VectorXf::LinSpaced(width_, 0.0, width_ - 1).replicate(height_, 1);
  row_indices_ = Eigen::VectorXf::LinSpaced(height_, 0.0, height_ - 1).replicate(1, width_).reshaped<Eigen::RowMajor>();
}

void DepthImage::reset(std::string const& image_path) {
  int32_t actual_channels = 0;
  auto data_ptr = stbi_load_16(image_path.c_str(), &width_, &height_, &actual_channels, STBI_grey);
  if (data_ptr == nullptr) {
    throw std::runtime_error("Error: Couldn't read image " + image_path);
  }
  image_.reset(data_ptr);
}

int32_t DepthImage::getWidth() const { return width_; }

int32_t DepthImage::getHeight() const { return height_; }

Eigen::MatrixX3f DepthImage::toPointCloud(Eigen::Matrix3f const& intrinsics) const {
  Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> pcd_points(width_ * height_, 3);

  float fx = intrinsics.row(0)[0];
  float cx = intrinsics.row(0)[2];
  float fy = intrinsics.row(1)[1];
  float cy = intrinsics.row(1)[2];

  typedef std::remove_reference<decltype(*image_)>::type t_image;
  pcd_points.col(2) = Eigen::Map<Eigen::Vector<t_image, Eigen::Dynamic>>(image_.get(), width_ * height_).cast<float>();

#pragma omp parallel default(none) shared(pcd_points, column_indices, row_indices, cx, cy, fx, fy)
  {
#pragma omp sections
    {
#pragma omp section
      { pcd_points.col(0) = (column_indices_.array() - cx) * pcd_points.col(2).array() / fx; }
#pragma omp section
      { pcd_points.col(1) = (row_indices_.array() - cy) * pcd_points.col(2).array() / fy; }
    }
  }

  return pcd_points;
}
}  // namespace utils
}  // namespace deplex