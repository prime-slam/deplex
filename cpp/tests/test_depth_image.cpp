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
#include <gtest/gtest.h>

#include <deplex/utils/utils.h>

#include "globals.hpp"

namespace deplex {
namespace {
TEST(ReadImage, ValidImage){
  auto image = utils::DepthImage(test_globals::tum::sample_image);
  ASSERT_EQ(image.getWidth(), 640);
  ASSERT_EQ(image.getHeight(), 480);
}

TEST(ReadImage, InvalidPath){
  ASSERT_THROW(utils::DepthImage("__INVALID_PATH"), std::runtime_error);
}

TEST(ReadImage, InvalidImageExtension){
  ASSERT_THROW(utils::DepthImage(test_globals::invalid::invalid_extension_image), std::runtime_error);
}

TEST(ReadImage, InvalidImage){
  ASSERT_THROW(utils::DepthImage(test_globals::invalid::invalid_depth_image), std::runtime_error);
}

TEST(TransformPointCloud, ValidTransform){
  auto intrinsics = utils::readIntrinsics(test_globals::tum::intrinsics);
  auto image = utils::DepthImage(test_globals::tum::sample_image);
  auto points = image.toPointCloud(intrinsics);
  ASSERT_EQ(points.col(2).maxCoeff(), 46655);
  ASSERT_EQ(points.col(2).minCoeff(), 0);
}
}
}  // namespace deplex