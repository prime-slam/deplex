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

#include <deplex/config.h>
#include <deplex/plane_extractor.h>
#include <deplex/utils/depth_image.h>
#include <deplex/utils/eigen_io.h>

#include "globals.hpp"

namespace deplex {
namespace {
TEST(TUMPlaneExtraction, DefaultConfigExtraction) {
  auto image = utils::DepthImage(test_globals::tum::sample_image);
  auto algorithm = PlaneExtractor(image.getHeight(), image.getWidth());

  auto labels = algorithm.process(image.toPointCloud(utils::readIntrinsics(test_globals::tum::intrinsics)));
  ASSERT_EQ(labels.maxCoeff(), 34);
}

TEST(TUMPlaneExtraction, ZeroLeadingConfigExtraction) {
  auto config = config::Config(test_globals::tum::config);
  config.min_region_planarity_score = 5000;

  auto image = utils::DepthImage(test_globals::tum::sample_image);
  auto algorithm = PlaneExtractor(image.getHeight(), image.getWidth(), config);
  auto points = image.toPointCloud(utils::readIntrinsics(test_globals::tum::intrinsics));
  auto labels = algorithm.process(points);
  ASSERT_TRUE(labels.isZero());
  ASSERT_EQ(points.rows(), labels.size());
}

TEST(TUMPlaneExtraction, ZeroPatchSize) {
  auto config = config::Config(test_globals::tum::config);
  config.patch_size = 0;

  auto image = utils::DepthImage(test_globals::tum::sample_image);
  ASSERT_THROW(PlaneExtractor(image.getHeight(), image.getWidth(), config), std::runtime_error);
}

TEST(TUMPlaneExtraction, EnormousPatchSize) {
  auto config = config::Config(test_globals::tum::config);
  config.patch_size = 1e6;

  auto image = utils::DepthImage(test_globals::tum::sample_image);
  auto algorithm = PlaneExtractor(image.getHeight(), image.getWidth(), config);
  auto points = image.toPointCloud(utils::readIntrinsics(test_globals::tum::intrinsics));
  auto labels = algorithm.process(points);
  ASSERT_TRUE(labels.isZero());
  ASSERT_EQ(points.rows(), labels.size());
}

TEST(InvalidInput, ZeroValuePoints) {
  auto width = 640, height = 480;
  auto algorithm = PlaneExtractor(height, width);
  auto points = Eigen::MatrixX3f::Zero(width * height, 3);
  auto labels = algorithm.process(points);
  ASSERT_TRUE(labels.isZero());
  ASSERT_EQ(points.rows(), labels.size());
}

TEST(InvalidInput, EmptyPoints) {
  auto width = 640, height = 480;
  auto algorithm = PlaneExtractor(height, width);
  auto points = Eigen::MatrixX3f();
  ASSERT_THROW(algorithm.process(points), std::runtime_error);
}

TEST(InvalidInput, WrongShape) {
  auto width = 640, height = 480;
  auto algorithm = PlaneExtractor(height / 2, width / 2);
  auto points = Eigen::MatrixX3f::Zero(width * height, 3);
  ASSERT_THROW(algorithm.process(points), std::runtime_error);
}

}  // namespace
}  // namespace deplex
