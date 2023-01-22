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
#include <gtest/gtest.h>

#include <deplex/config.h>
#include <deplex/plane_extractor.h>
#include <deplex/utils/eigen_io.h>
#include <deplex/utils/image.h>

#include "globals.hpp"

namespace deplex {
namespace {
TEST(TUMPlaneExtraction, DefaultConfigExtraction) {
  auto image = utils::Image(test_globals::tum::sample_image);
  auto algorithm = PlaneExtractor(image.getHeight(), image.getWidth());

  auto labels = algorithm.process(image.toPointCloud(utils::readIntrinsics(test_globals::tum::intrinsics)));
  ASSERT_GE(labels.maxCoeff(), 0);
  ASSERT_LE(labels.maxCoeff(), 30);
}

TEST(TUMPlaneExtraction, ZeroLeadingConfigExtraction) {
  auto config = config::Config(test_globals::tum::config);
  config.updateValue("minRegionPlanarityScore", "5000");

  auto image = utils::Image(test_globals::tum::sample_image);
  auto algorithm = PlaneExtractor(image.getHeight(), image.getWidth(), config);
  auto points = image.toPointCloud(utils::readIntrinsics(test_globals::tum::intrinsics));
  auto labels = algorithm.process(points);
  ASSERT_TRUE(labels.isZero());
  ASSERT_EQ(points.rows(), labels.size());
}

}  // namespace
}  // namespace deplex
