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
#include <deplex/cell_segment_stat.h>

#include "globals.hpp"

namespace deplex {
namespace {

float getPlaneMSE (PlaneExtractor& algorithm, Eigen::MatrixX3f points, int32_t label) {
  auto labels = algorithm.process(points);
  Eigen::MatrixX3f plane(std::count(labels.begin(), labels.end(), label), 3);
  int row_id = 0;
  for (int i = 0; i < labels.size(); ++i) {
    if (labels[i] == label) {
      plane.row(row_id++) = points.row(i);
    }
  }

  deplex::CellSegmentStat stats(plane);
  return stats.getMSE();
}

TEST(TUMPlaneExtraction, Refinement) {
  auto config = config::Config(test_globals::tum::config);
  auto config_no_refinement = config;
  config.ransac_refinement = true;

  auto image = utils::DepthImage(test_globals::tum::sample_image);
  auto points = image.toPointCloud(utils::readIntrinsics(test_globals::tum::intrinsics));

  auto algorithm = PlaneExtractor(image.getHeight(), image.getWidth(), config);
  float refined_MSE = getPlaneMSE(algorithm, points, 1);

  algorithm = PlaneExtractor(image.getHeight(), image.getWidth(), config_no_refinement);
  float coarse_MSE = getPlaneMSE(algorithm, points, 1);

  ASSERT_LE(refined_MSE, coarse_MSE);
}

TEST(ICLPlaneExtraction, Refinement) {
  auto config = config::Config(test_globals::icl::config);
  auto config_no_refinement = config;
  config.ransac_refinement = true;

  auto image = utils::DepthImage(test_globals::icl::sample_image);
  auto points = image.toPointCloud(utils::readIntrinsics(test_globals::icl::intrinsics));

  auto algorithm = PlaneExtractor(image.getHeight(), image.getWidth(), config);
  float refined_MSE = getPlaneMSE(algorithm, points, 1);

  algorithm = PlaneExtractor(image.getHeight(), image.getWidth(), config_no_refinement);
  float coarse_MSE = getPlaneMSE(algorithm, points, 1);

  ASSERT_LE(refined_MSE, coarse_MSE);
}


}  // namespace
}  // namespace deplex
