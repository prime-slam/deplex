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
#include "../deplex/src/deplex/cell_segment_stat.h"

#include "globals.hpp"

namespace deplex {
namespace {
TEST(TUMPlaneExtraction, Refinement) {
  auto config = config::Config(test_globals::tum::config);
  auto config_no_refinement = config;
  config.ransac_refinement = true;

  auto image = utils::DepthImage(test_globals::tum::sample_image);
  auto points = image.toPointCloud(utils::readIntrinsics(test_globals::tum::intrinsics));

  auto algorithm = PlaneExtractor(image.getHeight(), image.getWidth(), config);
  auto refined_labels = algorithm.process(points);

  algorithm = PlaneExtractor(image.getHeight(), image.getWidth(), config_no_refinement);
  auto coarse_labels = algorithm.process(points);

  Eigen::MatrixX3f coarse_plane(std::count(coarse_labels.begin(), coarse_labels.end(), 1), 3);
  int row_id = 0;
  for (int i = 0; i < coarse_labels.size(); ++i) {
    if (coarse_labels[i] == 1) {
      coarse_plane.row(row_id++) = points.row(i);
    }
  }
  Eigen::MatrixX3f refined_plane(std::count(refined_labels.begin(), refined_labels.end(), 1), 3);
  row_id = 0;
  for (int i = 0; i < refined_labels.size(); ++i) {
    if (refined_labels[i] == 1) {
      refined_plane.row(row_id++) = points.row(i);
    }
  }

  deplex::CellSegmentStat coarse_stats(coarse_plane);
  deplex::CellSegmentStat refined_stats(refined_plane);
  ASSERT_TRUE(refined_stats.getMSE() < coarse_stats.getMSE());
}

TEST(ICLPlaneExtraction, Refinement) {
  auto config = config::Config(test_globals::icl::config);
  auto config_no_refinement = config;
  config.ransac_refinement = true;

  auto image = utils::DepthImage(test_globals::icl::sample_image);
  auto points = image.toPointCloud(utils::readIntrinsics(test_globals::icl::intrinsics));

  auto algorithm = PlaneExtractor(image.getHeight(), image.getWidth(), config);
  auto refined_labels = algorithm.process(points);

  algorithm = PlaneExtractor(image.getHeight(), image.getWidth(), config_no_refinement);
  auto coarse_labels = algorithm.process(points);

  Eigen::MatrixX3f coarse_plane(std::count(coarse_labels.begin(), coarse_labels.end(), 1), 3);
  int row_id = 0;
  for (int i = 0; i < coarse_labels.size(); ++i) {
    if (coarse_labels[i] == 1) {
      coarse_plane.row(row_id++) = points.row(i);
    }
  }
  Eigen::MatrixX3f refined_plane(std::count(refined_labels.begin(), refined_labels.end(), 1), 3);
  row_id = 0;
  for (int i = 0; i < refined_labels.size(); ++i) {
    if (refined_labels[i] == 1) {
      refined_plane.row(row_id++) = points.row(i);
    }
  }

  deplex::CellSegmentStat coarse_stats(coarse_plane);
  deplex::CellSegmentStat refined_stats(refined_plane);
  ASSERT_TRUE(refined_stats.getMSE() < coarse_stats.getMSE());
}


}  // namespace
}  // namespace deplex
