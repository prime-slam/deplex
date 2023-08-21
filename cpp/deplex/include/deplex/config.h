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

#include <string>
#include <unordered_map>
#include <vector>

namespace deplex {
namespace config {
/**
 * Wrapper class for PlaneExtractor algorithm parameters.
 *
 * One should use this class for setting custom algorithm parameters.
 */
struct Config {
 public:
  Config();

  /**
   * Config constructor.
   *
   * @param param_map Key-value map with config parameters.
   */
  Config(std::unordered_map<std::string, std::string> const& param_map);

  /**
   * Config constructor.
   *
   * Constructor from .ini file.
   * Each line should be either ini-header or parameter given in following form:
   * paramName=paramValue
   *
   * @param config_path Path to .ini file with parameters.
   */
  Config(std::string const& config_path);
  // Minimal size of a region, unit: pixels
  int32_t patch_size = 10;
  // Seed selection, bigger value detects dominant normal direction more precisely
  int32_t histogram_bins_per_coord = 20;
  // Normal deviation angle threshold, unit: degree
  float min_cos_angle_merge = 0.90;
  // Distance between two regions threshold, unit: mm
  float max_merge_dist = 500;
  // Minimum number of cells to consider a dominant direction valid
  int32_t min_region_growing_candidate_size = 5;
  // Minimum number of cells considered to be a planar region
  int32_t min_region_growing_cells_activated = 4;
  // Score to consider a region as a plane
  float min_region_planarity_score = 0.55;
  // Depth-dependent threshold coefficient for depth-discontinuity evaluation
  float depth_sigma_coeff = 1.425e-6;
  // Depth-dependent threshold margin for depth-discontinuity evaluation
  float depth_sigma_margin = 10.;
  // Ratio of valid (not NaN) points in cell
  int32_t min_pts_per_cell = 3;
  // Difference between two adjacent pixels to consider a  depth-discontinuity
  float depth_discontinuity_threshold = 160;
  // Maximum depth-discontinuity occurrences inside one cell
  int32_t max_number_depth_discontinuity = 1;
  // RANSAC refinement flag
  bool ransac_refinement = false;
  // Maximum number of RANSAC iterations
  int32_t ransac_max_iterations = 1000;
  // RANSAC Threshold
  float ransac_threshold = 1.;
  // Minimal inliers ratio for plane points to be valid
  float ransac_inliers_ratio = 0.9;
  // The number of threads used in parallel algorithms.
  int32_t number_threads = 8;
};
}  // namespace config
}  // namespace deplex