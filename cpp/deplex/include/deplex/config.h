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
class Config {
 public:
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

  const int32_t patch_size = 10;
  const int32_t histogram_bins_per_coord = 20;
  const float min_cos_angle_merge = 0.90;
  const float max_merge_dist = 500;
  const int32_t min_region_growing_candidate_size = 5;
  const int32_t min_region_growing_cells_activated = 4;
  const float min_region_planarity_score = 0.55;
  const float depth_sigma_coeff = 1.425e-6;
  const float depth_sigma_margin = 10.;
  const int32_t min_pts_per_cell = 3;
  const float depth_discontinuity_threshold = 160;
  const int32_t max_number_depth_discontinuity = 1;
};
}  // namespace config
}  // namespace deplex