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
#include "deplex/config.h"

#include <fstream>
#include <iostream>

namespace deplex {
namespace config {
Config::Config() = default;

// TODO:
Config::Config(std::unordered_map<std::string, std::string> const& param_map) {}

Config::Config(std::string const& config_path) {
  std::ifstream ini_file(config_path);
  if (!ini_file.is_open()) {
    throw std::runtime_error("Couldn't open ini file: " + config_path);
  }
  while (ini_file) {
    std::string line;
    std::getline(ini_file, line);
    if (line.empty() || line[0] == '#') continue;
    std::string key, value;
    size_t eq_pos = line.find_first_of('=');
    if (eq_pos == std::string::npos || eq_pos == 0) {
      continue;
    }
    key = line.substr(0, eq_pos);
    value = line.substr(eq_pos + 1);
    if (key == "patchSize") {
      patch_size = std::stoi(value);
    } else if (key == "histogramBinsPerCoord") {
      histogram_bins_per_coord = std::stoi(value);
    } else if (key == "minCosAngleForMerge") {
      min_cos_angle_merge = std::stof(value);
    } else if (key == "maxMergeDist") {
      max_merge_dist = std::stof(value);
    } else if (key == "minRegionGrowingCandidateSize") {
      min_region_growing_candidate_size = std::stoi(value);
    } else if (key == "minRegionGrowingCellsActivated") {
      min_region_growing_cells_activated = std::stoi(value);
    } else if (key == "minRegionPlanarityScore") {
      min_region_planarity_score = std::stof(value);
    } else if (key == "depthSigmaCoeff") {
      depth_sigma_coeff = std::stof(value);
    } else if (key == "depthSigmaMargin") {
      depth_sigma_margin = std::stof(value);
    } else if (key == "minPtsPerCell") {
      min_pts_per_cell = std::stoi(value);
    } else if (key == "depthDiscontinuityThreshold") {
      depth_discontinuity_threshold = std::stof(value);
    } else if (key == "maxNumberDepthDiscontinuity") {
      max_number_depth_discontinuity = std::stoi(value);
    } else {
      std::cerr << "Unknown parameter name: " << key << '\n';
    }
  }
}

}  // namespace config
}  // namespace deplex