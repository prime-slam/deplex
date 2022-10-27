#pragma once

#include <cstdint>
#include <map>
#include <string>

namespace cape::config {
// Default parameters for 'TUM_fr3_long_office_validation'
const std::map<std::string, std::string> DEFAULT_PARAMETERS{
    // General parameters
    {"patchSize", "12"},
    {"histogramBinsPerCoord", "20"},
    {"minCosAngleForMerge", "0.93"},
    {"maxMergeDist", "500"},
    {"minRegionGrowingCandidateSize", "5"},
    // Parameters used in plane validation
    {"depthSigmaCoeff", "1.425e-6"},
    {"depthSigmaMargin", "10"},
    {"minPtsPerCell", "3"},
    {"depthDiscontinuityThreshold", "160"},
    {"maxNumberDepthDiscontinuity", "1"}};

class Config {
 public:
  Config();
  inline int32_t getInt(std::string const& param_name) const;
  inline float getFloat(std::string const& param_name) const;

 private:
  inline std::string findValue(std::string const& name) const;
  std::map<std::string, std::string> _param_map;
};

Config::Config() : _param_map(DEFAULT_PARAMETERS) {}

std::string Config::findValue(std::string const& name) const {
  auto value_ptr = _param_map.find(name);
  if (value_ptr == _param_map.end())
    throw std::runtime_error("Config invalid parameter name provided: " + name);
  return value_ptr->second;
}

int32_t Config::getInt(std::string const& name) const {
  return std::stoi(findValue(name));
}

float Config::getFloat(std::string const& name) const {
  return std::stof(findValue(name));
}

}  // namespace cape::config