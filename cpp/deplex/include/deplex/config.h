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

#include <map>
#include <string>
#include <vector>

namespace deplex::config {
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
  Config(std::map<std::string, std::string> const& param_map);

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

  int32_t getInt(std::string const& param_name) const;

  float getFloat(std::string const& param_name) const;

  bool getBool(std::string const& param_name) const;

 private:
  std::map<std::string, std::string> param_map_;

  std::string findValue(std::string const& name) const;

  std::map<std::string, std::string> iniLoad(std::string const& path) const;
};
}  // namespace deplex::config