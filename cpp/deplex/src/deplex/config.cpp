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
#include "deplex/config.h"

#include <fstream>
#include <iostream>

namespace deplex::config {
Config::Config(std::map<std::string, std::string> const& param_map) : param_map_(param_map) {}

Config::Config(std::string const& config_path) : param_map_(iniLoad(config_path)) {}

bool Config::updateValue(std::string const& name, std::string const& value) {
  auto value_ptr = param_map_.find(name);
  if (value_ptr == param_map_.end()) {
    std::cerr << "Warning: Couldn't update parameter: " + name << ". The name is not in parameters.";
    return false;
  }
  value_ptr->second = value;
  return true;
}

std::string Config::findValue(std::string const& name) const {
  auto value_ptr = param_map_.find(name);
  if (value_ptr == param_map_.end()) {
    throw std::runtime_error("Config invalid parameter name provided: " + name);
  }
  return value_ptr->second;
}

std::map<std::string, std::string> Config::iniLoad(std::string const& path) const {
  std::map<std::string, std::string> parameters;
  std::ifstream ini_file(path);
  if (!ini_file.is_open()) {
    throw std::runtime_error("Couldn't open ini file: " + path);
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
    parameters[key] = value;
  }

  return parameters;
}

int32_t Config::getInt(std::string const& name) const {
  return std::stoi(findValue(name));
}

float Config::getFloat(std::string const& name) const {
  return std::stof(findValue(name));
}

bool Config::getBool(std::string const& name) const {
  auto value = findValue(name);
  if (value == "1" || value == "true" || value == "True")
    return true;
  else if (value == "0" || value == "false" || value == "False")
    return false;

  throw std::runtime_error("Invalid value for boolean parameter: " + name +
                           ". Allowed values: "
                           "[1, true, True] or [0, false, False]");
}
}  // namespace deplex::config