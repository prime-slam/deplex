#pragma once

#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

namespace deplex::config {
namespace ini_read {
std::map<std::string, std::string> iniLoad(std::string iniFileName) {
  std::map<std::string, std::string> parameters;
  std::ifstream ini_file(iniFileName);
  if (!ini_file.is_open()) {
    std::cerr << "Error! Couldn't open ini file: " << iniFileName << '\n';
    return {};
  }
  while (ini_file) {
    std::string line;
    std::getline(ini_file, line);
    if (line.empty() || line[0] == '#') continue;
    std::string key, value;
    size_t eqPos = line.find_first_of("=");
    if (eqPos == std::string::npos || eqPos == 0) {
      continue;
    }
    key = line.substr(0, eqPos);
    value = line.substr(eqPos + 1);
    parameters[key] = value;
  }

  return parameters;
}
}  // namespace ini_read
// Default parameters for 'TUM_fr3_long_office_validation'

class Config {
 public:
  Config(std::map<std::string, std::string> const& param_map);
  Config(std::string const& config_path);
  inline int32_t getInt(std::string const& param_name) const;
  inline float getFloat(std::string const& param_name) const;
  inline bool getBool(std::string const& param_name) const;

 private:
  inline std::string findValue(std::string const& name) const;
  std::map<std::string, std::string> _param_map;
};

Config::Config(std::map<std::string, std::string> const& param_map)
    : _param_map(param_map) {}

Config::Config(std::string const& config_path)
    : _param_map(ini_read::iniLoad(config_path)) {}

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

bool Config::getBool(std::string const& name) const {
  auto value = findValue(name);
  if (value == "1" || value == "true" || value == "True")
    return true;
  else if (value == "0" || value == "false" || value == "False")
    return false;

  throw std::runtime_error(
      "Invalid value for parameter 'doRefinement'. Allowed values: "
      "[1, true, True] or [0, false, False]");
}

}  // namespace deplex::config