#include "deplex/algorithm/config.h"

#include <fstream>

namespace deplex::config {
Config::Config(std::map<std::string, std::string> const& param_map) : param_map_(param_map) {}

Config::Config(std::string const& config_path) : param_map_(iniLoad(config_path)) {}

std::string Config::findValue(std::string const& name) const {
  auto value_ptr = param_map_.find(name);
  if (value_ptr == param_map_.end())
    throw std::runtime_error("Config invalid parameter name provided: " + name);
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