#pragma once

#include <map>
#include <string>
#include <vector>

namespace deplex::config {
class Config {
 public:
  Config(std::map<std::string, std::string> const& param_map);
  Config(std::string const& config_path);
  inline int32_t getInt(std::string const& param_name) const;
  inline float getFloat(std::string const& param_name) const;
  inline bool getBool(std::string const& param_name) const;

 private:
  inline std::string findValue(std::string const& name) const;
  std::map<std::string, std::string> iniLoad(std::string const& path) const;
  std::map<std::string, std::string> _param_map;
};
}  // namespace deplex::config