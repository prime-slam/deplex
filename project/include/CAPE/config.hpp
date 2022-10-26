#pragma once

#include <cstdint>
#include <string>

namespace cape::config{
  class Config{
   public:
    int32_t getInt(std::string const& param_name) const;
    float getFloat(std::string const& param_name) const;
   private:
  };
}