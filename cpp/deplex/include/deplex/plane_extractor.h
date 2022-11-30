#pragma once

#include <memory>

#include <Eigen/Core>

#include "deplex/config.h"

namespace deplex {
class PlaneExtractor {
 public:
  PlaneExtractor(int32_t image_height, int32_t image_width, config::Config config = kDefaultConfig);
  ~PlaneExtractor();

  Eigen::VectorXi process(Eigen::MatrixXf const& pcd_array);

  PlaneExtractor(PlaneExtractor && op) noexcept;
  PlaneExtractor& operator=(PlaneExtractor && op) noexcept;

  const static config::Config kDefaultConfig;
 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace deplex