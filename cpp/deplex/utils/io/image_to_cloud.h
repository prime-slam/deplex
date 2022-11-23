#pragma once

#include <Eigen/Core>

namespace deplex::utils {
Eigen::MatrixXf readImage(std::string const& image_path,
                          Eigen::Matrix3f const& K);

Eigen::MatrixXf readImage(std::string const& image_path,
                          std::string const& intrinsics_path);
}  // namespace deplex::utils