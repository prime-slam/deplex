#pragma once

#include <Eigen/Core>
#include <fstream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgcodecs.hpp>

namespace deplex::reader {
Eigen::MatrixXf readImage(std::string const& image_path,
                          Eigen::Matrix3f const& K) {
  cv::Mat d_img = cv::imread(image_path, cv::IMREAD_ANYDEPTH);
  d_img.convertTo(d_img, CV_32F);
  Eigen::MatrixXf image_mx;
  cv::cv2eigen(d_img, image_mx);

  float fx = K(0, 0), fy = K(1, 1);
  float cx = K(0, 2), cy = K(1, 2);

  Eigen::Index image_height = d_img.rows;
  Eigen::Index image_width = d_img.cols;
  Eigen::MatrixXf pcd_points(image_width * image_height, 3);

  Eigen::VectorXf column_indices =
      Eigen::VectorXf::LinSpaced(image_width, 0.0, image_width - 1)
          .replicate(image_height, 1);

  Eigen::VectorXf row_indices =
      Eigen::VectorXf::LinSpaced(image_height, 0.0, image_height - 1)
          .replicate(1, image_width)
          .reshaped<Eigen::RowMajor>();

  pcd_points.col(2) = image_mx.reshaped<Eigen::RowMajor>();
  pcd_points.col(0) =
      (column_indices.array() - cx) * pcd_points.col(2).array() / fx;
  pcd_points.col(1) =
      (row_indices.array() - cy) * pcd_points.col(2).array() / fy;

  return pcd_points;
}

Eigen::MatrixXf readImage(std::string const& image_path,
                          std::string const& intrinsics_path) {
  std::ifstream in(intrinsics_path);
  Eigen::Matrix3f K;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      in >> K(i, j);
    }
  }

  return readImage(image_path, K);
}
}  // namespace deplex::reader