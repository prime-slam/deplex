#include <CAPE/cape.h>

#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>

constexpr int IMAGE_HEIGHT(480), IMAGE_WIDTH(640);

bool loadCalibParameters(std::string filepath, cv::Mat& intrinsics_rgb,
                         cv::Mat& dist_coeffs_rgb, cv::Mat& intrinsics_ir,
                         cv::Mat& dist_coeffs_ir, cv::Mat& R, cv::Mat& T) {
  cv::FileStorage fs(filepath, cv::FileStorage::READ);
  if (fs.isOpened()) {
    fs["RGB_intrinsic_params"] >> intrinsics_rgb;
    fs["RGB_distortion_coefficients"] >> dist_coeffs_rgb;
    fs["IR_intrinsic_params"] >> intrinsics_ir;
    fs["IR_distortion_coefficients"] >> dist_coeffs_ir;
    fs["Rotation"] >> R;
    fs["Translation"] >> T;
    fs.release();
    return true;
  } else {
    std::cerr << "Calibration file missing" << std::endl;
    return false;
  }
}

void projectPointCloud(cv::Mat& X, cv::Mat& Y, cv::Mat& Z, cv::Mat& U,
                       cv::Mat& V, float fx_rgb, float fy_rgb, float cx_rgb,
                       float cy_rgb, double z_min,
                       Eigen::MatrixXf& cloud_array) {
  int width = X.cols;
  int height = X.rows;

  // Project to image coordinates
  cv::divide(X, Z, U, 1);
  cv::divide(Y, Z, V, 1);
  U = U * fx_rgb + cx_rgb;
  V = V * fy_rgb + cy_rgb;
  // Reusing U as cloud index
  // U = V*width + U + 0.5;

  float *sz, *sx, *sy, *u_ptr, *v_ptr, *id_ptr;
  float z, u, v;
  int id;
  for (int r = 0; r < height; r++) {
    sx = X.ptr<float>(r);
    sy = Y.ptr<float>(r);
    sz = Z.ptr<float>(r);
    u_ptr = U.ptr<float>(r);
    v_ptr = V.ptr<float>(r);
    for (int c = 0; c < width; c++) {
      z = sz[c];
      u = u_ptr[c];
      v = v_ptr[c];
      if (z > z_min && u > 0 && v > 0 && u < width && v < height) {
        id = floor(v) * width + u;
        cloud_array(id, 0) = sx[c];
        cloud_array(id, 1) = sy[c];
        cloud_array(id, 2) = z;
      }
    }
  }
}

void organizePointCloudByCell(Eigen::MatrixXf& cloud_in,
                              Eigen::MatrixXf& cloud_out, cv::Mat& cell_map) {
  int width = cell_map.cols;
  int height = cell_map.rows;
  int mxn = width * height;
  int mxn2 = 2 * mxn;

  int id, it(0);
  int* cell_map_ptr;
  for (int r = 0; r < height; r++) {
    cell_map_ptr = cell_map.ptr<int>(r);
    for (int c = 0; c < width; c++) {
      id = cell_map_ptr[c];
      *(cloud_out.data() + id) = *(cloud_in.data() + it);
      *(cloud_out.data() + mxn + id) = *(cloud_in.data() + mxn + it);
      *(cloud_out.data() + mxn2 + id) = *(cloud_in.data() + mxn2 + it);
      it++;
    }
  }
}

Eigen::MatrixXf readImage(std::string const& image_path,
                          std::string const& intrinsics_path,
                          int32_t patch_size) {
  // Get intrinsics
  cv::Mat K_rgb, K_ir, dist_coeffs_rgb, dist_coeffs_ir, R_stereo, t_stereo;
  loadCalibParameters(intrinsics_path, K_rgb, dist_coeffs_rgb, K_ir,
                      dist_coeffs_ir, R_stereo, t_stereo);
  float fx_ir = K_ir.at<double>(0, 0);
  float fy_ir = K_ir.at<double>(1, 1);
  float cx_ir = K_ir.at<double>(0, 2);
  float cy_ir = K_ir.at<double>(1, 2);
  float fx_rgb = K_rgb.at<double>(0, 0);
  float fy_rgb = K_rgb.at<double>(1, 1);
  float cx_rgb = K_rgb.at<double>(0, 2);
  float cy_rgb = K_rgb.at<double>(1, 2);

  int nr_horizontal_cells = IMAGE_WIDTH / patch_size;
  int nr_vertical_cells = IMAGE_HEIGHT / patch_size;

  // Pre-computations for backprojection
  cv::Mat_<float> X_pre(IMAGE_HEIGHT, IMAGE_WIDTH);
  cv::Mat_<float> Y_pre(IMAGE_HEIGHT, IMAGE_WIDTH);
  cv::Mat_<float> U(IMAGE_HEIGHT, IMAGE_WIDTH);
  cv::Mat_<float> V(IMAGE_HEIGHT, IMAGE_WIDTH);
  for (int r = 0; r < IMAGE_HEIGHT; r++) {
    for (int c = 0; c < IMAGE_WIDTH; c++) {
      // Not efficient but at this stage doesn't t matter
      X_pre.at<float>(r, c) = (c - cx_ir) / fx_ir;
      Y_pre.at<float>(r, c) = (r - cy_ir) / fy_ir;
    }
  }

  // Pre-computations for maping an image point cloud to a cache-friendly array
  // where cell's local point clouds are contiguous
  cv::Mat_<int> cell_map(IMAGE_HEIGHT, IMAGE_WIDTH);

  for (int r = 0; r < IMAGE_HEIGHT; r++) {
    int cell_r = r / patch_size;
    int local_r = r % patch_size;
    for (int c = 0; c < IMAGE_WIDTH; c++) {
      int cell_c = c / patch_size;
      int local_c = c % patch_size;
      cell_map.at<int>(r, c) =
          (cell_r * nr_horizontal_cells + cell_c) * patch_size * patch_size +
          local_r * patch_size + local_c;
    }
  }

  cv::Mat_<float> X(IMAGE_HEIGHT, IMAGE_WIDTH);
  cv::Mat_<float> Y(IMAGE_HEIGHT, IMAGE_WIDTH);
  Eigen::MatrixXf cloud_array(IMAGE_WIDTH * IMAGE_HEIGHT, 3);
  Eigen::MatrixXf cloud_array_organized(IMAGE_WIDTH * IMAGE_HEIGHT, 3);

  auto d_img = cv::imread(image_path, cv::IMREAD_ANYDEPTH);
  d_img.convertTo(d_img, CV_32F);

  // Backproject to point cloud
  X = X_pre.mul(d_img);
  Y = Y_pre.mul(d_img);
  cloud_array.setZero();
  projectPointCloud(X, Y, d_img, U, V, fx_rgb, fy_rgb, cx_rgb, cy_rgb,
                    t_stereo.at<double>(2), cloud_array);
  organizePointCloudByCell(cloud_array, cloud_array_organized, cell_map);

  return cloud_array_organized;
}

int main() {
  std::filesystem::path data_dir =
      std::filesystem::current_path().parent_path().parent_path() / "data";
  std::filesystem::path image_path = data_dir / "tum/1341848230.910894.png";
  std::filesystem::path intrinsics_path =
      data_dir / "configs/TUM_fr3_long_val.xml";
  std::filesystem::path config_path = data_dir / "configs/TUM_fr3_long_val.ini";
  cape::config::Config config = cape::config::Config(config_path);

  auto algorithm = cape::CAPE(IMAGE_HEIGHT, IMAGE_WIDTH, config);
  Eigen::MatrixXf organized_pcd =
      readImage(image_path, intrinsics_path, config.getInt("patchSize"));

  algorithm.process(organized_pcd);

  return 0;
}