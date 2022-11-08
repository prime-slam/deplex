#include <CAPE/cape.h>

#include <filesystem>
#include <iostream>
#include <opencv2/core/eigen.hpp>

constexpr int IMAGE_HEIGHT(480), IMAGE_WIDTH(640);

void organizeByCell(Eigen::MatrixXf const& pcd_array, Eigen::MatrixXf& out) {
  auto PATCH_SIZE = 12;
  auto nr_horizontal_cells = IMAGE_WIDTH / PATCH_SIZE;

  int mxn = IMAGE_WIDTH * IMAGE_HEIGHT;
  int mxn2 = 2 * mxn;

  int stacked_id = 0;
  for (int r = 0; r < IMAGE_HEIGHT; r++) {
    int cell_r = r / PATCH_SIZE;
    int local_r = r % PATCH_SIZE;
    for (int c = 0; c < IMAGE_WIDTH; c++) {
      int cell_c = c / PATCH_SIZE;
      int local_c = c % PATCH_SIZE;
      auto shift =
          (cell_r * nr_horizontal_cells + cell_c) * PATCH_SIZE * PATCH_SIZE +
          local_r * PATCH_SIZE + local_c;

      *(out.data() + shift) = *(pcd_array.data() + stacked_id);
      *(out.data() + mxn + shift) = *(pcd_array.data() + mxn + stacked_id);
      *(out.data() + mxn2 + shift) = *(pcd_array.data() + mxn2 + stacked_id);
      stacked_id++;
    }
  }
}

Eigen::MatrixXf readImage(std::string const& image_path,
                          std::string const& intrinsics_path) {
  cv::Mat d_img = cv::imread(image_path, cv::IMREAD_ANYDEPTH);
  d_img.convertTo(d_img, CV_32F);
  Eigen::MatrixXf image_mx;
  cv::cv2eigen(d_img, image_mx);

  float fx = 535.4, fy = 539.2, cx = 320.1, cy = 247.6;
  float factor = 1;

  Eigen::MatrixXf points(IMAGE_HEIGHT * IMAGE_WIDTH, 3);

  Eigen::VectorXf column_indices =
      Eigen::VectorXf::LinSpaced(IMAGE_WIDTH, 0.0, IMAGE_WIDTH - 1)
          .replicate(IMAGE_HEIGHT, 1);

  Eigen::VectorXf row_indices =
      Eigen::VectorXf::LinSpaced(IMAGE_HEIGHT, 0.0, IMAGE_HEIGHT - 1)
          .replicate(1, IMAGE_WIDTH)
          .reshaped<Eigen::RowMajor>();

  points.col(2) = image_mx.reshaped<Eigen::RowMajor>().array() / factor;
  points.col(0) = (column_indices.array() - cx) * points.col(2).array() / fx;
  points.col(1) = (row_indices.array() - cy) * points.col(2).array() / fy;

  Eigen::MatrixXf organized(IMAGE_HEIGHT * IMAGE_WIDTH, 3);
  organizeByCell(points, organized);
  return organized;
}

int main() {
  std::filesystem::path data_dir =
      std::filesystem::current_path().parent_path().parent_path() / "data";
  std::filesystem::path image_path = data_dir / "tum/1341848230.910894.png";
  std::filesystem::path intrinsics_path =
      data_dir / "configs/TUM_fr3_long_val.xml";
  std::filesystem::path config_path = data_dir / "configs/TUM_fr3_long_val.ini";
  //    std::filesystem::path image_path = data_dir / "icl_nuim/0.png";
  //    std::filesystem::path intrinsics_path =
  //        data_dir / "configs/ICL_living_room.xml";
  //    std::filesystem::path config_path = data_dir /
  //    "configs/ICL_living_room.ini";
  cape::config::Config config = cape::config::Config(config_path);

  auto algorithm = cape::CAPE(IMAGE_HEIGHT, IMAGE_WIDTH, config);
  Eigen::MatrixXf organized_pcd = readImage(image_path, intrinsics_path);

  algorithm.process(organized_pcd);

  return 0;
}