#pragma once

#include "histogram.hpp"
#include "planeseg.hpp"

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <bitset>
#include <vector>
#include <memory>

#ifndef BITSET_SIZE
#define BITSET_SIZE 65536  // 2^16
#endif

namespace cape {

typedef uchar label_t;

class CAPE {
 public:
  CAPE(int32_t image_height, int32_t image_width,
       config::Config config = __default_config);
  cv::Mat process(Eigen::MatrixXf const& pcd_array);

 private:
  const static config::Config __default_config;
  config::Config _config;
  int32_t _nr_horizontal_cells;
  int32_t _nr_vertical_cells;
  int32_t _nr_total_cells;
  int32_t _nr_pts_per_cell;
  int32_t _image_height;
  int32_t _image_width;
  std::vector<std::shared_ptr<PlaneSeg>> _cell_grid;
  cv::Mat_<int32_t> _grid_plane_seg_map;
  cv::Mat_<label_t> _grid_plane_seg_map_eroded;
  std::vector<label_t> _seg_map_stacked;
  void organizeByCell(Eigen::MatrixXf const& pcd_array, Eigen::MatrixXf* out);
  std::bitset<BITSET_SIZE> findPlanarCells(Eigen::MatrixXf const& pcd_array);
  Histogram initializeHistogram(std::bitset<BITSET_SIZE> const& planar_flags);
  std::vector<float> computeCellDistTols(
      Eigen::MatrixXf const& pcd_array,
      std::bitset<BITSET_SIZE> const& planar_flags);
  std::vector<std::shared_ptr<PlaneSeg>> createPlaneSegments(
      Histogram hist, std::bitset<BITSET_SIZE> const& planar_flags,
      std::vector<float> const& cell_dist_tols);
  std::vector<int32_t> mergePlanes(
      std::vector<std::shared_ptr<PlaneSeg>>& plane_segments);
  void refinePlanes(
      std::vector<std::shared_ptr<PlaneSeg>> const& plane_segments,
      std::vector<int32_t> const& merge_labels,
      Eigen::MatrixXf const& pcd_array);
  cv::Mat toLabels();
  cv::Mat coarseToLabels(std::vector<int32_t> const& labels);
  void cleanArtifacts();
  void refineCells(std::shared_ptr<const PlaneSeg> const plane,
                   label_t label,
                   cv::Mat const& mask,
                   Eigen::MatrixXf const& pcd_array);
  void growSeed(int32_t x, int32_t y, int32_t prev_index,
                std::bitset<BITSET_SIZE> const& unassigned,
                std::bitset<BITSET_SIZE>* activation_map,
                std::vector<float> const& cell_dist_tols) const;

  std::vector<std::bitset<BITSET_SIZE>> getConnectedComponents(
      size_t nr_planes) const;

#ifdef DEBUG_CAPE
  void planarCellsToLabels(std::bitset<BITSET_SIZE> const& planar_flags,
                           std::string const& save_path);
  void planeSegmentsMapToLabels(std::string const& save_path,
                                cv::Mat_<int32_t> const& cell_map);
  void mergeSegmentsToLabels(std::string const& save_path,
                             std::vector<int32_t> const& merge_labels);
#endif
};
}  // namespace cape
