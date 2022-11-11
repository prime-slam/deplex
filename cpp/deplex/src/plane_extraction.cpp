#include "deplex/plane_extraction.h"

#include <numeric>
#include <opencv2/core/eigen.hpp>

#ifdef DEBUG_DEPLEX
#include <fstream>
#include <iostream>
#endif

namespace deplex {

const config::Config PlaneExtractor::kDefaultConfig{
    {// General parameters
     {"patchSize", "12"},
     {"histogramBinsPerCoord", "20"},
     {"minCosAngleForMerge", "0.93"},
     {"maxMergeDist", "500"},
     {"minRegionGrowingCandidateSize", "5"},
     {"minRegionGrowingCellsActivated", "4"},
     {"minRegionPlanarityScore", "50"},
     {"doRefinement", "true"},
     {"refinementMultiplierCoeff", "15"},
     // Parameters used in plane validation
     {"depthSigmaCoeff", "1.425e-6"},
     {"depthSigmaMargin", "10"},
     {"minPtsPerCell", "3"},
     {"depthDiscontinuityThreshold", "160"},
     {"maxNumberDepthDiscontinuity", "1"}}};

PlaneExtractor::PlaneExtractor(int32_t image_height, int32_t image_width,
                               config::Config config)
    : _config(config),
      _nr_horizontal_cells(image_width / config.getInt("patchSize")),
      _nr_vertical_cells(image_height / config.getInt("patchSize")),
      _nr_total_cells(_nr_horizontal_cells * _nr_vertical_cells),
      _nr_pts_per_cell(pow(config.getInt("patchSize"), 2)),
      _image_height(image_height),
      _image_width(image_width),
      _cell_grid(_nr_total_cells, nullptr),
      _grid_plane_seg_map(_nr_vertical_cells, _nr_horizontal_cells, 0),
      _grid_plane_seg_map_eroded(_nr_vertical_cells, _nr_horizontal_cells,
                                 uchar(0)),
      _seg_map_stacked(image_height * image_width, 0) {}

Eigen::VectorXi PlaneExtractor::process(Eigen::MatrixXf const& pcd_array) {
  // 0. Stack array by cell
  Eigen::MatrixXf organized_array(pcd_array.rows(), pcd_array.cols());
  organizeByCell(pcd_array, &organized_array);
  // 1. Planar cell fitting
  std::bitset<BITSET_SIZE> planar_flags = findPlanarCells(organized_array);
#ifdef DEBUG_DEPLEX
  planarCellsToLabels(planar_flags, "dbg_1_planar_cells.csv");
  std::clog << "[DebugInfo] Planar cell found: " << planar_flags.count()
            << '\n';
#endif
  // 2. Histogram initialization
  Histogram hist = initializeHistogram(planar_flags);
  // 3. Compute cell dist tols
  std::vector<float> cell_dist_tols =
      computeCellDistTols(organized_array, planar_flags);
  // 4. Region growing
  auto plane_segments = createPlaneSegments(hist, planar_flags, cell_dist_tols);
#ifdef DEBUG_DEPLEX
  planeSegmentsMapToLabels("dbg_2_plane_segments_raw.csv", _grid_plane_seg_map);
  std::clog << "[DebugInfo] Plane segments found: " << plane_segments.size()
            << '\n';
#endif
  // 5. Merge planes
  std::vector<int32_t> merge_labels = mergePlanes(plane_segments);
#ifdef DEBUG_DEPLEX
  mergeSegmentsToLabels("dbg_3_plane_segments_merged.csv", merge_labels);
  std::vector<int32_t> sorted_labels(merge_labels);
  std::sort(sorted_labels.begin(), sorted_labels.end());

  std::clog << "[DebugInfo] Planes number after merge: "
            << std::distance(
                   sorted_labels.begin(),
                   std::unique(sorted_labels.begin(), sorted_labels.end()))
            << '\n';
#endif
  // 6. Refinement (optional)
  cv::Mat_ labels(_image_height, _image_width, 0);
  if (_config.getBool("doRefinement")) {
    refinePlanes(plane_segments, merge_labels, organized_array);
    labels = toLabels();
  } else {
    labels = coarseToLabels(merge_labels);
  }
#ifdef DEBUG_DEPLEX
  std::ofstream of("dbg_4_labels.csv");
  of << cv::format(labels, cv::Formatter::FMT_CSV);
#endif
  // 7. Cleanup
  cleanArtifacts();
  Eigen::MatrixXi eigen_labels;
  cv::cv2eigen(labels, eigen_labels);
  return eigen_labels.reshaped<Eigen::RowMajor>();
}

void PlaneExtractor::organizeByCell(Eigen::MatrixXf const& pcd_array,
                                    Eigen::MatrixXf* out) {
  int32_t patch_size = _config.getInt("patchSize");
  int32_t mxn = _image_width * _image_height;
  int32_t mxn2 = 2 * mxn;

  int stacked_id = 0;
  for (int r = 0; r < _image_height; r++) {
    int cell_r = r / patch_size;
    int local_r = r % patch_size;
    for (int c = 0; c < _image_width; c++) {
      int cell_c = c / patch_size;
      int local_c = c % patch_size;
      auto shift =
          (cell_r * _nr_horizontal_cells + cell_c) * patch_size * patch_size +
          local_r * patch_size + local_c;

      *(out->data() + shift) = *(pcd_array.data() + stacked_id);
      *(out->data() + mxn + shift) = *(pcd_array.data() + mxn + stacked_id);
      *(out->data() + mxn2 + shift) = *(pcd_array.data() + mxn2 + stacked_id);
      stacked_id++;
    }
  }
}

std::bitset<BITSET_SIZE> PlaneExtractor::findPlanarCells(
    Eigen::MatrixXf const& pcd_array) {
  std::bitset<BITSET_SIZE> planar_flags;
  int32_t cell_width = _config.getInt("patchSize");
  int32_t cell_height = _config.getInt("patchSize");
  int32_t stacked_cell_id = 0;
  for (Eigen::Index cell_r = 0; cell_r < _nr_vertical_cells; ++cell_r) {
    for (Eigen::Index cell_h = 0; cell_h < _nr_horizontal_cells; ++cell_h) {
      _cell_grid[stacked_cell_id] = std::make_shared<PlaneSeg>(
          stacked_cell_id, cell_width, cell_height, pcd_array, _config);
      planar_flags[stacked_cell_id] = _cell_grid[stacked_cell_id]->isPlanar();
      ++stacked_cell_id;
    }
  }
  return planar_flags;
}

Histogram PlaneExtractor::initializeHistogram(
    std::bitset<BITSET_SIZE> const& planar_flags) {
  Eigen::MatrixXd spherical_coord(_nr_total_cells, 2);
  for (size_t cell_id = planar_flags._Find_first();
       cell_id != planar_flags.size();
       cell_id = planar_flags._Find_next(cell_id)) {
    Eigen::Vector3d cell_normal = _cell_grid[cell_id]->getNormal();
    double n_proj_norm =
        sqrt(cell_normal[0] * cell_normal[0] + cell_normal[1] * cell_normal[1]);
    spherical_coord(cell_id, 0) = acos(-cell_normal[2]);
    spherical_coord(cell_id, 1) =
        atan2(cell_normal[0] / n_proj_norm, cell_normal[1] / n_proj_norm);
  }
  int nr_bins_per_coord = _config.getInt("histogramBinsPerCoord");
  return Histogram{nr_bins_per_coord, spherical_coord, planar_flags};
}

std::vector<float> PlaneExtractor::computeCellDistTols(
    Eigen::MatrixXf const& pcd_array,
    std::bitset<BITSET_SIZE> const& planar_flags) {
  std::vector<float> cell_dist_tols(_nr_total_cells, 0);
  double cos_angle_for_merge = _config.getFloat("minCosAngleForMerge");
  float sin_angle_for_merge = sqrt(1 - pow(cos_angle_for_merge, 2));
  // TODO: Put "minMergeDist" to config
  float min_merge_dist = 20.0f;
  float max_merge_dist = _config.getFloat("maxMergeDist");

  for (size_t cell_id = planar_flags._Find_first();
       cell_id != planar_flags.size();
       cell_id = planar_flags._Find_next(cell_id)) {
    float cell_diameter =
        (pcd_array.block(cell_id * _nr_pts_per_cell + _nr_pts_per_cell - 1, 0,
                         1, 3) -
         pcd_array.block(cell_id * _nr_pts_per_cell, 0, 1, 3))
            .norm();
    float truncated_distance =
        std::min(std::max(cell_diameter * sin_angle_for_merge, min_merge_dist),
                 max_merge_dist);
    cell_dist_tols[cell_id] = powf(truncated_distance, 2);
  }

  return cell_dist_tols;
}

std::vector<std::shared_ptr<PlaneSeg>> PlaneExtractor::createPlaneSegments(
    Histogram hist, std::bitset<BITSET_SIZE> const& planar_flags,
    std::vector<float> const& cell_dist_tols) {
  std::vector<std::shared_ptr<PlaneSeg>> plane_segments;
  std::bitset<BITSET_SIZE> unassigned_mask(planar_flags);
  auto remaining_planar_cells = static_cast<int32_t>(planar_flags.count());

  while (remaining_planar_cells > 0) {
    // 1. Seeding
    std::vector<int32_t> seed_candidates = hist.getPointsFromMostFrequentBin();
    if (seed_candidates.size() <
        _config.getInt("minRegionGrowingCandidateSize")) {
      return plane_segments;
    }
    // 2. Select seed with minimum MSE
    int32_t seed_id;
    double min_mse = INT_MAX;
    for (int32_t seed_candidate : seed_candidates) {
      if (_cell_grid[seed_candidate]->getMSE() < min_mse) {
        seed_id = seed_candidate;
        min_mse = _cell_grid[seed_candidate]->getMSE();
      }
    }
    // 3. Grow seed
    std::shared_ptr<PlaneSeg> new_segment = _cell_grid[seed_id];
    int32_t y = seed_id / _nr_horizontal_cells;
    int32_t x = seed_id % _nr_horizontal_cells;
    std::bitset<BITSET_SIZE> activation_map;
    growSeed(x, y, seed_id, unassigned_mask, &activation_map, cell_dist_tols);
    // 4. Merge activated cells & remove from hist
    for (size_t i = activation_map._Find_first(); i != activation_map.size();
         i = activation_map._Find_next(i)) {
      *new_segment += *_cell_grid[i];
      hist.removePoint(static_cast<int32_t>(i));
      --remaining_planar_cells;
    }
    unassigned_mask &= (~activation_map);
    size_t nr_cells_activated = activation_map.count();

    if (nr_cells_activated < _config.getInt("minRegionGrowingCellsActivated")) {
      continue;
    }

    new_segment->calculateStats();

    // 5. Model fitting
    if (new_segment->getScore() > _config.getFloat("minRegionPlanarityScore")) {
      plane_segments.push_back(new_segment);
      auto nr_curr_planes = static_cast<int32_t>(plane_segments.size());
      // Mark cells
      int stacked_cell_id = 0;
      for (int32_t row_id = 0; row_id < _nr_vertical_cells; ++row_id){
        auto row = _grid_plane_seg_map.ptr<int32_t>(row_id);
        for (int32_t col_id = 0; col_id < _nr_horizontal_cells; ++col_id){
          if (activation_map[stacked_cell_id]){
            row[col_id] = nr_curr_planes;
          }
          ++stacked_cell_id;
        }
      }
    }
  }

  return plane_segments;
}

std::vector<int32_t> PlaneExtractor::mergePlanes(
    std::vector<std::shared_ptr<PlaneSeg>>& plane_segments) {
  size_t nr_planes = plane_segments.size();
  // Boolean matrix [nr_planes X nr_planes]
  auto planes_association_mx = getConnectedComponents(nr_planes);
  std::vector<int32_t> plane_merge_labels(nr_planes);
  std::iota(plane_merge_labels.begin(), plane_merge_labels.end(), 0);

  // Connect compatible planes
  for (size_t row_id = 0; row_id < nr_planes; ++row_id) {
    int32_t plane_id = plane_merge_labels[row_id];
    bool plane_expanded = false;
    for (size_t col_id = planes_association_mx[row_id]._Find_next(row_id);
         col_id != planes_association_mx[row_id].size();
         col_id = planes_association_mx[row_id]._Find_next(col_id)) {
      double cos_angle = plane_segments[plane_id]->getNormal().dot(
          plane_segments[col_id]->getNormal());
      double distance = pow(plane_segments[plane_id]->getNormal().dot(
                                plane_segments[col_id]->getMean()) +
                                plane_segments[plane_id]->getD(),
                            2);
      if (cos_angle > _config.getFloat("minCosAngleForMerge") &&
          distance < _config.getFloat("maxMergeDist")) {
        (*plane_segments[plane_id]) += (*plane_segments[col_id]);
        plane_merge_labels[col_id] = plane_id;
        plane_expanded = true;
      } else {
        planes_association_mx[row_id][col_id] = false;
      }
    }
    if (plane_expanded) plane_segments[plane_id]->calculateStats();
  }

  return plane_merge_labels;
}

void PlaneExtractor::refinePlanes(
    std::vector<std::shared_ptr<PlaneSeg>> const& plane_segments,
    std::vector<int32_t> const& merge_labels,
    Eigen::MatrixXf const& pcd_array) {
  assert(plane_segments.size() == merge_labels.size());
  std::vector<std::shared_ptr<PlaneSeg>> plane_segments_final;
  cv::Mat mask(_nr_vertical_cells, _nr_horizontal_cells, CV_8U);
  cv::Mat mask_eroded(_nr_vertical_cells, _nr_horizontal_cells, CV_8U);
  //  cv::Mat mask_square_eroded;
  cv::Mat mask_dilated(_nr_vertical_cells, _nr_horizontal_cells, CV_8U);
  cv::Mat mask_square_kernel = cv::Mat::ones(3, 3, CV_8U);
  cv::Mat mask_cross_kernel = cv::Mat::ones(3, 3, CV_8U);
  mask_cross_kernel.at<uchar>(0, 0) = 0;
  mask_cross_kernel.at<uchar>(2, 2) = 0;
  mask_cross_kernel.at<uchar>(0, 2) = 0;
  mask_cross_kernel.at<uchar>(2, 0) = 0;
  for (int32_t i = 0; i < plane_segments.size(); ++i) {
    mask = cv::Scalar(0);
    if (i != merge_labels[i]) continue;
    for (int32_t j = i; j < plane_segments.size(); ++j) {
      if (merge_labels[j] == merge_labels[i]) {
        // TODO: Check ... = j
        mask.setTo(1, _grid_plane_seg_map == j + 1);
      }
    }
    cv::erode(mask, mask_eroded, mask_cross_kernel);
    double min, max;
    cv::minMaxLoc(mask_eroded, &min, &max);

    // Ignore plane if completely eroded
    if (max == 0) {
      continue;
    }
    plane_segments_final.push_back(plane_segments[i]);

    cv::dilate(mask, mask_dilated, mask_square_kernel);
    cv::Mat mask_diff = mask_dilated - mask_eroded;
    _grid_plane_seg_map_eroded.setTo(plane_segments_final.size(),
                                     mask_eroded > 0);
    refineCells(plane_segments[i], plane_segments_final.size(), mask_diff,
                pcd_array);
  }
}

cv::Mat PlaneExtractor::toLabels() {
  cv::Mat seg_out(_image_height, _image_width, CV_8U);
  seg_out = cv::Scalar(0);
  int32_t cell_width = _config.getInt("patchSize");
  int32_t cell_height = _config.getInt("patchSize");
  for (int cell_r = 0; cell_r < _nr_vertical_cells; cell_r++) {
    auto grid_plane_eroded_row_ptr =
        _grid_plane_seg_map_eroded.ptr<label_t>(cell_r);
    int r_offset = cell_r * cell_height;
    int r_limit = r_offset + cell_height;
    for (int cell_c = 0; cell_c < _nr_horizontal_cells; cell_c++) {
      int c_offset = cell_c * cell_width;
      int c_limit = c_offset + cell_width;
      if (grid_plane_eroded_row_ptr[cell_c] > 0) {
        seg_out(cv::Rect(c_offset, r_offset, cell_width, cell_height))
            .setTo(grid_plane_eroded_row_ptr[cell_c]);
      } else {
        // Set cell pixels one by one
        auto idx = _nr_pts_per_cell * cell_r * _nr_horizontal_cells +
                   _nr_pts_per_cell * cell_c;
        for (int r = r_offset; r < r_limit; r++) {
          auto row_ptr = seg_out.ptr<uchar>(r);
          for (int c = c_offset; c < c_limit; c++) {
            if (_seg_map_stacked[idx] > 0) {
              row_ptr[c] = _seg_map_stacked[idx];
            }
            idx++;
          }
        }
      }
    }
  }
  return seg_out;
}

cv::Mat PlaneExtractor::coarseToLabels(std::vector<int32_t> const& labels) {
  cv::Mat_<int32_t> grid_plane_seg_map_merged;
  _grid_plane_seg_map.copyTo(grid_plane_seg_map_merged);

  for (int32_t i = 0; i < labels.size(); ++i) {
    if (labels[i] != i) {
      grid_plane_seg_map_merged.setTo(labels[i],
                                      grid_plane_seg_map_merged == i);
    }
  }

  return grid_plane_seg_map_merged;
}

void PlaneExtractor::cleanArtifacts() {
  _cell_grid.resize(_nr_total_cells, nullptr);
  _grid_plane_seg_map = 0;
  _grid_plane_seg_map_eroded = 0;
  _seg_map_stacked.resize(_image_height * _image_width, 0);
}

void PlaneExtractor::refineCells(const std::shared_ptr<const PlaneSeg> plane,
                                 label_t label, cv::Mat const& mask,
                                 Eigen::MatrixXf const& pcd_array) {
  int32_t stacked_cell_id = 0;
  auto refinement_coeff = _config.getFloat("refinementMultiplierCoeff");
  std::vector<float> distances_stacked(_image_width * _image_height, MAXFLOAT);
  for (int32_t row_id = 0; row_id < _nr_vertical_cells; ++row_id) {
    auto row_ptr = mask.ptr<uchar>(row_id);
    for (int32_t col_id = 0; col_id < _nr_horizontal_cells; ++col_id) {
      if (!row_ptr[col_id]) {
        ++stacked_cell_id;
        continue;
      }
      int32_t offset = stacked_cell_id * _nr_pts_per_cell;
      int32_t next_offset = offset + _nr_pts_per_cell;
      auto max_dist = refinement_coeff * plane->getMSE();
      Eigen::ArrayXf distances_cell_stacked =
          pcd_array.block(offset, 0, _nr_pts_per_cell, 1).array() *
              plane->getNormal()[0] +
          pcd_array.block(offset, 1, _nr_pts_per_cell, 1).array() *
              plane->getNormal()[1] +
          pcd_array.block(offset, 2, _nr_pts_per_cell, 1).array() *
              plane->getNormal()[2] +
          plane->getD();

      for (int pt = offset, j = 0; pt < next_offset; ++j, ++pt) {
        auto dist = powf(distances_cell_stacked(j), 2);
        if (dist < max_dist && dist < distances_stacked[pt]) {
          distances_stacked[pt] = dist;
          _seg_map_stacked[pt] = label;
        }
      }
      ++stacked_cell_id;
    }
  }
}

void PlaneExtractor::growSeed(int32_t x, int32_t y, int32_t prev_index,
                              std::bitset<BITSET_SIZE> const& unassigned,
                              std::bitset<BITSET_SIZE>* activation_map,
                              std::vector<float> const& cell_dist_tols) const {
  int32_t index = x + _nr_horizontal_cells * y;
  if (index >= _nr_total_cells)
    throw std::out_of_range("growSeed: Index out of total cell number");
  if (!unassigned[index] || (*activation_map)[index]) {
    return;
  }

  double d_1 = _cell_grid[prev_index]->getD();
  Eigen::Vector3d normal_1 = _cell_grid[prev_index]->getNormal();
  Eigen::Vector3d normal_2 = _cell_grid[index]->getNormal();
  Eigen::Vector3d mean_2 = _cell_grid[index]->getMean();

  double cos_angle = normal_1.dot(normal_2);
  double merge_dist = pow(normal_1.dot(mean_2) + d_1, 2);
  if (cos_angle < _config.getFloat("minCosAngleForMerge") ||
      merge_dist > cell_dist_tols[index]) {
    return;
  }

  activation_map->set(index);
  if (x > 0)
    growSeed(x - 1, y, index, unassigned, activation_map, cell_dist_tols);
  if (x < _nr_horizontal_cells - 1)
    growSeed(x + 1, y, index, unassigned, activation_map, cell_dist_tols);
  if (y > 0)
    growSeed(x, y - 1, index, unassigned, activation_map, cell_dist_tols);
  if (y < _nr_vertical_cells - 1)
    growSeed(x, y + 1, index, unassigned, activation_map, cell_dist_tols);
}

std::vector<std::bitset<BITSET_SIZE>> PlaneExtractor::getConnectedComponents(
    size_t nr_planes) const {
  std::vector<std::bitset<BITSET_SIZE>> planes_assoc_matrix(nr_planes);

  for (int32_t row_id = 0; row_id < _grid_plane_seg_map.rows - 1; ++row_id) {
    auto row = _grid_plane_seg_map.ptr<int>(row_id);
    auto next_row = _grid_plane_seg_map.ptr<int>(row_id + 1);
    for (int32_t col_id = 0; col_id < _grid_plane_seg_map.cols - 1; ++col_id) {
      auto plane_id = row[col_id];
      if (plane_id > 0) {
        if (row[col_id + 1] > 0 && plane_id != row[col_id + 1])
          planes_assoc_matrix[plane_id - 1][row[col_id + 1] - 1] = true;
        if (next_row[col_id] > 0 && plane_id != next_row[col_id])
          planes_assoc_matrix[plane_id - 1][next_row[col_id] - 1] = true;
      }
    }
  }
  for (int32_t row_id = 0; row_id < planes_assoc_matrix.size(); ++row_id) {
    for (int32_t col_id = 0; col_id < planes_assoc_matrix.size(); ++col_id) {
      planes_assoc_matrix[row_id][col_id] =
          planes_assoc_matrix[row_id][col_id] ||
          planes_assoc_matrix[col_id][row_id];
    }
  }

  return planes_assoc_matrix;
}

#ifdef DEBUG_DEPLEX

template <typename T>
void vectorToCSV(std::vector<std::vector<T>> const& data,
                 std::string const& out_path, char sep = ',') {
  std::ofstream f_out(out_path);
  for (const auto& row : data) {
    for (auto value = row.begin(); value != row.end(); ++value) {
      if (value != row.begin()) {
        f_out << sep;
      }
      f_out << *value;
    }
    f_out << '\n';
  }
}

void PlaneExtractor::planarCellsToLabels(
    std::bitset<BITSET_SIZE> const& planar_flags,
    std::string const& save_path) {
  std::vector<std::vector<int32_t>> labels(
      _image_height, std::vector<int32_t>(_image_width, 0));

  int32_t cell_width = _config.getInt("patchSize");
  int32_t cell_height = _config.getInt("patchSize");

  for (auto cell_id = planar_flags._Find_first();
       cell_id != planar_flags.size();
       cell_id = planar_flags._Find_next(cell_id)) {
    auto cell_row = cell_id / _nr_horizontal_cells;
    auto cell_col = cell_id % _nr_horizontal_cells;
    // Fill cell with label
    auto label_row = cell_row * cell_height;
    auto label_col = cell_col * cell_width;
    for (auto i = label_row; i < label_row + cell_height; ++i) {
      for (auto j = label_col; j < label_col + cell_width; ++j) {
        labels[i][j] = static_cast<int32_t>(cell_id);
      }
    }
  }

  vectorToCSV(labels, save_path);
}

void PlaneExtractor::planeSegmentsMapToLabels(
    std::string const& save_path, cv::Mat_<int32_t> const& cell_map) {
  std::vector<std::vector<int32_t>> labels(
      _image_height, std::vector<int32_t>(_image_width, 0));

  int32_t cell_width = _config.getInt("patchSize");
  int32_t cell_height = _config.getInt("patchSize");

  int32_t stacked_cell_id = 0;
  for (auto row = 0; row < cell_map.rows; ++row) {
    for (auto col = 0; col < cell_map.cols; ++col) {
      auto cell_row = stacked_cell_id / _nr_horizontal_cells;
      auto cell_col = stacked_cell_id % _nr_horizontal_cells;
      // Fill cell with label
      auto label_row = cell_row * cell_height;
      auto label_col = cell_col * cell_width;
      for (auto i = label_row; i < label_row + cell_height; ++i) {
        for (auto j = label_col; j < label_col + cell_width; ++j) {
          labels[i][j] = cell_map[row][col];
        }
      }
      ++stacked_cell_id;
    }
  }

  vectorToCSV(labels, save_path);
}

void PlaneExtractor::mergeSegmentsToLabels(
    std::string const& save_path, std::vector<int32_t> const& merge_labels) {
  cv::Mat_<int32_t> grid_plane_seg_map_merged;
  _grid_plane_seg_map.copyTo(grid_plane_seg_map_merged);

  for (int32_t i = 0; i < merge_labels.size(); ++i) {
    if (merge_labels[i] != i) {
      grid_plane_seg_map_merged.setTo(merge_labels[i],
                                      grid_plane_seg_map_merged == i);
    }
  }

  planeSegmentsMapToLabels(save_path, grid_plane_seg_map_merged);
}
#endif

}  // namespace deplex
