#pragma once

#include <Eigen/Core>
#include <bitset>
#include <vector>

#ifndef BITSET_SIZE
#define BITSET_SIZE 65536  // 2^16
#endif

namespace deplex {
class Histogram {
 public:
  Histogram(int32_t nr_bins_per_coord, Eigen::MatrixXd const& P,
            std::bitset<BITSET_SIZE> const& mask);
  std::vector<int32_t> getPointsFromMostFrequentBin() const;
  void removePoint(int32_t point_id);

 private:
  std::vector<int32_t> _bins;
  std::vector<int32_t> _hist;
  int32_t _nr_bins_per_coord;
  int32_t _nr_bins;
  int32_t _nr_points;
};
}  // namespace deplex