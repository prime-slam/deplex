#pragma once

#include <Eigen/Core>
#include <bitset>
#include <cstdint>
#include <vector>

#ifndef BITSET_SIZE
#define BITSET_SIZE 65536  // 2^16
#endif

namespace cape {
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

Histogram::Histogram(int32_t nr_bins_per_coord, Eigen::MatrixXd const& P,
                     std::bitset<BITSET_SIZE> const& mask)
    : _nr_bins_per_coord(nr_bins_per_coord),
      _nr_bins(nr_bins_per_coord * nr_bins_per_coord),
      _nr_points(static_cast<int32_t>(P.rows())),
      _hist(nr_bins_per_coord * nr_bins_per_coord, 0),
      _bins(static_cast<int32_t>(P.rows()), -1) {
  // Set limits
  // Polar angle [0 pi]
  double min_X(0), max_X(M_PI);
  // Azimuth angle [-pi pi]
  double min_Y(-M_PI), max_Y(M_PI);

  int X_q(0), Y_q(0);
  for (size_t i = mask._Find_first(); i != mask.size();
       i = mask._Find_next(i)) {
    X_q = static_cast<int>((_nr_bins_per_coord - 1) * (P(i, 0) - min_X) /
                           (max_X - min_X));
    if (X_q > 0) {
      Y_q = static_cast<int>((_nr_bins_per_coord - 1) * (P(i, 1) - min_Y) /
                             (max_Y - min_Y));
    } else {
      Y_q = 0;
    }
    int bin = Y_q * _nr_bins_per_coord + X_q;
    _bins[i] = bin;
    ++_hist[bin];
  }
}

std::vector<int32_t> Histogram::getPointsFromMostFrequentBin() const {
  std::vector<int32_t> point_ids;

  auto most_frequent_bin = std::max_element(_hist.begin(), _hist.end());
  int32_t max_nr_occurrences = *most_frequent_bin;
  size_t max_bin_id = std::distance(_hist.begin(), most_frequent_bin);

  if (max_nr_occurrences > 0) {
    for (int32_t i = 0; i < _nr_points; ++i) {
      if (_bins[i] == max_bin_id) {
        point_ids.push_back(i);
      }
    }
  }

  return point_ids;
}

void Histogram::removePoint(int32_t point_id) {
  --_hist[_bins[point_id]];
  _bins[point_id] = -1;
}
}  // namespace cape