#include "deplex/algorithm/histogram.h"

namespace deplex {
Histogram::Histogram(int32_t nr_bins_per_coord, Eigen::MatrixXd const& P,
                     std::bitset<BITSET_SIZE> const& mask)
    : nr_bins_per_coord_(nr_bins_per_coord),
      nr_points_(static_cast<int32_t>(P.rows())),
      hist_(nr_bins_per_coord * nr_bins_per_coord, 0),
      bins_(static_cast<int32_t>(P.rows()), -1) {
  // Set limits
  // Polar angle [0 pi]
  double min_X(0), max_X(M_PI);
  // Azimuth angle [-pi pi]
  double min_Y(-M_PI), max_Y(M_PI);

  int X_q(0), Y_q(0);
  for (size_t i = mask._Find_first(); i != mask.size();
       i = mask._Find_next(i)) {
    X_q = static_cast<int>((nr_bins_per_coord_ - 1) * (P(i, 0) - min_X) /
                           (max_X - min_X));
    if (X_q > 0) {
      Y_q = static_cast<int>((nr_bins_per_coord_ - 1) * (P(i, 1) - min_Y) /
                             (max_Y - min_Y));
    } else {
      Y_q = 0;
    }
    int bin = Y_q * nr_bins_per_coord_ + X_q;
    bins_[i] = bin;
    ++hist_[bin];
  }
}

std::vector<int32_t> Histogram::getPointsFromMostFrequentBin() const {
  std::vector<int32_t> point_ids;

  auto most_frequent_bin = std::max_element(hist_.begin(), hist_.end());
  int32_t max_nr_occurrences = *most_frequent_bin;
  size_t max_bin_id = std::distance(hist_.begin(), most_frequent_bin);

  if (max_nr_occurrences > 0) {
    for (int32_t i = 0; i < nr_points_; ++i) {
      if (bins_[i] == max_bin_id) {
        point_ids.push_back(i);
      }
    }
  }

  return point_ids;
}

void Histogram::removePoint(int32_t point_id) {
  --hist_[bins_[point_id]];
  bins_[point_id] = -1;
}
}  // namespace deplex