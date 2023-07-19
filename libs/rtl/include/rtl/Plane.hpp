#ifndef RTL_PLANE_HEADER
#define RTL_PLANE_HEADER

#include "Base.hpp"
#include <cmath>
#include <limits>
#include <random>


class PlaneEstimator : virtual public RTL::Estimator<Eigen::Vector4f, Eigen::Vector3f, Eigen::MatrixX3f>
{
public:
    virtual Eigen::Vector4f ComputeModel(const Eigen::MatrixX3f& data, const std::set<int>& samples)
    {
      auto it = samples.begin();

      auto p0 = data.row(*it++);
      auto p1 = data.row(*it++);
      auto p2 = data.row(*it++);

      float x0 = p0[0];
      float x1 = p1[0];
      float x2 = p2[0];

      float y0 = p0[1];
      float y1 = p1[1];
      float y2 = p2[1];

      float z0 = p0[2];
      float z1 = p1[2];
      float z2 = p2[2];

      float a = (z0*(y1 - y2)) / (x0*y1 - x1*y0 - x0*y2 + x2*y0 + x1*y2 - x2*y1) - (z1*(y0 - y2)) / (x0*y1 - x1*y0 - x0*y2 + x2*y0 + x1*y2 - x2*y1) + (z2*(y0 - y1)) / (x0*y1 - x1*y0 - x0*y2 + x2*y0 + x1*y2 - x2*y1);
		  float b = (z1*(x0 - x2)) / (x0*y1 - x1*y0 - x0*y2 + x2*y0 + x1*y2 - x2*y1) - (z0*(x1 - x2)) / (x0*y1 - x1*y0 - x0*y2 + x2*y0 + x1*y2 - x2*y1) - (z2*(x0 - x1)) / (x0*y1 - x1*y0 - x0*y2 + x2*y0 + x1*y2 - x2*y1);
		  float d = (z2*(x0*y1 - x1*y0)) / (x0*y1 - x1*y0 - x0*y2 + x2*y0 + x1*y2 - x2*y1) - (z1*(x0*y2 - x2*y0)) / (x0*y1 - x1*y0 - x0*y2 + x2*y0 + x1*y2 - x2*y1) + (z0*(x1*y2 - x2*y1)) / (x0*y1 - x1*y0 - x0*y2 + x2*y0 + x1*y2 - x2*y1);
		  float c = -1.0;

      float l = sqrt(a * a + b * b + c * c);

      Eigen::Vector4f plane_model;
      plane_model << a / l, b / l, c / l, d / l;

      return plane_model;
    }

    virtual double ComputeError(const Eigen::Vector4f& plane, const Eigen::Vector3f& point)
    {
        return plane[0] * point[0] + plane[1] * point[1] + plane[2] * point[2] + plane[3];
    }
}; // End of 'PlaneEstimator'

#endif
