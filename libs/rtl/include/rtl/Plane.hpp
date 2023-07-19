#ifndef RTL_PLANE_HEADER
#define RTL_PLANE_HEADER

#include <rtl/Base.hpp>
#include <cmath>
#include <limits>
#include <random>

class Point
{
public:
    Point() : x_(0), y_(0), z_(0) {}

    Point(float x, float y, float z) : x_(x), y_(y), z_(z) { }

    float x_, y_, z_;
};

class Plane
{
public: 
    Plane() : a_(0), b_(0), c_(0), d_(0) {}

    Plane(float a, float b, float c, float d) : a_(a), b_(b), c_(c), d_(d) {}
    
    friend std::ostream& operator<<(std::ostream& out, const Plane& plane) {
      return out << plane.a_ << ' ' << plane.b_ << ' ' << plane.c_ << ' ' << plane.d_;
    }

    float a_, b_, c_, d_;
};

class PlaneEstimator : virtual public RTL::Estimator<Plane, Point, std::vector<Point> >
{
public:
    virtual Plane ComputeModel(const std::vector<Point>& data, const std::set<int>& samples)
    {
      auto it = samples.begin();

      Point p0 = data[*it++];
      Point p1 = data[*it++];
      Point p2 = data[*it++];

      float x0 = p0.x_;
      float x1 = p1.x_;
      float x2 = p2.x_;

      float y0 = p0.y_;
      float y1 = p1.y_;
      float y2 = p2.y_;

      float z0 = p0.z_;
      float z1 = p1.z_;
      float z2 = p2.z_;

      float a = (z0*(y1 - y2)) / (x0*y1 - x1*y0 - x0*y2 + x2*y0 + x1*y2 - x2*y1) - (z1*(y0 - y2)) / (x0*y1 - x1*y0 - x0*y2 + x2*y0 + x1*y2 - x2*y1) + (z2*(y0 - y1)) / (x0*y1 - x1*y0 - x0*y2 + x2*y0 + x1*y2 - x2*y1);
		  float b = (z1*(x0 - x2)) / (x0*y1 - x1*y0 - x0*y2 + x2*y0 + x1*y2 - x2*y1) - (z0*(x1 - x2)) / (x0*y1 - x1*y0 - x0*y2 + x2*y0 + x1*y2 - x2*y1) - (z2*(x0 - x1)) / (x0*y1 - x1*y0 - x0*y2 + x2*y0 + x1*y2 - x2*y1);
		  float d = (z2*(x0*y1 - x1*y0)) / (x0*y1 - x1*y0 - x0*y2 + x2*y0 + x1*y2 - x2*y1) - (z1*(x0*y2 - x2*y0)) / (x0*y1 - x1*y0 - x0*y2 + x2*y0 + x1*y2 - x2*y1) + (z0*(x1*y2 - x2*y1)) / (x0*y1 - x1*y0 - x0*y2 + x2*y0 + x1*y2 - x2*y1);
		  float c = -1.0;

      float l = sqrt(a * a + b * b + c * c);

      return Plane(a / l, b / l, c / l, d / l);
    }

    virtual double ComputeError(const Plane& plane, const Point& point)
    {
        return plane.a_ * point.x_ + plane.b_ * point.y_ + plane.c_ * point.z_ + plane.d_;
    }
}; // End of 'PlaneEstimator'

#endif
