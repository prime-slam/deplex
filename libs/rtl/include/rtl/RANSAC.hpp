#ifndef PLANE_RANSAC_HEADER
#define PLANE_RANSAC_HEADER

#include "Base.hpp"
#include <random>
#include <cmath>
#include <cassert>

namespace RTL
{

class PlaneRANSAC
{
public:
    PlaneRANSAC(Estimator<Eigen::Vector4f, Eigen::Vector3f, Eigen::MatrixX3f>* estimator)
    {
        assert(estimator != NULL);

        toolEstimator = estimator;
        SetParamIteration();
        SetParamThreshold();
        SetParamTargetInliersRatio();
    }

    virtual double FindBest(Eigen::Vector4f& best, const Eigen::MatrixX3f& data, int N, int M)
    {
        assert(N > 0 && M > 0);

        Initialize(data, N);

        // Run PlaneRANSAC
        double bestloss = HUGE_VAL;
        int iteration = 0;
        while (IsContinued(iteration, N - bestloss, N))
        {
            iteration++;

            // 1. Generate hypotheses
            Eigen::Vector4f model = GenerateModel(data, M);

            // 2. Evaluate the hypotheses
            double loss = EvaluateModel(model, data, N);
            if (loss < bestloss)
                if (!UpdateBest(best, bestloss, model, loss))
                    goto RANSAC_FIND_BEST_EXIT;
        }

RANSAC_FIND_BEST_EXIT:
        Terminate(best, data, N);
        return bestloss;
    }

    virtual std::vector<int> FindInliers(const Eigen::Vector4f& model, const Eigen::MatrixX3f& data, int N)
    {
        std::vector<int> inliers;
        for (int i = 0; i < N; i++)
        {
            double error = toolEstimator->ComputeError(model, data.row(i));
            if (fabs(error) < paramThreshold) inliers.push_back(i);
        }
        return inliers;
    }

    void SetParamIteration(int iteration = 100) { paramIteration = iteration; }

    int GetParamIteration(void) { return paramIteration; }

    void SetParamThreshold(double threshold = 1) { paramThreshold = threshold; }

    int GetParamThreshold(void) { return paramThreshold; }

    void SetParamTargetInliersRatio(double target = 1) { paramTargetInliersRatio = target; }

    double GetParamTargetInliersRatio(void) { return paramTargetInliersRatio; }

protected:
    virtual bool IsContinued(int iteration, int inliers_count, int sample_size) { 
      return (iteration < paramIteration && inliers_count < paramTargetInliersRatio * sample_size); 
    }

    virtual Eigen::Vector4f GenerateModel(const Eigen::MatrixX3f& data, int M)
    {
        std::set<int> samples;
        while (static_cast<int>(samples.size()) < M)
            samples.insert(toolUniform(toolGenerator));
        return toolEstimator->ComputeModel(data, samples);
    }

    virtual double EvaluateModel(const Eigen::Vector4f& model, const Eigen::MatrixX3f& data, int N)
    {
        double loss = 0;
        for (int i = 0; i < N; i++)
        {
            double error = toolEstimator->ComputeError(model, data.row(i));
            loss += (fabs(error) >= paramThreshold);
        }
        return loss;
    }

    virtual bool UpdateBest(Eigen::Vector4f& bestModel, double& bestCost, const Eigen::Vector4f& model, double cost)
    {
        bestModel = model;
        bestCost = cost;
        return true;
    }

    virtual void Initialize(const Eigen::MatrixX3f& data, int N) { toolUniform = std::uniform_int_distribution<int>(0, N - 1); }

    virtual void Terminate(const Eigen::Vector4f& bestModel, const Eigen::MatrixX3f& data, int N) { }

    std::mt19937 toolGenerator;

    std::uniform_int_distribution<int> toolUniform;

    Estimator<Eigen::Vector4f, Eigen::Vector3f, Eigen::MatrixX3f>* toolEstimator;

    int paramSampleSize;

    int paramIteration;

    double paramThreshold;

    double paramTargetInliersRatio;
}; // End of 'PlaneRANSAC'

} // End of 'RTL'

#endif // End of '__RTL_RANSAC__'
