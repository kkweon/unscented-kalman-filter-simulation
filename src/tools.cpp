#include "tools.h"
#include <exception>
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  if (estimations.empty() || ground_truth.empty()) {
    throw std::invalid_argument("Empty Vectors are given");
  }
  int n_dim = estimations[0].size();
  VectorXd RMSE(n_dim);
  RMSE.setZero();

  for (auto i = 0; i < estimations.size(); ++i) {
    VectorXd diff = estimations[i] - ground_truth[i];
    VectorXd diff_sq = diff.array().square();
    RMSE += diff_sq;
  }

  RMSE /= estimations.size();
  RMSE = RMSE.array().sqrt();

  return RMSE;
}
