#include "Eigen/Dense"
#include "tools.h"
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

TEST(Tools, ReturnsOneWhenDifferBy1) {
  Tools tool;

  VectorXd same_value(4);
  same_value.fill(0);

  VectorXd another_value(4);
  another_value.fill(1);

  VectorXd expected_vector(4);
  expected_vector.fill(1);

  std::vector<VectorXd> estimations = {same_value};
  std::vector<VectorXd> ground_truth = {another_value};

  VectorXd result = tool.CalculateRMSE(estimations, ground_truth);

  VectorXd diff = result - expected_vector;
  ASSERT_LE(diff.squaredNorm(), 1e-7);
}

TEST(Tools, ReturnsRMSE) {
  Tools tool;
  VectorXd temp1(4);
  temp1 << 1, 0, 0, 0;
  VectorXd temp2(4);
  temp2 << 2, 0, 1, 0;
  VectorXd temp3(4);
  temp3 << 3, 2, 4, 3;

  std::vector<VectorXd> estimations{temp1, temp2, temp3};

  temp1 << 0, 1, 0, 0;
  temp2 << 1, 0, 2, 3;
  temp3 << 0, 1, 5, 2;

  std::vector<VectorXd> ground_truths{temp1, temp2, temp3};

  VectorXd expected(4);
  expected << 11. / 3., 2. / 3., 2. / 3., 10. / 3.;
  expected = expected.array().sqrt();

  VectorXd result = tool.CalculateRMSE(estimations, ground_truths);
  VectorXd diff = result - expected;
  ASSERT_LE(diff.squaredNorm(), 1e-7);
}
