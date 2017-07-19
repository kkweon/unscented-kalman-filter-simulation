#ifndef UKF_H
#define UKF_H

#include "Eigen/Dense"
#include "measurement_package.h"
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// For Debugging
template <typename T> inline void print(T val, std::string name) {
  std::cout << "===========================" << std::endl;
  std::cout << "[" << name << "]" << std::endl;
  std::cout << val << std::endl;
}

class UKF {
public:
  ///* initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  ///* if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  ///* if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  VectorXd x_;

  ///* state covariance matrix
  MatrixXd P_;

  ///* predicted sigma points matrix
  MatrixXd Xsig_pred_;

  ///* time when the state is true, in us
  long long time_us_;

  ///* Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  ///* Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  ///* Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  ///* Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  ///* Radar measurement noise standard deviation radius in m
  double std_radr_;

  ///* Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  ///* Radar measurement noise standard deviation radius change in m/s
  double std_radrd_;

  ///* Weights of sigma points
  VectorXd weights_;

  ///* State dimension
  int n_x_;

  ///* Augmented state dimension
  int n_aug_;

  ///* Sigma point spreading parameter
  double lambda_;


  // Normalized Innovation Squared (NIS)
  double radar_NIS_;

  // Normalized Innovation Squared (NIS)
  double laser_NIS_;

  VectorXd backup_x_;
  MatrixXd backup_P_;

  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage meas_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(MeasurementPackage meas_package);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(MeasurementPackage meas_package);

  /**
   * Generates (augmented) sigma points
   * @return X_sig Matrix of shape (n_x, 2 * n_x + 1)
   */
  MatrixXd GenerateSigmaPoints();
  MatrixXd GenerateAugmentedSigmaPoints();

  /**
   * Predicts sigma points
   * @param Xsig_aug
   * @param delta_t
   * @return MatrixXd sigma_points
   */
  MatrixXd PredictSigmaPoints(const MatrixXd &Xsig_aug, const double delta_t);
  VectorXd PredictOneSigmaPoint(const VectorXd &sigma_point,
                                const double delta_t);

  /**
   * Returns a mean of sigma points
   * @param Xsig_pred
   * @return mean (n_x)
   */
  VectorXd PredictMean(const MatrixXd &Xsig_pred);

  /**
   * Returnsa a covariance of sigma points
   * @param Xsig_pred
   * @param mean
   * @return covariance (n_x, 2 * n_aug + 1)
   */
  MatrixXd PredictCovariance(const MatrixXd &Xsig_pred, const VectorXd &mean);

  /**
   * Predicts Measurements
   * @return tuple<z_predictions, S, Z_sigma_points> measurement prediction and covariance
   */
  std::tuple<VectorXd, MatrixXd, MatrixXd> PredictRadarMeasurement();
  std::tuple<VectorXd, MatrixXd, MatrixXd> PredictLaserMeasurement();

  /**
   * Normalizes a PI to between -pi and pi
   * @param theta
   * @return normalized theta
   */
  inline double normalize_PI(double theta) {
    if (-M_PI < theta && theta < M_PI) {
      return theta;
    }
    double val = theta - M_2_PI * std::floor((theta + M_PI) / M_2_PI);
    return val;
  }
  template <typename T> inline T square(T val) { return val * val; };
  bool is_unstable();

};

#endif /* UKF_H */
