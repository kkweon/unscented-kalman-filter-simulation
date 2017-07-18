#include "ukf.h"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  n_x_ = 5;
  n_aug_ = 7;
  is_initialized_ = false;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */

  if (!is_initialized_) {
    double px, py, v, psi, psi_dot;

    if (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) {
      // LIDAR
      px = meas_package.raw_measurements_[0];
      py = meas_package.raw_measurements_[1];
      v = 0;
      psi = 0;
      psi_dot = 0;

    } else if (use_radar_ &&
               meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      // RADAR
      double rho = meas_package.raw_measurements_[0];
      double phi = meas_package.raw_measurements_[1];
      double rho_dot = meas_package.raw_measurements_[2];

      px = rho * std::cos(phi);
      py = rho * std::sin(phi);

      double vx = rho_dot * std::cos(phi);
      double vy = rho_dot * std::sin(phi);

      v = sqrt(square(vx) + square(vy));

      auto numerator_term =
          sqrt(-(square(px) + square(py)) * (rho_dot - v) * (rho_dot + v));
      auto denominator = px * v + rho_dot * sqrt(square(px) + square(py));

      auto possible_psi1_before_atan = (py * v - numerator_term) / denominator;
      auto possible_psi2_before_atan = (py * v + numerator_term) / denominator;
      auto possible_psi1 = 2 * atan(possible_psi1_before_atan);
      possible_psi1 = normalize_PI(possible_psi1);
      auto possible_psi2 = 2 * atan(possible_psi2_before_atan);
      possible_psi2 = normalize_PI(possible_psi2);
      psi = fmin(fabs(possible_psi1), fabs(possible_psi2));
      psi_dot = 0;

    } else {
      throw std::invalid_argument("You cannot turn off both Laser and Radar");
    }

    x_ << px, py, v, psi, psi_dot;

    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance
  matrix.
  */
}

/**
 * Updates the state and the state covariance matrix using a laser
 * measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the
  object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}

/**
 * Updates the state and the state covariance matrix using a radar
 * measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the
  object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
}

MatrixXd UKF::GenerateSigmaPoints() {
  // X_sig = [x, x + sqrt((lambda + n_x) * P), x - sqrt((lambda + n_x) * P)]
  MatrixXd X_sig(n_x_, 2 * n_x_ + 1);
  lambda_ = 3 - n_x_;
  MatrixXd sqrt_P = P_.llt().matrixL();
  MatrixXd sqrt_term = sqrt(lambda_ + n_x_) * sqrt_P;

  X_sig.col(0) = x_;
  for (int i = 0; i < n_x_; ++i) {
    X_sig.col(i + 1) = x_ + sqrt_term.col(i);
    X_sig.col(i + n_x_ + 1) = x_ - sqrt_term.col(i);
  }

  return X_sig;
}

MatrixXd UKF::GenerateAugmentedSigmaPoints() {
  MatrixXd Xsig_aug(n_aug_, 2 * n_aug_ + 1);
  Xsig_aug.setZero();
  lambda_ = 3 - n_aug_;

  VectorXd x_aug(n_aug_);
  MatrixXd P_aug(n_aug_, n_aug_);

  x_aug.setZero();
  x_aug.head(n_x_) = x_;

  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  int n_new = n_aug_ - n_x_;

  MatrixXd Q = MatrixXd::Zero(n_new, n_new);
  Q(0, 0) = square(std_a_);
  Q(1, 1) = square(std_yawdd_);
  P_aug.bottomRightCorner(n_new, n_new) = Q;

  MatrixXd sqrt_P_aug = P_aug.llt().matrixL();
  MatrixXd sqrt_term = sqrt(lambda_ + n_aug_) * sqrt_P_aug;

  Xsig_aug.col(0) = x_aug;

  for (int i = 0; i < n_aug_; ++i) {
    Xsig_aug.col(i + 1) = x_aug + sqrt_term.col(i);
    Xsig_aug.col(i + n_aug_ + 1) = x_aug - sqrt_term.col(i);
  }

  return Xsig_aug;
}

MatrixXd UKF::PredictSigmaPoints(const MatrixXd &Xsig_aug,
                                 const double delta_t) {
  MatrixXd result(n_x_, 2 * n_aug_ + 1);

  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    result.col(i) = PredictOneSigmaPoint(Xsig_aug.col(i), delta_t);
  }

  return result;
}

VectorXd UKF::PredictOneSigmaPoint(const VectorXd &sigma_point,
                                   const double delta_t) {

  if (sigma_point.size() == 7) {
    double px = sigma_point(0);
    double py = sigma_point(1);
    double v = sigma_point(2);
    double psi = sigma_point(3);
    double psi_dot = sigma_point(4);
    double nu_a = sigma_point(5);
    double nu_psi_dot = sigma_point(6);

    VectorXd x(5);

    if (psi_dot != 0) {
      x(0) = px + v / psi_dot * (sin(psi + psi_dot * delta_t) - sin(psi)) +
             0.5 * square(delta_t) * cos(psi) * nu_a;
      x(1) = py + v / psi_dot * (-cos(psi + psi_dot * delta_t) + cos(psi)) +
             0.5 * square(delta_t) * sin(psi) * nu_a;
    } else {
      x(0) = px + v * (v * cos(psi) * delta_t) +
             0.5 * square(delta_t) * cos(psi) * nu_a;
      x(1) = py + v * (v * sin(psi) * delta_t) +
             0.5 * square(delta_t) * sin(psi) * nu_a;
    }
    x(2) = v + 0 + delta_t * nu_a;
    x(3) = psi + psi_dot * delta_t + 0.5 * square(delta_t) * nu_psi_dot;
    x(4) = psi_dot + 0 + delta_t * nu_psi_dot;

    return x;

  } else {
    throw std::invalid_argument("Type of SigmaPoint is wrong");
  }
}

VectorXd UKF::PredictMean(const MatrixXd &Xsig_pred) {
  lambda_ = 3 - n_aug_;
  VectorXd weights(2 * n_aug_ + 1);
  double w1 = lambda_ / (lambda_ + n_aug_);
  double w2 = 1 / (2 * (lambda_ + n_aug_));
  weights.setConstant(w2);
  weights(0) = w1;

  return Xsig_pred * weights;
}

MatrixXd UKF::PredictCovariance(const MatrixXd &Xsig_pred,
                                const VectorXd &mean) {
  // Xsig_pred (2-D Array): shape (n_x, 2 * n_aug + 1)

  if (mean.size() != Xsig_pred.rows()) {
    throw std::invalid_argument("Wrong size mean");
  }

  double w1 = lambda_ / (lambda_ + n_aug_);
  double w2 = 1 / (2 * (lambda_ + n_aug_));
  MatrixXd mean_subtracted = Xsig_pred.colwise() - mean;

  // check psi  between -pi and pi
  for (int i = 0; i < mean_subtracted.cols(); ++i) {
    auto psi = mean_subtracted(3, i);
    psi = normalize_PI(psi);
    mean_subtracted(3, i) = psi;
  }

  MatrixXd temp = mean_subtracted * w2;
  temp.col(0) = temp.col(0) / w2 * w1;
  return temp * mean_subtracted.transpose();
}
