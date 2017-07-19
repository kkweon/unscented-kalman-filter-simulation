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
  // Should be less than 5.991
  std_a_ = 0.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  // Should be less than 7.8
  std_yawdd_ = 0.5;

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

  // Initialization
  n_x_ = 5;
  n_aug_ = 7;
  is_initialized_ = false;
  Xsig_pred_.setZero();
  lambda_ = 3 - n_aug_;

  // This weights will be used to compute mean and covariance
  double w1 = lambda_ / (lambda_ + n_aug_);
  double w2 = 0.5 / (lambda_ + n_aug_);
  weights_.setZero(2 * n_aug_ + 1);
  weights_.setConstant(w2);
  weights_(0) = w1;
}

UKF::~UKF() {}

/**
 * Main function will call this
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (!is_initialized_) {
    double px, py, v, psi, psi_dot;

    if (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) {
      // IF LIDAR, only positions
      px = meas_package.raw_measurements_[0];
      py = meas_package.raw_measurements_[1];
      v = 0;
      psi = 0;
      psi_dot = 0;

    } else if (use_radar_ &&
               meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      // IF RADAR, except psi_dot
      // rho : angle
      // v : velocity
      // rho_dot : change of angle
      double rho = meas_package.raw_measurements_[0];
      double phi = meas_package.raw_measurements_[1];
      double rho_dot = meas_package.raw_measurements_[2];

      px = rho * std::cos(phi);
      py = rho * std::sin(phi);

      double vx = rho_dot * std::cos(phi);
      double vy = rho_dot * std::sin(phi);

      v = sqrt(square(vx) + square(vy));

      psi = 0;
      psi_dot = 0;

    } else {
      throw std::invalid_argument("You cannot turn off both LASER and RADAR");
    }
    x_ << px, py, v, psi, psi_dot;
    P_.setIdentity();

    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;

    backup_x_ = x_;
    backup_P_ = P_;

    return;
  } // !is_initialized

  double delta_t = (meas_package.timestamp_ - time_us_) / 1'000'000.0;
  Prediction(delta_t);

  if (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);

  } else if (use_radar_ &&
             meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);

  } else {
    throw std::invalid_argument("You cannot turn off both LASER and RADAR");
  }
  if (is_unstable()) {
    is_initialized_ = false;
    x_ = backup_x_;
    P_ = backup_P_;
  } else {
    backup_x_ = x_;
    backup_P_ = P_;
  }

  time_us_ = meas_package.timestamp_;
}

// 1. Generates Augmented Sigma Points
// 2. Predicts using the sigma points
// 3. Compute Mean and Covariance
void UKF::Prediction(double delta_t) {

  Xsig_pred_ = GenerateAugmentedSigmaPoints();
  Xsig_pred_ = PredictSigmaPoints(Xsig_pred_, delta_t);

  x_ = PredictMean(Xsig_pred_);
  P_ = PredictCovariance(Xsig_pred_, x_);

  if (!(std::isfinite(x_.sum()) && std::isfinite(P_.sum()))) {
    is_initialized_ = false;
  }
}

// 1. Predict Radar measurements
// 2. Returns measurement prediction (z_pred)
// 3. Returns measurement covariance (S)
// 4. Returns measurement sigma points (z_sig)
std::tuple<VectorXd, MatrixXd, MatrixXd> UKF::PredictRadarMeasurement() {
  int n_z = 3;

  VectorXd z_ = VectorXd::Zero(n_z);
  MatrixXd S_ = MatrixXd::Zero(n_z, n_z);
  MatrixXd z_sig = MatrixXd::Zero(n_z, 2 * n_aug_ + 1);

  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double psi = Xsig_pred_(3, i);

    double v1 = cos(psi) * v;
    double v2 = sin(psi) * v;

    // rho
    z_sig(0, i) = sqrt(square(px) + square(py));
    // psi
    z_sig(1, i) = atan2(py, px);
    // rho_dot
    z_sig(2, i) = (px * v1 + py * v2) / z_sig(0, i);
  }

  z_ = z_sig * weights_;

  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd z_diff = z_sig.col(i) - z_;

    // fix psi angle
    z_diff(1) = normalize_PI(z_diff(1));

    S_ = S_ + weights_(i) * z_diff * z_diff.transpose();
  }
  MatrixXd R = MatrixXd::Zero(n_z, n_z);
  R(0, 0) = square(std_radr_);
  R(1, 1) = square(std_radphi_);
  R(2, 2) = square(std_radrd_);
  S_ += R;

  return std::make_tuple(z_, S_, z_sig);
}

// Similar to PredictRadarMeasurement
std::tuple<VectorXd, MatrixXd, MatrixXd> UKF::PredictLaserMeasurement() {
  int n_z = 2;
  VectorXd z_ = VectorXd::Zero(n_z);
  MatrixXd S_ = MatrixXd::Zero(n_z, n_z);

  MatrixXd z_sig(n_z, 2 * n_aug_ + 1);

  z_sig = Xsig_pred_.topRows(n_z);
  z_ = z_sig * weights_;

  MatrixXd diff = z_sig.colwise() - z_;
  MatrixXd weighted_diff =
      diff.array().rowwise() * weights_.transpose().array();

  S_ = weighted_diff * diff.transpose();

  MatrixXd R = MatrixXd::Zero(n_z, n_z);
  R(0, 0) = square(std_laspx_);
  R(1, 1) = square(std_laspy_);
  S_ += R;

  return std::make_tuple(z_, S_, z_sig);
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
     1) Compute T
     2) Compute K
     3) Update x_ and P_
  */
  VectorXd z_pred;
  MatrixXd S_;
  MatrixXd z_sig;
  std::tie(z_pred, S_, z_sig) = PredictLaserMeasurement();

  // Compute T
  MatrixXd X_diff = Xsig_pred_.colwise() - x_;
  MatrixXd Z_diff = z_sig.colwise() - z_pred;

  MatrixXd weighted_X_diff =
      X_diff.array().rowwise() * weights_.transpose().array();
  MatrixXd T = weighted_X_diff * Z_diff.transpose();

  MatrixXd K = T * S_.inverse();
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

  x_ = x_ + K * z_diff;
  P_ = P_ - K * S_ * K.transpose();

  laser_NIS_ = z_diff.transpose() * S_.inverse() * z_diff;
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
     1) Compute T
     2) Compute K
     3) Update x_ and P_
  */
  VectorXd z_pred;
  MatrixXd S_;
  MatrixXd z_sig;
  std::tie(z_pred, S_, z_sig) = PredictRadarMeasurement();

  // Compute T
  MatrixXd X_diff = Xsig_pred_.colwise() - x_;
  MatrixXd Z_diff = z_sig.colwise() - z_pred;
  for (auto i = 0; i < Z_diff.cols(); ++i) {
    // yaw
    Z_diff(1, i) = normalize_PI(Z_diff(1, i));
    // yaw
    X_diff(3, i) = normalize_PI(X_diff(3, i));
  }

  MatrixXd weighted_X_diff =
      X_diff.array().rowwise() * weights_.transpose().array();
  MatrixXd T = weighted_X_diff * Z_diff.transpose();

  MatrixXd K = T * S_.inverse();
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;
  // angle normalization
  z_diff(1) = normalize_PI(z_diff(1));

  x_ = x_ + K * z_diff;
  P_ = P_ - K * S_ * K.transpose();

  radar_NIS_ = z_diff.transpose() * S_.inverse() * z_diff;
}

// Generates unaugmented sigma points
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

// Generates augmented sigma points
MatrixXd UKF::GenerateAugmentedSigmaPoints() {
  MatrixXd Xsig_aug(n_aug_, 2 * n_aug_ + 1);
  Xsig_aug.setZero();

  VectorXd x_aug(n_aug_);
  MatrixXd P_aug(n_aug_, n_aug_);
  P_aug.setZero();

  x_aug.setZero();
  x_aug.head(n_x_) = x_;

  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  int n_new = n_aug_ - n_x_;

  MatrixXd Q = MatrixXd::Zero(n_new, n_new);
  Q(0, 0) = square(std_a_);
  Q(1, 1) = square(std_yawdd_);
  P_aug.bottomRightCorner(n_new, n_new) = Q;

  MatrixXd sqrt_P_aug = P_aug.llt().matrixL();

  if (std::isnan(sqrt_P_aug.sum())) {
    print(P_aug, "P_aug");
    print(sqrt_P_aug, "sqrt_P_aug");
  }

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
  // Returns:
  //     SigmaPoints (2-D Array): shape (n_x, 2 * n_aug + 1);
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

    if (fabs(psi_dot) > 1e-2) {

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
  // Returns:
  //     mean (1-D Array): shape (n_x,);
  return Xsig_pred * weights_;
}

MatrixXd UKF::PredictCovariance(const MatrixXd &Xsig_pred,
                                const VectorXd &mean) {
  // Xsig_pred (2-D Array): shape (n_x, 2 * n_aug + 1)
  // Returns:
  //     P_x (2-D Array): shape (n_x, 2 * n_aug + 1);
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

bool UKF::is_unstable() {
  if (!std::isfinite(x_.sum())) {
    return true;
  }
  if (!std::isfinite(P_.sum())) {
    return true;
  }
  return false;
}
