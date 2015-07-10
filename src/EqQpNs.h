// This file is part of SimpleSolver.
//
// SimpleSolver is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// SimpleSolver is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with SimpleSolver.  If not, see <http://www.gnu.org/licenses/>.

#pragma once

// includes
// Eigen
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/QR>

// SimpleSolver
#include "Macros.h"


namespace Eigen
{

namespace qp
{


double eqQpLagrangian(const MatrixXd& G, const VectorXd& c,
  const MatrixXd& A, const VectorXd& b,
  const VectorXd& x, const VectorXd& lambda);


VectorXd eqQpLagrangianGrad(const MatrixXd& G, const VectorXd& c,
  const MatrixXd& A,
  const VectorXd& x, const VectorXd& lambda);


/**
  * Use the nullspace method described in [Nocedal, Wright] p 457
  * to solve an equality constraint only QP with m consraint and n variables.
  */
template <typename MatrixType>
class EqQpNullSpace
{
public:
  typedef typename MatrixType::Index Index;
  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, Dynamic, 1> XVectorType;
  typedef Matrix<Scalar, Dynamic, 1> LVectorType;

public:
  EqQpNullSpace();
  EqQpNullSpace(Index n, Index m);

  void compute(const MatrixType& G, const MatrixType& A);

  template <typename Rhs1, typename Rhs2>
  void solve(const MatrixBase<Rhs1>& c, const MatrixBase<Rhs2>& b,
    bool computeLambda=true);

  template <typename Rhs1>
  void solveLambda(const MatrixBase<Rhs1>& c);

  const XVectorType& x() const
  {
    return x_;
  }

  const LVectorType& lambda() const
  {
    return l_;
  }


private:
  ColPivHouseholderQR<MatrixType> qrAT_;
  LLT<MatrixType> lltZTGZ_;
  // preallocation for the computation of Q
  Matrix<Scalar, Dynamic, Dynamic> Q_, AT_;
  Matrix<Scalar, Dynamic, 1> Qw_;
  // Store the G and A matrix and buffer to avoid allocations
  Matrix<Scalar, Dynamic, Dynamic> ZTG_, ZTGY_, ZTGZ_, YTG_, AY_;
  Matrix<Scalar, Dynamic, Dynamic> Y_, Z_;
  XVectorType xy_, xz_, x_;
  LVectorType l_; // lambda
};



inline double eqQpLagrangian(const MatrixXd& G, const VectorXd& c,
  const MatrixXd& A, const VectorXd& b,
  const VectorXd& x, const VectorXd& lambda)
{
  return 0.5*x.transpose()*G*x + x.dot(c) - lambda.dot(A*x - b);
}


inline VectorXd eqQpLagrangianGrad(const MatrixXd& G, const VectorXd& c,
  const MatrixXd& A,
  const VectorXd& x, const VectorXd& lambda)
{
  return G*x + c -
    (A.array().colwise()*lambda.array()).matrix().colwise().sum().transpose();
}


template <typename MatrixType>
inline EqQpNullSpace<MatrixType>::EqQpNullSpace()
{}


template <typename MatrixType>
inline EqQpNullSpace<MatrixType>::EqQpNullSpace(Index n, Index m)
  : qrAT_{n, m}
  , lltZTGZ_{n-m}
  , Q_{n, n}
  , AT_{n, m}
  , Qw_{n}
  , ZTG_{n-m, n}
  , ZTGY_{n-m, m}
  , ZTGZ_{n-m, n-m}
  , YTG_{m, n}
  , AY_{m, m}
  , Y_{n, m}
  , Z_{n, n-m}
  , xy_{m}
  , xz_{n-m}
  , x_{n}
  , l_{m}
{
  eigen_assert(n >= m);
}


template <typename MatrixType>
inline void EqQpNullSpace<MatrixType>::compute(const MatrixType& G,
  const MatrixType& A)
{
  SS_CHECK_MALLOC(true);

  const Index m = Y_.cols();
  const Index n = Y_.rows();
  eigen_assert(G.rows() == n && G.cols() == n);
  eigen_assert(A.rows() == m && A.cols() == n);

  // compute the Q matrix and extract Y and Z
  AT_ = A.transpose();
  qrAT_.compute(AT_);
  // avoid one allocation by using evalTo with already allocated workspace
  qrAT_.householderQ().evalTo(Q_, Qw_);
  Y_.noalias() = Q_.topLeftCorner(n, m);
  Z_.noalias() = Q_.topRightCorner(n, n - m);

  // compute YTG and ZTG to avoid to have the
  // G matrix give in the next step
  // Compute AY, ZTGZ and ZTGY to avoid allocations
  ZTG_.noalias() = Z_.transpose()*G;
  ZTGY_.noalias() = ZTG_*Y_;
  ZTGZ_.noalias() = ZTG_*Z_;
  YTG_.noalias() = Y_.transpose()*G;
  /// @todo figure out if is better to use AY = P*R^T that seem to be
  /// thresholded
  AY_.noalias() = A*Y_;

  lltZTGZ_.compute(ZTGZ_);

  SS_CHECK_MALLOC(false);
}


template <typename MatrixType>
template <typename Rhs1, typename Rhs2>
inline void EqQpNullSpace<MatrixType>::solve(const MatrixBase<Rhs1>& c,
  const MatrixBase<Rhs2>& b, bool computeLambda)
{
  eigen_assert(c.rows() == Y_.rows());
  eigen_assert(b.rows() == Y_.cols());

  SS_CHECK_MALLOC(true);

  // solve A*Y*x_y = b
  // since A*Y = P*R^{T}
  // with P a permutation matrix and R an upper triangular matrix
  // P*R^{T}*x_y = b
  // R^{T}*x_y = P*b
  xy_.noalias() = qrAT_.colsPermutation()*b;
  qrAT_.matrixR().topRows(Y_.cols()).template
    triangularView<Upper>().transpose().solveInPlace(xy_);

  // solve Z^T*G*Z*x_z = -Z^T*G*Y*X_y - Z^T*c
  // use xz as a buffer
  xz_.noalias() = -ZTGY_*xy_;
  xz_.noalias() -= Z_.transpose()*c;
  xz_ = lltZTGZ_.solve(xz_);

  // compute x = Y*x_y + Z*x_z
  x_.noalias() = Y_*xy_;
  x_.noalias() += Z_*xz_;

  SS_CHECK_MALLOC(false);

  if(computeLambda)
  {
    solveLambda(c);
  }
}


template <typename MatrixType>
template <typename Rhs1>
inline void EqQpNullSpace<MatrixType>::solveLambda(const MatrixBase<Rhs1>& c)
{
  eigen_assert(c.rows() == Y_.rows());

  SS_CHECK_MALLOC(true);

  // use xy_ as a buffer
  xy_.noalias() = Y_.transpose()*c;
  xy_.noalias() += YTG_*x_;

  // solve (A*Y)^T*lambda = Y^T*(c + G*x)
  // see xy_ computation for some explanations
  // (P*R^{T})^T*lambda = Y^T*(c + G*x)
  // R*P*lambda = Y^T(c + G*x)
  // P*lambda = R^{-1}*Y^T(c + G*x)
  // lambda = P*R^{T^{-1}}*Y^T(c + G*x)
  qrAT_.matrixR().topRows(Y_.cols()).template
    triangularView<Upper>().solveInPlace(xy_);
  l_.noalias() = qrAT_.colsPermutation()*xy_;

  SS_CHECK_MALLOC(false);
}


}

}
