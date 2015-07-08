#pragma once

// includes
// Eigen
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/QR>

namespace Eigen
{

namespace qp
{

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
  void solve(const MatrixBase<Rhs1>& c, const MatrixBase<Rhs2>& b);

  const XVectorType& x() const
  {
    return x_;
  }

  const LVectorType& lambda() const
  {
    return l_;
  }


private:
  ColPivHouseholderQR<MatrixType> qrAT_, qrAY_;
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



template <typename MatrixType>
EqQpNullSpace<MatrixType>::EqQpNullSpace()
{}


template <typename MatrixType>
EqQpNullSpace<MatrixType>::EqQpNullSpace(Index n, Index m)
  : qrAT_{n, m}
  , qrAY_{m, m}
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
  eigen_assert(n > m);
}


template <typename MatrixType>
void EqQpNullSpace<MatrixType>::compute(const MatrixType& G,
  const MatrixType& A)
{
#if defined(EIGEN_RUNTIME_NO_MALLOC)
  internal::set_is_malloc_allowed(false);
#endif

  const Index m = Y_.cols();
  const Index n = Y_.rows();
  eigen_assert(G.rows() == n && G.cols() == n);
  eigen_assert(A.rows() == m && A.cols() == n);

  // compute the Q matrix and extract Y and Z
  AT_ = A.transpose();
  qrAT_.compute(AT_);
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
  AY_.noalias() = A*Y_;

  lltZTGZ_.compute(ZTGZ_);
  qrAY_.compute(AY_);

#if defined(EIGEN_RUNTIME_NO_MALLOC)
  internal::set_is_malloc_allowed(true);
#endif
}


template <typename MatrixType>
template <typename Rhs1, typename Rhs2>
void EqQpNullSpace<MatrixType>::solve(const MatrixBase<Rhs1>& c,
  const MatrixBase<Rhs2>& b)
{
  eigen_assert(c.rows() == Y_.rows());
  eigen_assert(b.rows() == Y_.cols());

  // solve A*Y*x_y = b
  // allocation in the qr solve method :(
  xy_ = qrAY_.solve(b);

#if defined(EIGEN_RUNTIME_NO_MALLOC)
  internal::set_is_malloc_allowed(false);
#endif

  // solve Z^T*G*Z*x_z = -Z^T*G*Y*X_y - Z^T*c
  // use xz as a buffer
  xz_.noalias() = -ZTGY_*xy_;
  xz_.noalias() -= Z_.transpose()*c;
  xz_ = lltZTGZ_.solve(xz_);

  // compute x = Y*x_y + Z*x_z
  x_.noalias() = Y_*xy_;
  x_.noalias() += Z_*xz_;

  // use xy as a buffer
  xy_.noalias() = Y_.transpose()*c;
  xy_.noalias() += YTG_*x_;

#if defined(EIGEN_RUNTIME_NO_MALLOC)
  internal::set_is_malloc_allowed(true);
#endif

  // solve (A*Y)^T = Y^T*(c + G*x)
  // allocation because of the transpose but even
  // inverse is calling solve that will make an allocation too
  l_.noalias() = qrAY_.inverse().transpose()*xy_;
}


}

}
