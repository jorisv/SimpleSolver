#pragma once

// includes
#include <iostream>
// Eigen
#include <Eigen/Core>
#include <Eigen/Cholesky>

// EQP
#include "eq_qp_ns.h"
#include "macros.h"


namespace Eigen
{

namespace qp
{


template <typename MatrixType>
class QpNullSpace
{
public:
  typedef typename MatrixType::Index Index;
  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, Dynamic, 1> XVectorType;

public:
  QpNullSpace();
  QpNullSpace(Index n, Index mEq, Index mIneq);

  template <typename Rhs1, typename Rhs2, typename Rhs3>
  void solve(const MatrixType& G, const MatrixBase<Rhs1>& c,
    const MatrixType& Aeq, const MatrixBase<Rhs2>& beq,
    const MatrixType& Aineq, const MatrixBase<Rhs3>& bineq,
    const XVectorType& x0, std::vector<Index> w0);

  const XVectorType& x() const
  {
    return x_;
  }

private:
  void buildWNeg(Index mInEq, const std::vector<Index>& w);

  Index buildAw(Index mEq, const MatrixType& Aineq,
    const std::vector<Index>& w);

  template <typename Rhs1>
  void buildgw(const MatrixType& G, const MatrixBase<Rhs1>& c,
    const XVectorType& x);

  void addToW(std::size_t index);
  void removeToW(std::size_t index);

public:
  /// workspace and negative of the workspace
  /// Absolute index in Aineq matrix
  std::vector<Index> w_, wNeg_;

  EqQpNullSpace<MatrixType> eqQpNs_;
  LDLT<MatrixType> ldltQ_;
  /// Aw_ after Aeq.rows() is in w_ order
  Matrix<Scalar, Dynamic, Dynamic> Aw_;
  Matrix<Scalar, Dynamic, 1> gw_;
  XVectorType x_, p_;
};



template <typename MatrixType>
inline QpNullSpace<MatrixType>::QpNullSpace()
{}


template <typename MatrixType>
inline QpNullSpace<MatrixType>::QpNullSpace(Index n, Index mEq, Index mIneq)
  : Aw_{mEq + mIneq, n}
  , gw_{n}
  , x_{n}
  , p_{n}
{}


template <typename MatrixType>
template <typename Rhs1, typename Rhs2, typename Rhs3>
inline void QpNullSpace<MatrixType>::solve(
  const MatrixType& G, const MatrixBase<Rhs1>& c,
  const MatrixType& Aeq, const MatrixBase<Rhs2>& beq,
  const MatrixType& Aineq, const MatrixBase<Rhs3>& bineq,
  const XVectorType& x0, std::vector<QpNullSpace<MatrixType>::Index> w0)
{
  static_cast<void>(beq);

  const Index n = G.rows();
  const Index mEq = Aeq.rows();
  const Index mInEq = Aineq.rows();

  w_ = std::move(w0);
  std::sort(w_.begin(), w_.end());
  buildWNeg(mInEq, w_);

  Aw_.resize(mEq + mInEq, n);
  gw_.resize(n);
  p_.resize(n);

  Aw_.topRows(mEq) = Aeq;
  x_ = x0;

  for(int iter = 0; iter < 10; ++iter)
  {
    Index m = buildAw(mEq, Aineq, w_);
    buildgw(G, c, x_);
    bool isConsrained = m != 0;

    if(isConsrained)
    {
      eqQpNs_ = EqQpNullSpace<MatrixType>{n, m};
      eqQpNs_.compute(G, Aw_.topRows(m));
      /// @todo don't solve the full problem each time
      eqQpNs_.solve(gw_, VectorXd::Zero(m), false);
      p_ = eqQpNs_.x();
    }
    else
    {
      ldltQ_.compute(G);
      p_ = -ldltQ_.solve(gw_);
    }

    /// @todo better zero check
    std::cout << "\niter: " << iter << "\n";
    std::cout << "x: " << x_.transpose() << std::endl;
    std::cout << "p: " << p_.transpose() << std::endl;
    std::cout << "Aw: \n";
    std::cout << Aw_.topRows(m) << std::endl;
    std::cout << "gw: " << gw_.transpose() << std::endl;
    if(p_.isZero(1e-8))
    {
      if(isConsrained)
      {
        eqQpNs_.solveLambda(gw_);
        std::cout << "lambda: " << eqQpNs_.lambda().tail(m - mEq).transpose() << "\n";

        if((eqQpNs_.lambda().tail(m - mEq).array() > 0.).all())
        {
          return;
        }
        else
        {
          Index blockingInW;
          eqQpNs_.lambda().tail(m - mEq).minCoeff(&blockingInW);
          removeToW(std::size_t(blockingInW));
        }
      }
      else
      {
        return;
      }
    }
    else
    {
      Scalar alpha = 1.;
      int newConstrInWNeg = -1;
      for(std::size_t i = 0; i < wNeg_.size(); ++i)
      {
        Index AineqIndex = wNeg_[i];
        double aTp = Aineq.row(AineqIndex).dot(p_);
        if(aTp < 0.)
        {
          double aTx = Aineq.row(AineqIndex).dot(x_);
          double alphaCandidate = (bineq[AineqIndex] - aTx)/aTp;
          if(alphaCandidate < alpha)
          {
            alpha = alphaCandidate;
            newConstrInWNeg = i;
          }
        }
      }
      std::cout << "alpha: " << alpha << "\n";

      x_ += alpha*p_;
      if(newConstrInWNeg != -1)
      {
        addToW(newConstrInWNeg);
      }
    }
  }
}


template <typename MatrixType>
void QpNullSpace<MatrixType>::buildWNeg(Index mInEq, const std::vector<Index>& w)
{
  std::vector<Index> wFull(mInEq);
  for(Index i = 0; i < mInEq; ++i)
  {
    wFull[i] = i;
  }

  wNeg_.resize(mInEq - w.size());
  std::set_difference(wFull.begin(), wFull.end(), w.begin(), w.end(),
    wNeg_.begin());
}


template <typename MatrixType>
inline typename QpNullSpace<MatrixType>::Index
QpNullSpace<MatrixType>::buildAw(Index mEq, const MatrixType& Aineq,
  const std::vector<QpNullSpace<MatrixType>::Index>& w)
{
  for(std::size_t i = 0; i < w.size(); ++i)
  {
    Aw_.row(mEq + i) = Aineq.row(w[i]);
  }

  return mEq + Index(w.size());
}


template <typename MatrixType>
template <typename Rhs1>
inline void QpNullSpace<MatrixType>::buildgw(const MatrixType& G,
  const MatrixBase<Rhs1>& c, const XVectorType& x)
{
  gw_.noalias() = c;
  gw_.noalias() += G*x;
}


template <typename MatrixType>
void QpNullSpace<MatrixType>::addToW(std::size_t indexInWNeg)
{
  std::cout << "add: " << wNeg_[indexInWNeg] << "\n";
  w_.push_back(wNeg_[indexInWNeg]);
  wNeg_.erase(wNeg_.begin() + indexInWNeg);
}


template <typename MatrixType>
void QpNullSpace<MatrixType>::removeToW(std::size_t indexInW)
{
  std::cout << "remove: " << w_[indexInW] << "\n";
  wNeg_.push_back(w_[indexInW]);
  w_.erase(w_.begin() + indexInW);
}


}

}
