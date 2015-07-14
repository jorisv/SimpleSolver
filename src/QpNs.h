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

// SimpleSolver
#include "EqQpNs.h"
#include "Macros.h"


namespace Eigen
{

namespace qp
{

enum struct LoggerType { Dummy, Full };


template <LoggerType Type>
struct QPLogger
{
  enum IterType { RemoveW, AddW, KeepW };

  void newIteration(int iterate);
  void setIterType(IterType iType);
  void setWIter(int iterW);
  void setWSet(const std::vector<VectorXd::Index>& w);
  void setX(const VectorXd& x);
  void setP(const VectorXd& p);
  void setLambda(const VectorXd& lambda);
  void clear();
};


template <>
struct QPLogger<LoggerType::Dummy>
{
  enum IterType { RemoveW, AddW, KeepW };

  void newIteration(int iterate) {static_cast<void>(iterate);}
  void setIterType(IterType iType) {static_cast<void>(iType);}
  void setWIter(int iterW) {static_cast<void>(iterW);}
  void setWSet(const std::vector<VectorXd::Index>& w) {static_cast<void>(w);}
  void setX(const VectorXd& x) {static_cast<void>(x);}
  void setP(const VectorXd& p) {static_cast<void>(p);}
  void setLambda(const VectorXd& lambda) {static_cast<void>(lambda);}
  void clear() {}
};


template <>
struct QPLogger<LoggerType::Full>
{
  enum IterType { RemoveW, AddW, KeepW };

  struct Data
  {
    Data(int iter) : iterate{iter}
    {}

    int iterate;
    IterType iterType;
    int iterW;
    std::vector<VectorXd::Index> wSet;
    Eigen::VectorXd x, p, lambda;
  };

  void newIteration(int iterate)
  { datas.emplace_back(iterate); }
  void setIterType(IterType iType)
  { datas.back().iterType = iType; }
  void setWIter(int iterW)
  { datas.back().iterW = iterW; }
  void setWSet(const std::vector<VectorXd::Index>& w)
  { datas.back().wSet = w; }
  void setX(const VectorXd& x)
  { datas.back().x = x; }
  void setP(const VectorXd& p)
  { datas.back().p = p; }
  void setLambda(const VectorXd& lambda)
  { datas.back().lambda = lambda; }
  void clear()
  { datas.clear(); }

  std::vector<Data> datas;
};




template <typename MatrixType, LoggerType LType=LoggerType::Dummy>
class QpNullSpace
{
public:
  enum {
    RowsAtCompileTime = MatrixType::RowsAtCompileTime,
    ColsAtCompileTime = MatrixType::ColsAtCompileTime,
    Options = MatrixType::Options,
    MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime,
  };

  typedef typename MatrixType::Index Index;
  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, Dynamic, 1, Options> XVectorType;

  typedef Matrix<Scalar, Dynamic, Dynamic, Options> TmpMatrixType;
  typedef Matrix<Scalar, Dynamic, 1, Options> TmpVectorType;

  typedef QPLogger<LType> Logger;

public:
  QpNullSpace();
  QpNullSpace(Index n, Index mEq, Index mIneq);

  template <typename Rhs1, typename Rhs2, typename Rhs3>
  void solve(const MatrixType& G, const MatrixBase<Rhs1>& c,
    const MatrixType& Aeq, const MatrixBase<Rhs2>& beq,
    const MatrixType& Aineq, const MatrixBase<Rhs3>& bineq,
    const XVectorType& x0, std::vector<Index> w0,
    int maxIter=NumTraits<int>::highest());

  const XVectorType& x() const
  {
    return x_;
  }

  const Logger& logger() const
  {
    return logger_;
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
  TmpMatrixType Aw_;
  TmpVectorType gw_;
  XVectorType x_, p_;

  Logger logger_;
};



template <typename MatrixType, LoggerType LType>
inline QpNullSpace<MatrixType, LType>::QpNullSpace()
{}


template <typename MatrixType, LoggerType LType>
inline QpNullSpace<MatrixType, LType>::QpNullSpace(Index n, Index mEq, Index mIneq)
  : Aw_{mEq + mIneq, n}
  , gw_{n}
  , x_{n}
  , p_{n}
{}


template <typename MatrixType, LoggerType LType>
template <typename Rhs1, typename Rhs2, typename Rhs3>
inline void QpNullSpace<MatrixType, LType>::solve(
  const MatrixType& G, const MatrixBase<Rhs1>& c,
  const MatrixType& Aeq, const MatrixBase<Rhs2>& beq,
  const MatrixType& Aineq, const MatrixBase<Rhs3>& bineq,
  const XVectorType& x0, std::vector<QpNullSpace<MatrixType, LType>::Index> w0,
  int maxIter)
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

  bool wHasChanged = true;
  bool isLDLTComputed = false;
  for(int iter = 0; iter < maxIter; ++iter)
  {
    logger_.newIteration(iter);
    logger_.setX(x_);
    logger_.setWSet(w_);
    logger_.setIterType(QPLogger<LType>::KeepW);

    Index m = buildAw(mEq, Aineq, w_);
    buildgw(G, c, x_);
    bool isConsrained = m != 0;

    // solve a quadratic problem to move along all active constraints
    // ( p . a_i = 0) while minimizing the cost function.
    if(isConsrained)
    {
      // only call compute if the w set has changed
      if(wHasChanged)
      {
        /// @todo make a rank update only
        eqQpNs_.compute(G, Aw_.topRows(m));
      }
      eqQpNs_.solve(gw_, VectorXd::Zero(m), false);
      p_ = eqQpNs_.x();
    }
    else
    {
      // only compute he LDLT of Q once
      if(!isLDLTComputed)
      {
        ldltQ_.compute(G);
        isLDLTComputed = true;
      }
      p_ = -ldltQ_.solve(gw_);
    }

    logger_.setP(p_);

    wHasChanged = false;
    /// @todo verify zero check
    /// We check zero against the machine precision multiply by the diagonal
    /// of the K matrix
    if(p_.isZero(NumTraits<Scalar>::epsilon()*Scalar(2*n - m)))
    {

      // p is a null vector so we have two choice.
      // 1) All lambda are positive, so all KKT condition are fulfill and
      // x is a global optimum point.
      // 2) one or more lambda are negative, so the point is not a global
      // optimum. We add the most negative lambda constraint to the w set.
      if(isConsrained)
      {
        eqQpNs_.solveLambda(gw_);
        logger_.setLambda(eqQpNs_.lambda());

        if((eqQpNs_.lambda().tail(m - mEq).array() > Scalar(0.)).all())
        {
          return;
        }
        else
        {
          Index blockingInW;
          eqQpNs_.lambda().tail(m - mEq).minCoeff(&blockingInW);
          logger_.setWIter(int(w_[blockingInW]));
          logger_.setIterType(QPLogger<LType>::RemoveW);
          removeToW(std::size_t(blockingInW));

          wHasChanged = true;
        }
      }
      else
      {
        return;
      }
    }
    else
    {
      // p is not null so it indicate the descent direction.
      // We looking to know of much we must move on this direction (alpha)
      // by examining the descent direction and constraints that are not in
      // the w set.
      // If the dot product a_i . p_k is negative then the descent direction
      // can violate the constraint for some value of alpha.
      // We can compute the alpha that will make the constraint i active with
      // (b_i - a_i . x_k)/(a_i . p_k).
      // By taking the minimal value of alpha we ensure that no inequality
      // constraint will be violated.
      // We then add this constraint to the w set.

      Scalar alpha(1.);
      int newConstrInWNeg = -1;
      for(std::size_t i = 0; i < wNeg_.size(); ++i)
      {
        Index AineqIndex = wNeg_[i];
        double aTp = Aineq.row(AineqIndex).dot(p_);
        if(aTp < Scalar(0.))
        {
          double aTx = Aineq.row(AineqIndex).dot(x_);
          double alphaCandidate = (bineq[AineqIndex] - aTx)/aTp;
          if(alphaCandidate < alpha)
          {
            alpha = alphaCandidate;
            newConstrInWNeg = int(i);
          }
        }
      }

      // move along p
      x_ += alpha*p_;
      if(newConstrInWNeg != -1)
      {
        logger_.setWIter(int(wNeg_[newConstrInWNeg]));
        logger_.setIterType(QPLogger<LType>::AddW);

        addToW(newConstrInWNeg);
        wHasChanged = true;
      }
    }
  }
}


template <typename MatrixType, LoggerType LType>
void QpNullSpace<MatrixType, LType>::buildWNeg(Index mInEq, const std::vector<Index>& w)
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


template <typename MatrixType, LoggerType LType>
inline typename QpNullSpace<MatrixType, LType>::Index
QpNullSpace<MatrixType, LType>::buildAw(Index mEq, const MatrixType& Aineq,
  const std::vector<QpNullSpace<MatrixType, LType>::Index>& w)
{
  for(std::size_t i = 0; i < w.size(); ++i)
  {
    Aw_.row(mEq + i) = Aineq.row(w[i]);
  }

  return mEq + Index(w.size());
}


template <typename MatrixType, LoggerType LType>
template <typename Rhs1>
inline void QpNullSpace<MatrixType, LType>::buildgw(const MatrixType& G,
  const MatrixBase<Rhs1>& c, const XVectorType& x)
{
  gw_.noalias() = c;
  gw_.noalias() += G*x;
}


template <typename MatrixType, LoggerType LType>
inline void QpNullSpace<MatrixType, LType>::addToW(std::size_t indexInWNeg)
{
  w_.push_back(wNeg_[indexInWNeg]);
  wNeg_.erase(wNeg_.begin() + indexInWNeg);
}


template <typename MatrixType, LoggerType LType>
inline void QpNullSpace<MatrixType, LType>::removeToW(std::size_t indexInW)
{
  wNeg_.push_back(w_[indexInW]);
  w_.erase(w_.begin() + indexInW);
}


}

}
