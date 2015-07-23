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
#include <Eigen/LU>

// SimpleSolver
#include "Macros.h"


namespace Eigen
{

namespace lp
{

enum struct LoggerType { Dummy, Full };


template <LoggerType Type>
struct LPLogger
{
  void newIteration(int iterate);
  void setEnteringW(int w);
  void setLeavingW(int w);
  void setWSet(const std::vector<VectorXd::Index>& w);
  void setX(const VectorXd& x);
  void setD(const VectorXd& d);
  void setLambda(const VectorXd& lambda);
  void clear();
};


template <>
struct LPLogger<LoggerType::Dummy>
{
  void newIteration(int iterate) {static_cast<void>(iterate);}
  void setEnteringW(int w) {static_cast<void>(w);}
  void setLeavingW(int w) {static_cast<void>(w);}
  void setWSet(const std::vector<VectorXd::Index>& w) {static_cast<void>(w);}
  void setX(const VectorXd& x) {static_cast<void>(x);}
  void setD(const VectorXd& d) {static_cast<void>(d);}
  void setLambda(const VectorXd& lambda) {static_cast<void>(lambda);}
  void clear() {}
};


template <>
struct LPLogger<LoggerType::Full>
{
  struct Data
  {
    Data(int iter) : iterate{iter}
    {}

    int iterate;
    int eW, lW;
    std::vector<VectorXd::Index> wSet;
    Eigen::VectorXd x, d, lambda;
  };

  void newIteration(int iterate)
  { datas.emplace_back(iterate); }
  void setEnteringW(int w)
  { datas.back().eW = w; }
  void setLeavingW(int w)
  { datas.back().lW = w; }
  void setWSet(const std::vector<VectorXd::Index>& w)
  { datas.back().wSet = w; }
  void setX(const VectorXd& x)
  { datas.back().x = x; }
  void setD(const VectorXd& d)
  { datas.back().d = d; }
  void setLambda(const VectorXd& lambda)
  { datas.back().lambda = lambda; }
  void clear()
  { datas.clear(); }

  std::vector<Data> datas;
};




template <typename MatrixType, LoggerType LType=LoggerType::Dummy>
class LpPrimal
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

  typedef LPLogger<LType> Logger;

  enum struct Exit { Success, Unbounded, MaxIter };

public:
  LpPrimal();
  LpPrimal(Index n, Index mEq, Index mIneq);

  template <typename Rhs1, typename Rhs2, typename Rhs3>
  Exit solve(const MatrixBase<Rhs1>& c,
    const MatrixType& Aeq, const MatrixBase<Rhs2>& beq,
    const MatrixType& Aineq, const MatrixBase<Rhs3>& bineq,
    const std::vector<Index>& w0,
    int maxIter=NumTraits<int>::highest());

  const XVectorType& x() const
  {
    return x_;
  }

  const std::vector<Index>& w() const
  {
    return w_;
  }

  const Logger& logger() const
  {
    return logger_;
  }

private:
  void buildWNeg(Index mInEq, const std::vector<Index>& w);

  template <typename Rhs1>
  void buildAwbw(Index mEq,
    const MatrixType& Aineq, const MatrixBase<Rhs1>& bineq,
    const std::vector<Index>& w);

  void addToW(std::size_t index);
  void removeToW(std::size_t index);

public:
  /// workspace and negative of the workspace
  /// Absolute index in Aineq matrix
  std::vector<Index> w_, wNeg_;

  PartialPivLU<MatrixType> luAw_;
  TmpMatrixType Aw_;
  TmpVectorType bw_, lambda_;
  XVectorType x_, d_;

  Logger logger_;
};



template <typename MatrixType, LoggerType LType>
inline LpPrimal<MatrixType, LType>::LpPrimal()
{}


template <typename MatrixType, LoggerType LType>
inline LpPrimal<MatrixType, LType>::LpPrimal(Index n, Index mEq, Index mIneq)
  : luAw_{n}
  , Aw_{n, n}
  , bw_{n}
  , lambda_{n}
  , x_{n}
  , d_{n}
{
  static_cast<void>(mEq);
  static_cast<void>(mIneq);
}


template <typename MatrixType, LoggerType LType>
template <typename Rhs1, typename Rhs2, typename Rhs3>
inline typename LpPrimal<MatrixType, LType>::Exit
LpPrimal<MatrixType, LType>::solve(
  const MatrixBase<Rhs1>& c,
  const MatrixType& Aeq, const MatrixBase<Rhs2>& beq,
  const MatrixType& Aineq, const MatrixBase<Rhs3>& bineq,
  const std::vector<LpPrimal<MatrixType, LType>::Index>& w0,
  int maxIter)
{
  const Index n = c.rows();
  const Index mEq = Aeq.rows();
  const Index mInEq = Aineq.rows();

  eigen_assert(mEq + mInEq >= n);
  eigen_assert(int(w0.size()) == (n - mEq));

  logger_.clear();

  w_ = w0;
  std::sort(w_.begin(), w_.end());
  buildWNeg(mInEq, w_);

  Aw_.resize(n, n);
  bw_.resize(n);
  lambda_.resize(n);
  d_.resize(n);

  Aw_.topRows(mEq) = Aeq;
  bw_.topRows(mEq) = beq;

  buildAwbw(mEq, Aineq, bineq, w_);
  luAw_.compute(Aw_);
  x_ = luAw_.solve(bw_);

  for(int iter = 0; iter < maxIter; ++iter)
  {
    logger_.newIteration(iter);
    logger_.setX(x_);
    logger_.setWSet(w_);

    // Terminate if All lambda are positive, so all KKT condition are fulfill
    // and x is a global optimum point.
    // Aw = PLU with P a permutation matrix, L an unit-lower triangular matrix
    // and U an upper triangular matrix
    // Aw^{T} = U^{T} L^{T} P^{T}
    // Aw^{T} lambda = c
    // lambda = Aw^{T^{-1}} c
    // lambda = (U^{T}·L^{T}·P^{T})^{-1} c
    // lambda = (P L^{T^{-1}} U^{T^{-1}}) c
    lambda_ = luAw_.matrixLU().template triangularView<Upper>().transpose().solve(c);
    luAw_.matrixLU().template triangularView<UnitLower>().transpose().solveInPlace(lambda_);
    lambda_ = luAw_.permutationP()*lambda_;
    logger_.setLambda(lambda_);

    if((lambda_.tail(n - mEq).array() >= Scalar(0.)).all())
    {
      return Exit::Success;
    }

    // one or more lambda are negative, so the point is not a global
    // optimum. We remove the most negative lambda constraint to the w set.
    // @TODO use a different removing strategy to avoid cycling
    Index blockingInW, leavingIndex;
    lambda_.tail(n - mEq).minCoeff(&blockingInW);
    leavingIndex = w_[blockingInW];
    logger_.setLeavingW(int(leavingIndex));

    // compute the descent direction d that must move in the active constraint
    // direction and in the leaving constraint normal direction
    // a_i^{T} d = 0, with i in w\leavingIndex
    // a_i^{T} d = 1, with i = leavingIndex
    // So it's the leavingIndex column of the Aw^{-1} matrix
    // d_ = luAw_.inverse().col(mEq + blockingInW);
    // @TODO is that better than using directly the inverse matrix ?
    d_ = luAw_.solve(VectorXd::Unit(n, mEq + blockingInW));

    logger_.setD(d_);

    // We looking to know of much we must move on d (alpha)
    // by examining the descent direction and constraints that are not in
    // the w set.
    // If the dot product a_i . p_k is negative then the descent direction
    // can violate the constraint for some value of alpha.
    // We can compute the alpha that will make the constraint i active with
    // (b_i - a_i . x_k)/(a_i . p_k).
    // By taking the minimal value of alpha we ensure that no inequality
    // constraint will be violated.
    // We then add this constraint to the w set.

    Scalar alpha(std::numeric_limits<Scalar>::infinity());
    int newConstrInWNeg = -1;
    for(std::size_t i = 0; i < wNeg_.size(); ++i)
    {
      Index AineqIndex = wNeg_[i];
      double aTd = Aineq.row(AineqIndex).dot(d_);
      if(aTd < Scalar(0.))
      {
        double aTx = Aineq.row(AineqIndex).dot(x_);
        double alphaCandidate = (bineq[AineqIndex] - aTx)/aTd;
        if(alphaCandidate < alpha)
        {
          alpha = alphaCandidate;
          newConstrInWNeg = int(i);
        }
      }
    }

    // no bounding constraint found, so the problem is unbounded
    if(newConstrInWNeg == -1)
    {
      return Exit::Unbounded;
    }

    Index enteringIndex = wNeg_[newConstrInWNeg];
    logger_.setEnteringW(int(enteringIndex));

    // move along d
    x_ += alpha*d_;

    // compute the new Aw matrix and his LU decomposition
    // @TODO make a rank update on the LU matrix
    Aw_.row(mEq + blockingInW) = Aineq.row(enteringIndex);
    bw_(mEq + blockingInW) = bineq(enteringIndex);
    luAw_.compute(Aw_);

    removeToW(std::size_t(blockingInW));
    addToW(newConstrInWNeg);
  }

  return Exit::MaxIter;
}


template <typename MatrixType, LoggerType LType>
void LpPrimal<MatrixType, LType>::buildWNeg(Index mInEq, const std::vector<Index>& w)
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
template <typename Rhs1>
void LpPrimal<MatrixType, LType>::buildAwbw(Index mEq,
  const MatrixType& Aineq, const MatrixBase<Rhs1>& bineq,
  const std::vector<Index>& w)
{
  for(std::size_t i = 0; i < w.size(); ++i)
  {
    Aw_.row(mEq + i) = Aineq.row(w[i]);
    bw_.row(mEq + i) = bineq.row(w[i]);
  }
}


template <typename MatrixType, LoggerType LType>
inline void LpPrimal<MatrixType, LType>::addToW(std::size_t indexInWNeg)
{
  w_.push_back(wNeg_[indexInWNeg]);
  wNeg_.erase(wNeg_.begin() + indexInWNeg);
}


template <typename MatrixType, LoggerType LType>
inline void LpPrimal<MatrixType, LType>::removeToW(std::size_t indexInW)
{
  wNeg_.push_back(w_[indexInW]);
  w_.erase(w_.begin() + indexInW);
}

}

}
