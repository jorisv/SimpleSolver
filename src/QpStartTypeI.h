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


namespace Eigen
{

namespace simple_solver
{

template <typename QpType>
class QpStartTypeI
{
public:
  enum {
    RowsAtCompileTime = QpType::RowsAtCompileTime,
    ColsAtCompileTime = QpType::ColsAtCompileTime,
    Options = QpType::Options,
    MaxRowsAtCompileTime = QpType::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = QpType::MaxColsAtCompileTime,
  };

  typedef typename QpType::Index Index;
  typedef typename QpType::Scalar Scalar;
  typedef Matrix<Scalar, Dynamic, Dynamic, Options> MatrixType;
  typedef Matrix<Scalar, Dynamic, 1, Options> VectorType;

  typedef LpPrimal<MatrixType> SolverType;

  enum struct Exit {
    Success=SolverType::Exit::Success,
    Unbounded=SolverType::Exit::Unbounded,
    MaxIter=SolverType::Exit::MaxIter,
    Infeasible
  };

public:
  QpStartTypeI();
  QpStartTypeI(Index n, Index mEq, Index mIneq);

  template <typename Rhs1, typename Rhs2>
  Exit findInit(const MatrixType& Aeq, const MatrixBase<Rhs1>& beq,
    const MatrixType& Aineq, const MatrixBase<Rhs2>& bineq,
    const VectorType& x0, Scalar precision,
    int maxIter=NumTraits<int>::highest());

  const VectorType& x() const
  {
    return x_;
  }

  const std::vector<Index>& w() const
  {
    return w_;
  }

private:
  SolverType solver_;

  std::vector<Index> w_, violEq_;
  MatrixType Aeq_, Aineq_;
  VectorType c_, bineq_, x_;
  VectorType tmp_;
};



template <typename QpType>
inline QpStartTypeI<QpType>::QpStartTypeI()
{}


template <typename QpType>
inline QpStartTypeI<QpType>::QpStartTypeI(Index n, Index mEq, Index mIneq)
  : solver_{n + mEq + mIneq, mEq, mEq + 2*mIneq}
  , Aeq_{mEq, n + mEq + mIneq}
  , Aineq_{mEq + 2*mIneq, n + mEq + mIneq}
  , c_{n + mEq + mIneq}
  , bineq_{mEq + 2*mIneq}
  , x_{n + mEq + mIneq}
  , tmp_{mEq + mIneq}
{}


template <typename QpType>
template <typename Rhs1, typename Rhs2>
typename QpStartTypeI<QpType>::Exit QpStartTypeI<QpType>::findInit(
  const QpStartTypeI<QpType>::MatrixType& Aeq, const MatrixBase<Rhs1>& beq,
  const QpStartTypeI<QpType>::MatrixType& Aineq, const MatrixBase<Rhs2>& bineq,
  const QpStartTypeI<QpType>::VectorType& x0,
  Scalar precision,
  int maxIter)
{
  eigen_assert(Aeq.cols() == Aineq.cols());
  eigen_assert(Aeq.rows() == beq.rows());
  eigen_assert(Aineq.rows() == bineq.rows());

  const Index n = Aeq.cols();
  const Index mEq = Aeq.rows();
  const Index mIneq = Aineq.rows();

  // clear the buffers
  w_.clear();
  violEq_.clear();

  // compute constraint violation
  tmp_.resize(mEq + mIneq);

  tmp_.head(mEq) = -beq;
  tmp_.head(mEq).noalias() += Aeq*x0;
  tmp_.tail(mIneq) = -bineq;
  tmp_.tail(mIneq).noalias() += Aineq*x0;

  // Identify the violated equality constraints and add
  // one z variable for each equality constraints violated.
  Index zEq = 0;
  for(Index i = 0; i < mEq; ++i)
  {
    if(std::abs(tmp_(i)) > precision)
    {
      zEq += 1;
      violEq_.push_back(i);
    }
  }

  // Identify the violated inequality constraints and add
  // one z variable for each inequality constraints violated.
  // Also add violated inequality constraint to the initial working set.
  // This should prevent adding linearly dependent constraint
  // (except if a_i == a_j)
  /// @todo in some case there no enough violated constraints to cover
  /// the n variables
  Index zIneq = 0;
  for(Index i = mEq; i < mEq + mIneq; ++i)
  {
    if(tmp_(i) < precision)
    {
      zIneq += 1;
      w_.push_back(i - mEq);
    }
  }

  eigen_assert((zIneq + mEq) >= n);

  // if no violation, then the problem is feasible
  if((zIneq + zEq) == 0)
  {
    x_ = x0;
    return Exit::Success;
  }

  Index newN = n + zEq + zIneq;

  // minimize z
  c_.resize(newN);
  c_.head(n).setZero();
  c_.tail(newN - n).setOnes();
  x_.resize(n);

  Aeq_.resize(mEq, newN);
  Aeq_.block(0, 0, mEq, n) = Aeq;
  Aeq_.block(0, n, mEq, zEq + zIneq);
  // set the z violation variable for all violated equality constraints
  for(std::size_t i = 0; i < violEq_.size(); ++i)
  {
    Index row = violEq_[i];
    Aeq_(row, n + i) = tmp_(i) > 0. ? -1. : 1.;
  }

  Aineq_.resize(mIneq + zEq + zIneq, newN);
  bineq_.resize(mIneq + zEq + zIneq);

  Aineq_.block(0, 0, mIneq, n) = Aineq;
  Aineq_.block(0, n, mIneq, zEq + zIneq).setZero();
  // set the z violation variable for all violated inequality constraints
  for(std::size_t i = 0; i < w_.size(); ++i)
  {
    Index row = w_[i];
    Aineq_(row, n + zEq + i) = 1.;
  }
  // set the z >= 0 constraints
  Aineq_.block(mIneq, 0, zEq + zIneq, n).setZero();
  Aineq_.block(mIneq, n, zEq + zIneq, zEq + zIneq).setIdentity();
  bineq_.head(mIneq) = bineq;
  bineq_.tail(zEq + zIneq).setZero();

  // fill w_ with z constraints until w_ reach newN size
  for(Index i = 0; i < (zEq + zIneq) - (zIneq - (n - mEq)); ++i)
  {
    w_.push_back(mIneq + i);
  }

  eigen_assert(Index(w_.size()) == newN);

  Exit status = Exit(solver_.solve(c_, Aeq_, beq, Aineq_, bineq_, w_, maxIter));

  if(status == Exit::Success)
  {
    if(c_.dot(solver_.x()) > precision)
    {
      return Exit::Infeasible;
    }

    x_ = solver_.x().head(n);
    // only keep active constraints in Aineq
    w_.clear();
    for(Index i: solver_.w())
    {
      if(i < mIneq)
      {
        w_.push_back(i);
      }
    }
  }

  return status;
}

}

}
