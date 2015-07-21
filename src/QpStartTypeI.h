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
// SimpleSolver
#include "QpNs.h"

namespace Eigen
{

namespace qp
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

public:
  QpStartTypeI();
  QpStartTypeI(Index n, Index mEq, Index mIneq);

  template <typename Rhs1, typename Rhs2>
  void findInit(QpType& solver,
    const MatrixType& Aeq, const MatrixBase<Rhs1>& beq,
    const MatrixType& Aineq, const MatrixBase<Rhs2>& bineq,
    const VectorType& x0, const std::vector<Index>& w0,
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
  std::vector<Index> w_;
  MatrixType G_, Aeq_, Aineq_;
  VectorType c_, bineq_, x_;
  VectorType tmp_;
};



template <typename QpType>
inline QpStartTypeI<QpType>::QpStartTypeI()
{}


template <typename QpType>
inline QpStartTypeI<QpType>::QpStartTypeI(Index n, Index mEq, Index mIneq)
  : G_{n + mEq + mIneq, n + mEq + mIneq}
  , Aeq_{mEq, n + mEq + mIneq}
  , Aineq_{mEq + 2*mIneq, n + mEq + mIneq}
  , c_{n + mEq + mIneq}
  , bineq_{mEq + 2*mIneq}
  , x_{n + mEq + mIneq}
  , tmp_{std::max(mEq, mIneq)}
{}


template <typename QpType>
template <typename Rhs1, typename Rhs2>
void QpStartTypeI<QpType>::findInit(QpType& solver,
  const QpStartTypeI<QpType>::MatrixType& Aeq, const MatrixBase<Rhs1>& beq,
  const QpStartTypeI<QpType>::MatrixType& Aineq, const MatrixBase<Rhs2>& bineq,
  const QpStartTypeI<QpType>::VectorType& x0,
  const std::vector<QpStartTypeI<QpType>::Index>& w0,
  int maxIter)
{
  eigen_assert(Aeq.cols() == Aineq.cols());
  eigen_assert(Aeq.rows() == beq.rows());
  eigen_assert(Aineq.rows() == bineq.rows());

  Index n = Aeq.cols();
  Index mEq = Aeq.rows();
  Index mIneq = Aineq.rows();
  Index newN = n + mEq + mIneq;

  G_.setZero(newN, newN);
  c_.resize(newN);
  c_.head(n).setZero();
  c_.tail(newN - n).setOnes();

  Aeq_.resize(mEq, newN);
  tmp_.resize(std::max(mEq, mIneq));

  Aeq_.block(0, 0, mEq, n) = Aeq;
  tmp_.head(mEq).noalias() = Aeq*x0;
  tmp_.head(mEq).noalias() -= beq;
  Aeq_.block(0, n, mEq, mEq) =
    ((tmp_.head(mEq).array() > 0.)
     .select(-VectorType::Ones(mEq), VectorType::Ones(mEq))).asDiagonal();
  Aeq_.block(0, n + mEq, mEq, mIneq).setZero();

  Aineq_.resize(mEq + 2*mIneq, newN);
  bineq_.resize(mEq + 2*mIneq);

  Aineq_.block(0, 0, mIneq, n) = Aineq;
  Aineq_.block(0, n, mIneq, mEq).setZero();
  Aineq_.block(0, n + mEq, mIneq, mIneq).setIdentity();
  Aineq_.block(mIneq, 0, mEq + mIneq, n).setZero();
  Aineq_.block(mIneq, n, mEq + mIneq, mEq + mIneq).setIdentity();
  bineq_.head(mIneq) = bineq;
  bineq_.tail(mEq + mIneq).setZero();

  x_.resize(newN);
  x_.head(n) = x0;
  x_.segment(n, mEq) = tmp_.head(mEq).array().abs();
  tmp_.head(mIneq) = bineq;
  tmp_.head(mIneq).noalias() -= Aineq*x0;
  x_.segment(n + mEq, mIneq) = (tmp_.array() > 0)
    .select(tmp_, VectorType::Zero(mIneq));

  w_ = w0;
  // add inequality constraints to the working set
  for(Index i = 0; i < mIneq; ++i)
  {
    // the z_i value is 0 then the z_i >= 0 constraint is active
    // in the other case then the a_i^t x_i > b_i constraint is active
    if(x_(i + n + mEq) == 0.)
    {
      w_.push_back(mIneq + i);
    }
    else
    {
      w_.push_back(i);
    }
  }

  solver.solve(G_, c_, Aeq_, beq, Aineq_, bineq_, x_, w_, maxIter);

  x_ = solver.x();
  w_ = solver.w();
}

}

}
