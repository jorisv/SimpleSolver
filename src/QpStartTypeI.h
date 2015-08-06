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

  typedef StdConstraints<MatrixType> StdConstraintsType;
  typedef typename StdConstraintsType::StdWIndex StdWIndex;

public:
  QpStartTypeI();
  QpStartTypeI(Index n, Index mEq, Index mIneq);

  Exit findInit(const StdConstraintsType& constrs,
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

  std::vector<Index> w_;
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
typename QpStartTypeI<QpType>::Exit QpStartTypeI<QpType>::findInit(
  const StdConstraintsType& constrs,
  const QpStartTypeI<QpType>::VectorType& x0,
  Scalar precision,
  int maxIter)
{
  eigen_assert(constrs.Aeq().cols() == constrs.Aineq().cols());
  eigen_assert(constrs.Aeq().rows() == constrs.beq().rows());
  eigen_assert(constrs.Aineq().rows() == constrs.bineq().rows());

  const Index n = constrs.Aeq().cols();
  const Index mEq = constrs.Aeq().rows();
  const Index mIneq = constrs.Aineq().rows();
  const Index newN = n + mEq + mIneq;
  const Index newIneq = n + mEq + mIneq*3;
  const Index zEqPos = n;
  const Index zIneqPos = n + mEq;

  // clear the buffers
  w_.clear();

  // compute constraint violation
  tmp_.resize(mEq + mIneq);
  tmp_.head(mEq) = -constrs.beq();
  tmp_.head(mEq).noalias() += constrs.Aeq()*x0;
  tmp_.tail(mIneq) = -constrs.bineq();
  tmp_.tail(mIneq).noalias() += constrs.Aineq()*x0;

  if((tmp_.head(mEq).array().abs() >= precision).all() &&
    (tmp_.tail(mIneq).array() >= precision).all())
  {
    x_ = x0;
    w_.clear();
    return Exit::Success;
  }

  // fill the matrices
  // minimize z
  c_.resize(newN);
  c_.head(n).setZero();
  c_.tail(newN - n).setOnes();
  x_.resize(n);

  //   x             zEq            zIneq
  // [Aeq    -sign(Aeq*x0 - beq)      0]   == beq
  Aeq_.resize(mEq, newN);
  Aeq_.block(0, 0, mEq, n) = constrs.Aeq();
  Aeq_.block(0, zEqPos, mEq, mEq) =
    ((tmp_.head(mEq).array() >= Scalar(0.))
     .select(-VectorType::Ones(mEq), VectorType::Ones(mEq))).asDiagonal();
  Aeq_.block(0, zIneqPos, mEq, mIneq).setZero();

  Aineq_.resize(newIneq, newN);
  bineq_.resize(newIneq);

  //   x             zEq            zIneq
  // [Aineq           0               I]    >= bineq
  // [0               I               0]    >= 0
  // [0               0               I]    >= 0
  // [M               0               0]    >= M*x0
  // [0               0              -I]    >= max(bineq - Aineq*x0, 0)

  // line 1
  Aineq_.block(0, 0, mIneq, n) = constrs.Aineq();
  Aineq_.block(0, zEqPos, mIneq, mEq).setZero();
  Aineq_.block(0, zIneqPos, mIneq, mIneq).setIdentity();

  // line 2-3
  const Index ineqL2 = mIneq;
  Aineq_.block(ineqL2, 0, mEq + mIneq, n).setZero();
  Aineq_.block(ineqL2, zEqPos, mEq + mIneq, mEq + mIneq).setIdentity();

  // line 4
  const Index ineqL4 = ineqL2 + mEq + mIneq;
  // compute the M diagonal matrix
  // if the n_i variable is imply in the ineq_i violated constraint
  // then the n_i variable can only move in the ineq_i constraint direction
  // If no constraint are violated by the n_i variable we don't allow it
  // to decrease
  /// @TODO find if there is a better way...
  Aineq_.block(ineqL4, 0, n, n).setIdentity();
  for(Index ni = 0; ni < n; ++ni)
  {
    for(Index ineqi = 0; ineqi < mIneq; ++ineqi)
    {
      Scalar val = constrs.Aineq()(ineqi, ni);
      if(tmp_(mEq + ineqi) < Scalar(0.) && val != Scalar(0.))
      {
        Aineq_(ineqL4 + ni, ni) =
          constrs.Aineq()(ineqi, ni) > Scalar(0.) ? Scalar(1.) : Scalar(-1.);
        break;
      }
    }
  }
  Aineq_.block(ineqL4, zEqPos, n, mEq + mIneq).setZero();

  // line 5
  const Index ineqL5 = ineqL4 + n;
  Aineq_.block(ineqL5, 0, mIneq, n + mEq).setZero();
  // set to minus identity because we don't allow to increase zIneq
  Aineq_.block(ineqL5, zIneqPos, mIneq, mIneq) =
    -MatrixType::Identity(mIneq, mIneq);

  bineq_.segment(0, mIneq) = constrs.bineq();
  bineq_.segment(ineqL2, mEq + mIneq).setZero();
  bineq_.segment(ineqL4, n) =
    (Aineq_.block(ineqL4, 0, n, n).diagonal().array())*x0.array();
  bineq_.segment(ineqL5, mIneq) = (tmp_.tail(mIneq).array() <= Scalar(0.))
    .select(tmp_.tail(mIneq), VectorType::Zero(mIneq));

  // fill w_ with line 4 and 5
  for(Index i = 0; i < n + mIneq; ++i)
  {
    w_.push_back(ineqL4 + i);
  }

  eigen_assert(mEq + Index(w_.size()) == newN);

  Exit status = Exit(solver_.solve(c_, Aeq_, constrs.beq(), Aineq_, bineq_,
    w_, maxIter, ineqL4));

  if(status == Exit::Success)
  {
    // if [zEq zIneq] != 0 then the problem is probably infeasible
    if((solver_.x().tail(mEq + mIneq).array().abs() > precision).any())
    {
      return Exit::Infeasible;
    }

    x_ = solver_.x().head(n);

    // only keep the n first active constraints in Aineq
    w_.clear();
    for(std::size_t i = 0;
        i < solver_.w().size() && w_.size() < std::size_t(n);
        ++i)
    {
      const Index wi = solver_.w()[i];
      if(wi < mIneq)
      {
        w_.push_back(wi);
      }
    }
  }

  return status;
}

}

}
