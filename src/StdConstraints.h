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


template <typename MatrixType>
class StdConstraints
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
  typedef Matrix<Scalar, Dynamic, 1, Options> VectorType;

  struct StdWIndex
  {
    enum struct Type {Lower, Upper};

    Index index;
    Type type;

    bool operator<(const StdWIndex& other) const
    {
      return (index < other.index) ||
        (index == other.index && type < other.type);
    }

    bool operator==(const StdWIndex& other) const
    {
      return (index == other.index) && (type == other.type);
    }

    bool operator!=(const StdWIndex& other) const
    {
      return !this->operator==(other);
    }
  };

  struct StdWIndexHasher
  {
    std::size_t operator()(const StdWIndex& swi) const
    {
      typedef typename std::underlying_type<typename StdWIndex::Type>::type
        h2Type;

      std::size_t h1 = std::hash<Index>()(swi.index);
      std::size_t h2 = std::hash<h2Type>()(h2Type(swi.type));
      return h1 ^ (h2 << 1);
    }
  };

public:
  StdConstraints();
  StdConstraints(Index n, Index mEq, Index mGIneq);

  void resize(Index n, Index mEq, Index mGIneq);

  const std::vector<StdWIndex>& userW() const
  {
    return userW_;
  }

  std::vector<StdWIndex>& userW()
  {
    return userW_;
  }

  const std::vector<Index>& solverW() const
  {
    return solverW_;
  }

  const MatrixType& Aeq() const
  {
    return Aeq_;
  }

  MatrixType& Aeq()
  {
    return Aeq_;
  }

  const VectorType& beq() const
  {
    return beq_;
  }

  VectorType& beq()
  {
    return beq_;
  }

  const MatrixType& Agineq() const
  {
    return Agineq_;
  }

  MatrixType& Agineq()
  {
    return Agineq_;
  }

  const VectorType& Agl() const
  {
    return Agl_;
  }

  VectorType& Agl()
  {
    return Agl_;
  }

  const VectorType& Agu() const
  {
    return Agu_;
  }

  VectorType& Agu()
  {
    return Agu_;
  }

  const MatrixType& Aineq() const
  {
    return Aineq_;
  }

  const VectorType& bineq() const
  {
    return bineq_;
  }

  void buildIneq();

  void buildSolverW(const std::vector<StdWIndex>& userW);
  void buildUserW(const std::vector<Index>& solverW);

private:
  std::vector<StdWIndex> userW_;
  std::vector<Index> solverW_;
  std::unordered_map<Index, StdWIndex> solverWToUserW_;
  std::unordered_map<StdWIndex, Index, StdWIndexHasher> userWToSolverW_;

  MatrixType Aeq_, Agineq_;
  VectorType beq_, Agl_, Agu_;

  MatrixType Aineq_;
  VectorType bineq_;
};



template <typename MatrixType>
inline StdConstraints<MatrixType>::StdConstraints()
{}


template <typename MatrixType>
inline
StdConstraints<MatrixType>::StdConstraints(Index n, Index mEq, Index mGIneq)
  : Aeq_(mEq, n)
  , Agineq_(mGIneq, n)
  , beq_(mEq)
  , Agl_(mGIneq)
  , Agu_(mGIneq)
{}


template <typename MatrixType>
inline void StdConstraints<MatrixType>::resize(Index n, Index mEq, Index mGIneq)
{
  Aeq_.resize(mEq, n);
  Agineq_.resize(mGIneq, n);
  beq_.resize(mEq);
  Agl_.resize(mGIneq);
  Agu_.resize(mGIneq);
}


template <typename MatrixType>
void StdConstraints<MatrixType>::buildIneq()
{
  eigen_assert(Agineq_.rows() == Agl_.rows());
  eigen_assert(Agineq_.rows() == Agu_.rows());

  Index mIneq = 0;
  for(Index i = 0; i < Agineq_.rows(); ++i)
  {
    if(Agl_[i] != -std::numeric_limits<Scalar>::infinity())
    {
      ++mIneq;
    }
    if(Agu_[i] != std::numeric_limits<Scalar>::infinity())
    {
      ++mIneq;
    }
  }

  Aineq_.resize(mIneq, Agineq_.cols());
  bineq_.resize(mIneq);

  userWToSolverW_.clear();
  solverWToUserW_.clear();

  mIneq = 0;
  for(Index i = 0; i < Agineq_.rows(); ++i)
  {
    if(Agl_[i] != -std::numeric_limits<Scalar>::infinity())
    {
      Aineq_.row(mIneq) = Agineq_.row(i);
      bineq_(mIneq) = Agl_(i);
      userWToSolverW_[{i, StdWIndex::Type::Lower}] = mIneq;
      solverWToUserW_[mIneq] = {i, StdWIndex::Type::Lower};
      ++mIneq;
    }
    if(Agu_[i] != std::numeric_limits<Scalar>::infinity())
    {
      Aineq_.row(mIneq) = -Agineq_.row(i);
      bineq_(mIneq) = -Agl_(i);
      userWToSolverW_[{i, StdWIndex::Type::Upper}] = mIneq;
      solverWToUserW_[mIneq] = {i, StdWIndex::Type::Upper};
      ++mIneq;
    }
  }
}


template <typename MatrixType>
void StdConstraints<MatrixType>::buildSolverW(const std::vector<StdWIndex>& userW)
{
  solverW_.resize(userW.size());
  for(Index i = 0; i < Index(userW.size()); ++i)
  {
    solverW_[i] = userWToSolverW_[userW[i]];
  }
}


template <typename MatrixType>
void StdConstraints<MatrixType>::buildUserW(const std::vector<Index>& solverW)
{
  userW_.resize(solverW.size());
  for(Index i = 0; i < Index(solverW.size()); ++i)
  {
    userW_[i] = solverWToUserW_[solverW[i]];
  }
}


}

}
