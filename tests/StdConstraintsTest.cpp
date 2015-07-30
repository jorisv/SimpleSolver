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

// includes
// std
#include <iostream>

// boost
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE StdConstraints
#include <boost/test/unit_test.hpp>

// Eigen
#define EIGEN_RUNTIME_NO_MALLOC
#include <Eigen/Core>

// SimpleSolver
#include "SimpleSolver"


using namespace Eigen;


BOOST_AUTO_TEST_CASE(StdConstaints)
{
  typedef simple_solver::StdConstraints<MatrixXd> StdConstraintsType;
  typedef StdConstraintsType::StdWIndex StdWIndex;
  typedef StdConstraintsType::Index Index;

  StdConstraintsType constrs(5, 2, 3);

  BOOST_CHECK_EQUAL(constrs.Aeq().rows(), 2);
  BOOST_CHECK_EQUAL(constrs.Aeq().cols(), 5);
  BOOST_CHECK_EQUAL(constrs.beq().rows(), 2);

  BOOST_CHECK_EQUAL(constrs.Agineq().rows(), 3);
  BOOST_CHECK_EQUAL(constrs.Agineq().cols(), 5);
  BOOST_CHECK_EQUAL(constrs.Agl().rows(), 3);
  BOOST_CHECK_EQUAL(constrs.Agu().rows(), 3);

  constrs.resize(6, 3, 3);

  BOOST_CHECK_EQUAL(constrs.Aeq().rows(), 3);
  BOOST_CHECK_EQUAL(constrs.Aeq().cols(), 6);
  BOOST_CHECK_EQUAL(constrs.beq().rows(), 3);

  BOOST_CHECK_EQUAL(constrs.Agineq().rows(), 3);
  BOOST_CHECK_EQUAL(constrs.Agineq().cols(), 6);
  BOOST_CHECK_EQUAL(constrs.Agl().rows(), 3);
  BOOST_CHECK_EQUAL(constrs.Agu().rows(), 3);

  const double inf = std::numeric_limits<double>::infinity();
  constrs.Agineq().setRandom();
  constrs.Agl() <<  1., -inf, -3; // 2 ineq
  constrs.Agu() << inf,  1.,  1.; // 2 ineq

  constrs.buildIneq();

  BOOST_CHECK_EQUAL(constrs.Aineq().rows(), 4); // 4 ineq
  BOOST_CHECK_EQUAL(constrs.Aineq().cols(), 6);
  BOOST_CHECK_EQUAL(constrs.bineq().rows(), 4);

  BOOST_CHECK_EQUAL(constrs.Agineq().row(0), constrs.Aineq().row(0));
  BOOST_CHECK_EQUAL(constrs.Agineq().row(1), -constrs.Aineq().row(1));
  BOOST_CHECK_EQUAL(constrs.Agineq().row(2), constrs.Aineq().row(2));
  BOOST_CHECK_EQUAL(constrs.Agineq().row(2), -constrs.Aineq().row(3));

  std::vector<StdWIndex> userW =
    {{0, StdWIndex::Type::Lower},
     {1, StdWIndex::Type::Upper}};
  std::vector<Index> solverW = {0, 1};

  constrs.buildSolverW(userW);
  constrs.buildUserW(solverW);

  BOOST_CHECK_EQUAL(userW.size(), constrs.userW().size());
  for(std::size_t i = 0; i < userW.size(); ++i)
  {
    BOOST_CHECK(userW[i] == constrs.userW()[i]);
  }

  BOOST_CHECK_EQUAL(solverW.size(), constrs.solverW().size());
  for(std::size_t i = 0; i < solverW.size(); ++i)
  {
    BOOST_CHECK(solverW[i] == constrs.solverW()[i]);
  }
}
