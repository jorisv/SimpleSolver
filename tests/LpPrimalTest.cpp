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
#define BOOST_TEST_MODULE LpPrimal
#include <boost/test/unit_test.hpp>

// Eigen
#include <Eigen/Core>
#include <Eigen/QR>

// SimpleSolver
#include "SimpleSolver"


using namespace Eigen;


BOOST_AUTO_TEST_CASE(LpPrimalTest1)
{
  MatrixXd Aeq{0,2}, Aineq{5,2};
  VectorXd c{2}, beq{0, 1}, bineq{5, 1};
  VectorXd x{2};

  Aineq << 4., -1.,
           -1., -2.,
           -2., -1.,
           -5., 2.,
           4., 2.;

  c << -1., -1.;
  bineq << 4., -10., -11., -23., 4.;
  x << 4., 3.;

  typedef simple_solver::LpPrimal<MatrixXd, simple_solver::LoggerType::Full> LpSolver;
  LpSolver lp{2, 0, 5};
  BOOST_CHECK(lp.solve(c, Aeq, beq, Aineq, bineq, {0, 4}) ==
    LpSolver::Exit::Success);

  BOOST_CHECK((x - lp.x()).isZero(1e-8));
  BOOST_CHECK_EQUAL(lp.logger().datas.size(), 3);

  // iter 0
  VectorXd x0{2}, l0{2}, d0{2};
  x0 << 1., 0.; l0 << 1./6., -5/12.; d0 << 1./12., 1./3.;

  BOOST_CHECK_SMALL((lp.logger().datas[0].x - x0).norm(), 1e-8);
  BOOST_CHECK_SMALL((lp.logger().datas[0].lambda - l0).norm(), 1e-8);
  BOOST_CHECK_SMALL((lp.logger().datas[0].d - d0).norm(), 1e-8);
  BOOST_CHECK_EQUAL(lp.logger().datas[0].lW, 4);
  BOOST_CHECK_EQUAL(lp.logger().datas[0].eW, 1);

  // iter 1
  VectorXd x1{2}, l1{2}, d1{2};
  x1 << 2., 4.; l1 << -1./9., 5./9.; d1 << 2./9., -1./9.;

  BOOST_CHECK_SMALL((lp.logger().datas[1].x - x1).norm(), 1e-8);
  BOOST_CHECK_SMALL((lp.logger().datas[1].lambda - l1).norm(), 1e-8);
  BOOST_CHECK_SMALL((lp.logger().datas[1].d - d1).norm(), 1e-8);
  BOOST_CHECK_EQUAL(lp.logger().datas[1].lW, 0);
  BOOST_CHECK_EQUAL(lp.logger().datas[1].eW, 2);

  // iter 2
  VectorXd x2{2}, l2{2};
  x2 << 4., 3.; l2 << 1./3., 1./3.;

  BOOST_CHECK_SMALL((lp.logger().datas[2].x - x2).norm(), 1e-8);
  BOOST_CHECK_SMALL((lp.logger().datas[2].lambda - l2).norm(), 1e-8);
}


BOOST_AUTO_TEST_CASE(LpPrimalTest2)
{
  VectorXd c{2};
  VectorXd x{2};

  const double inf = std::numeric_limits<double>::infinity();
  typedef simple_solver::StdConstraints<MatrixXd> Constraints;
  Constraints constrs{2, 0, 4};

  constrs.Agineq() << 1., 0.,
                      0, 1,
                      1, -1,
                     -3./2., -1.;

  constrs.Agl() << 0., 0., -1., -3.;
  constrs.Agu() << inf, inf, inf, inf;
  constrs.userW() = {
    {0, Constraints::StdWIndex::Type::Lower},
    {1, Constraints::StdWIndex::Type::Lower}};

  constrs.buildIneq();
  constrs.buildSolverW(constrs.userW());

  c << -0., -1.;
  x << 4./5., 9./5.;

  typedef simple_solver::LpPrimal<MatrixXd, simple_solver::LoggerType::Full> LpSolver;
  LpSolver lp{2, 0, 4};
  BOOST_CHECK(lp.solve(c, constrs) == LpSolver::Exit::Success);

  BOOST_CHECK((x - lp.x()).isZero(1e-8));
  BOOST_CHECK_EQUAL(lp.logger().datas.size(), 3);

  // iter 0
  VectorXd x0{2}, l0{2}, d0{2};
  x0 << 0., 0.; l0 << 0., -1.; d0 << 0., 1.;

  BOOST_CHECK_SMALL((lp.logger().datas[0].x - x0).norm(), 1e-8);
  BOOST_CHECK_SMALL((lp.logger().datas[0].lambda - l0).norm(), 1e-8);
  BOOST_CHECK_SMALL((lp.logger().datas[0].d - d0).norm(), 1e-8);
  BOOST_CHECK_EQUAL(lp.logger().datas[0].lW, 1);
  BOOST_CHECK_EQUAL(lp.logger().datas[0].eW, 2);

  // iter 1
  VectorXd x1{2}, l1{2}, d1{2};
  x1 << 0., 1.; l1 << -1., 1.; d1 << 1., 1.;

  BOOST_CHECK_SMALL((lp.logger().datas[1].x - x1).norm(), 1e-8);
  BOOST_CHECK_SMALL((lp.logger().datas[1].lambda - l1).norm(), 1e-8);
  BOOST_CHECK_SMALL((lp.logger().datas[1].d - d1).norm(), 1e-8);
  BOOST_CHECK_EQUAL(lp.logger().datas[1].lW, 0);
  BOOST_CHECK_EQUAL(lp.logger().datas[1].eW, 3);

  // iter 2
  VectorXd x2{2}, l2{2};
  x2 << 4./5., 9./5.; l2 << 2./5., 3./5.;

  BOOST_CHECK_SMALL((lp.logger().datas[2].x - x2).norm(), 1e-8);
  BOOST_CHECK_SMALL((lp.logger().datas[2].lambda - l2).norm(), 1e-8);
}
