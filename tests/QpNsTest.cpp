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
#define BOOST_TEST_MODULE QpNs
#include <boost/test/unit_test.hpp>

// Eigen
#include <Eigen/Core>
#include <Eigen/QR>

// SimpleSolver
#include "SimpleSolver"


using namespace Eigen;


BOOST_AUTO_TEST_CASE(QpNsTest)
{
  MatrixXd G{3,3}, Aeq{2,3}, Aineq{0,3};
  VectorXd c{3}, beq{2, 1}, bineq{0, 1};
  VectorXd x{3};
  G << 6., 2., 1.,
       2., 5., 2.,
       1., 2., 4.;
  Aeq << 1., 0., 1.,
       0., 1., 1.;

  c << -8., -3., -3.;
  beq << 3., 0.;

  x << 2., -1., 1.;

  simple_solver::QpNullSpace<MatrixXd, simple_solver::LoggerType::Full> qpNs{3, 2, 0};
  qpNs.solve(G, c, Aeq, beq, Aineq, bineq, x, {});

  BOOST_CHECK((x - qpNs.x()).isZero(1e-8));
  // just on iteration
  BOOST_CHECK_EQUAL(qpNs.logger().datas.size(), 1);
}


BOOST_AUTO_TEST_CASE(QpNsTest1)
{
  MatrixXd G{2,2};
  VectorXd c{2};
  VectorXd x{2}, x0{2};

  const double inf = std::numeric_limits<double>::infinity();
  typedef simple_solver::StdConstraints<MatrixXd> Constraints;
  Constraints constrs{2, 0, 4};

  G << 2., 0.,
       0., 2.;
  c << -2., -5.;

  constrs.Agineq() << 1., -2.,
                     -1., -2.,
                      1., 0.,
                      0., 1.;
  constrs.Agl() << -2., -6, 0., 0.;
  constrs.Agu() << 2., inf, inf, inf;
  constrs.userW() = {
    {0, Constraints::StdWIndex::Type::Upper},
    {3, Constraints::StdWIndex::Type::Lower}
  };

  constrs.buildIneq();
  constrs.buildSolverW(constrs.userW());

  BOOST_CHECK_EQUAL(constrs.Aineq().rows(), 5);
  BOOST_CHECK_EQUAL(constrs.solverW().size(), 2);

  x << 1.4, 1.7;
  x0 << 2., 0.;

  typedef simple_solver::QpNullSpace<MatrixXd, simple_solver::LoggerType::Full> Solver;

  Solver qpNs{2, 0, 5};
  qpNs.solve(G, c, constrs, x0);

  BOOST_CHECK((x - qpNs.x()).isZero(1e-8));
  const Solver::Logger& logger = qpNs.logger();

  BOOST_CHECK_EQUAL(logger.datas.size(), 6);

  // iter 0
  VectorXd lambda0{2};
  lambda0 << -2., -1.;
  BOOST_CHECK(logger.datas[0].p.isZero(1e-8));
  BOOST_CHECK_SMALL((logger.datas[0].lambda - lambda0).norm(), 1e-8);
  BOOST_CHECK_EQUAL(logger.datas[0].iterType, Solver::Logger::RemoveW);
  BOOST_CHECK_EQUAL(logger.datas[0].iterW,
    (constrs.userWToSolverW().at({0, Constraints::StdWIndex::Type::Upper})));

  // iter 1
  VectorXd p1{2};
  p1 << -1., 0.;
  BOOST_CHECK_SMALL((logger.datas[1].p - p1).norm(), 1e-8);
  BOOST_CHECK_EQUAL(logger.datas[1].iterType, Solver::Logger::KeepW);

  // iter 2
  VectorXd x2{2}, lambda2{1};
  x2 << 1., 0.;
  lambda2 << -5.;
  BOOST_CHECK(logger.datas[2].p.isZero(1e-8));
  BOOST_CHECK_SMALL((logger.datas[2].x - x2).norm(), 1e-8);
  BOOST_CHECK_SMALL((logger.datas[2].lambda - lambda2).norm(), 1e-8);
  BOOST_CHECK_EQUAL(logger.datas[2].iterType, Solver::Logger::RemoveW);
  BOOST_CHECK_EQUAL(logger.datas[2].iterW,
    (constrs.userWToSolverW().at({3, Constraints::StdWIndex::Type::Lower})));

  // iter 3
  VectorXd p3{2};
  p3 << 0., 2.5;
  BOOST_CHECK_SMALL((logger.datas[3].p - p3).norm(), 1e-8);
  BOOST_CHECK_EQUAL(logger.datas[3].iterType, Solver::Logger::AddW);
  BOOST_CHECK_EQUAL(logger.datas[3].iterW,
    (constrs.userWToSolverW().at({0, Constraints::StdWIndex::Type::Lower})));

  // iter 4
  VectorXd x4{2}, p4{2};
  x4 << 1., 1.5;
  p4 << 0.4, 0.2;
  BOOST_CHECK_SMALL((logger.datas[4].x - x4).norm(), 1e-8);
  BOOST_CHECK_SMALL((logger.datas[4].p - p4).norm(), 1e-8);
  BOOST_CHECK_EQUAL(logger.datas[4].iterType, Solver::Logger::KeepW);

  // iter 5
  VectorXd x5{2}, lambda5{1};
  x5 << 1.4, 1.7;
  lambda5 << 0.8;
  BOOST_CHECK(logger.datas[5].p.isZero(1e-8));
  BOOST_CHECK_SMALL((logger.datas[5].x - x5).norm(), 1e-8);
  BOOST_CHECK_SMALL((logger.datas[5].lambda - lambda5).norm(), 1e-8);
  BOOST_CHECK_EQUAL(logger.datas[5].iterType, Solver::Logger::KeepW);
}


// test QpStartTypeI on the problem define p. 474 fig. 16.3
// in numerical opitimization textboook.
BOOST_AUTO_TEST_CASE(QpStartTypeITest1)
{
  MatrixXd G{2,2};
  VectorXd c{2};
  VectorXd x{2}, x0_1{2}, x0_2{2}, x0_3{2}, x0_4{2}, x0_5{2}, x0_6{2}, x0_7{2};

  const double inf = std::numeric_limits<double>::infinity();
  typedef simple_solver::StdConstraints<MatrixXd> Constraints;
  Constraints constrs{2, 0, 4};

  G << 2., 0.,
       0., 2.;
  c << -2., -5.;

  constrs.Agineq() << 1., -2.,
                     -1., -2.,
                      1., 0.,
                      0., 1.;
  constrs.Agl() << -2., -6, 0., 0.;
  constrs.Agu() << 2., inf, inf, inf;

  constrs.buildIneq();

  x << 1.4, 1.7;
  x0_1 << -1., -1.; // a3(L) and a4(L) violated
  x0_2 << 0.1, -.5; // a4(L) violated
  x0_3 << 0., -2.; // a1(U), a3(L), a4(L) violated
  x0_4 << 5., -1.; // a1(U), a4(L) violated
  x0_5 << 5., 1.; // a1(U), a2(L) violated
  x0_6 << 2., 3.; // a1(L), a2(L) violated
  x0_7 << 2., 1.; // no violation

  typedef simple_solver::QpNullSpace<MatrixXd, simple_solver::LoggerType::Full> Solver;
  typedef simple_solver::QpStartTypeI<Solver> SolverStart;

  Solver qpNs{2, 0, 5};
  SolverStart qpNsStart{2, 0, 5};

  auto check = [&qpNsStart, &constrs, &qpNs, &G, &c, &x](const VectorXd& x0)
  {
    BOOST_CHECK(qpNsStart.findInit(constrs, x0, 1e-8) ==
      SolverStart::Exit::Success);
    constrs.solverW() = qpNsStart.w();
    qpNs.solve(G, c, constrs, qpNsStart.x());
    BOOST_CHECK((x - qpNs.x()).isZero(1e-8));
  };

  check(x0_1);
  check(x0_2);
  check(x0_3);
  check(x0_4);
  check(x0_5);
  check(x0_6);
  check(x0_7);
}


// mirror the previous problem on the x2 axis (not the objective fonction)
BOOST_AUTO_TEST_CASE(QpStartTypeITest2)
{
  MatrixXd G{2,2};
  VectorXd c{2};
  VectorXd x{2}, x0_1{2}, x0_2{2}, x0_3{2}, x0_4{2}, x0_5{2}, x0_6{2}, x0_7{2};

  const double inf = std::numeric_limits<double>::infinity();
  typedef simple_solver::StdConstraints<MatrixXd> Constraints;
  Constraints constrs{2, 0, 4};

  G << 2., 0.,
       0., 2.;
  c << -2., -5.;

  constrs.Agineq() << -1., -2.,
                       1., -2.,
                      -1.,  0.,
                       0.,  1.;
  constrs.Agl() << -2., -6, 0., 0.;
  constrs.Agu() << 2., inf, inf, inf;

  constrs.buildIneq();

  x << 0., 1.;
  x0_1 << 1., -1.; // a3(L) and a4(L) violated
  x0_2 << -0.1, -.5; // a4(L) violated
  x0_3 << -0., -2.; // a1(U), a3(L), a4(L) violated
  x0_4 << -5., -1.; // a1(U), a4(L) violated
  x0_5 << -5., 1.; // a1(U), a2(L) violated
  x0_6 << -2., 3.; // a1(L), a2(L) violated
  x0_7 << -2., 1.; // no violation

  typedef simple_solver::QpNullSpace<MatrixXd, simple_solver::LoggerType::Full> Solver;
  typedef simple_solver::QpStartTypeI<Solver> SolverStart;

  Solver qpNs{2, 0, 5};
  SolverStart qpNsStart{2, 0, 5};

  auto check = [&qpNsStart, &constrs, &qpNs, &G, &c, &x](const VectorXd& x0)
  {
    BOOST_CHECK(qpNsStart.findInit(constrs, x0, 1e-8) ==
      SolverStart::Exit::Success);
    constrs.solverW() = qpNsStart.w();
    qpNs.solve(G, c, constrs, qpNsStart.x());
    BOOST_CHECK((x - qpNs.x()).isZero(1e-8));
  };

  check(x0_1);
  check(x0_2);
  check(x0_3);
  check(x0_4);
  check(x0_5);
  check(x0_6);
  check(x0_7);
}
