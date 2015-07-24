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
#include "Qp.h"


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

  qp::QpNullSpace<MatrixXd, qp::LoggerType::Full> qpNs{3, 2, 0};
  qpNs.solve(G, c, Aeq, beq, Aineq, bineq, x, {});

  BOOST_CHECK((x - qpNs.x()).isZero(1e-8));
  // just on iteration
  BOOST_CHECK_EQUAL(qpNs.logger().datas.size(), 1);
}


BOOST_AUTO_TEST_CASE(QpNsTest1)
{
  MatrixXd G{2,2}, Aeq{0,2}, Aineq{5,2};
  VectorXd c{2}, beq{0, 1}, bineq{5, 1};
  VectorXd x{2}, x0{2};

  G << 2., 0.,
       0., 2.;
  Aineq << 1., -2.,
           -1., -2.,
           -1, 2.,
           1., 0.,
           0., 1.;

  c << -2., -5.;
  bineq << -2., -6., -2., 0., 0.;

  x << 1.4, 1.7;
  x0 << 2., 0.;

  typedef qp::QpNullSpace<MatrixXd, qp::LoggerType::Full> Solver;

  Solver qpNs{2, 0, 5};
  qpNs.solve(G, c, Aeq, beq, Aineq, bineq, x0, {2, 4});

  BOOST_CHECK((x - qpNs.x()).isZero(1e-8));
  const Solver::Logger& logger = qpNs.logger();

  BOOST_CHECK_EQUAL(logger.datas.size(), 6);

  // iter 0
  VectorXd lambda0{2};
  lambda0 << -2., -1.;
  BOOST_CHECK(logger.datas[0].p.isZero(1e-8));
  BOOST_CHECK_SMALL((logger.datas[0].lambda - lambda0).norm(), 1e-8);
  BOOST_CHECK_EQUAL(logger.datas[0].iterType, Solver::Logger::RemoveW);
  BOOST_CHECK_EQUAL(logger.datas[0].iterW, 2);

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
  BOOST_CHECK_EQUAL(logger.datas[2].iterW, 4);

  // iter 3
  VectorXd p3{2};
  p3 << 0., 2.5;
  BOOST_CHECK_SMALL((logger.datas[3].p - p3).norm(), 1e-8);
  BOOST_CHECK_EQUAL(logger.datas[3].iterType, Solver::Logger::AddW);
  BOOST_CHECK_EQUAL(logger.datas[3].iterW, 0);

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


BOOST_AUTO_TEST_CASE(QpStartTypeITest)
{
  MatrixXd G{2,2}, Aeq{0,2}, Aineq{5,2};
  VectorXd c{2}, beq{0, 1}, bineq{5, 1};
  VectorXd x{2}, x0_1{2}, x0_2{2}, x0_3{2};

  G << 2., 0.,
       0., 2.;
  Aineq << 1., -2.,
           -1., -2.,
           -1, 2.,
           1., 0.,
           0., 1.;

  c << -2., -5.;
  bineq << -2., -6., -2., 0., 0.;

  x << 1.4, 1.7;
  x0_1 << -1., -1.; // 2 constraints violated
  x0_2 << 0.1, -.5; // 1 constraints violated
  x0_3 << 0., -2.; // 3 constraints violated

  typedef qp::QpNullSpace<MatrixXd, qp::LoggerType::Full> Solver;
  typedef qp::QpStartTypeI<Solver> SolverStart;

  Solver qpNs{2, 0, 5};
  SolverStart qpNsStart{2, 0, 5};

  BOOST_CHECK(qpNsStart.findInit(Aeq, beq, Aineq, bineq, x0_1, 1e-8) ==
    SolverStart::Exit::Success);
  std::cout << qpNsStart.x().transpose() << "\n";
  for(auto i: qpNsStart.w()) std::cout << i << " ";
  std::cout << "\n";

  qpNs.solve(G, c, Aeq, beq, Aineq, bineq, qpNsStart.x(), qpNsStart.w());
  BOOST_CHECK((x - qpNs.x()).isZero(1e-8));

  /*
  BOOST_CHECK(qpNsStart.findInit(Aeq, beq, Aineq, bineq, x0_2, 1e-8) ==
    SolverStart::Exit::Success);
  std::cout << qpNsStart.x().transpose() << "\n";
  for(auto i: qpNsStart.w()) std::cout << i << " ";
  std::cout << "\n";
  */

  BOOST_CHECK(qpNsStart.findInit(Aeq, beq, Aineq, bineq, x0_3, 1e-8) ==
    SolverStart::Exit::Success);
  std::cout << qpNsStart.x().transpose() << "\n";
  for(auto i: qpNsStart.w()) std::cout << i << " ";
  std::cout << "\n";

  qpNs.solve(G, c, Aeq, beq, Aineq, bineq, qpNsStart.x(), qpNsStart.w());
  BOOST_CHECK((x - qpNs.x()).isZero(1e-8));
}
