// includes
// std
#include <iostream>

// boost
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Algo
#include <boost/test/unit_test.hpp>

// Eigen
#include <Eigen/Core>
#include <Eigen/QR>

// EQP
#include "qp_ns.h"


using namespace Eigen;


BOOST_AUTO_TEST_CASE(EqQpNsTest)
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

  qp::QpNullSpace<MatrixXd> qpNs{3, 2, 0};
  qpNs.solve(G, c, Aeq, beq, Aineq, bineq, x, {});

  BOOST_CHECK((x - qpNs.x()).isZero(1e-8));
}


BOOST_AUTO_TEST_CASE(EqQpNsTest1)
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

  qp::QpNullSpace<MatrixXd> qpNs{2, 0, 5};
  qpNs.solve(G, c, Aeq, beq, Aineq, bineq, x0, {2, 4});

  BOOST_CHECK((x - qpNs.x()).isZero(1e-8));
}
