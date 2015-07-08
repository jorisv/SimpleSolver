// includes
// std
#include <iostream>

// boost
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Algo
#include <boost/test/unit_test.hpp>

// Eigen
#define EIGEN_RUNTIME_NO_MALLOC 1
#include <Eigen/Core>
#include <Eigen/QR>

// EQP
#include "eq_qp_ns.h"


using namespace Eigen;

BOOST_AUTO_TEST_CASE(QRTest)
{
  MatrixXd A(2,3);
  A << 1., 0., 1.,
       0., 1., 1.;

  ColPivHouseholderQR<MatrixXd> qr(A.transpose());
  MatrixXd Q(qr.householderQ());
  MatrixXd Y(Q.topLeftCorner(3, 2));
  MatrixXd Z(Q.topRightCorner(3, 1));

  // check if the nullspace computation is OK (AZ = 0)
  MatrixXd AZ(A*Z);
  BOOST_CHECK(AZ.isZero(1e-8));

  // check if AY is full rank
  MatrixXd AY(A*Y);
  qr.compute(AY);
  BOOST_CHECK_EQUAL(qr.rank(), 2);
}


BOOST_AUTO_TEST_CASE(EqQpNsTest)
{
  MatrixXd G{3,3}, A{2,3};
  VectorXd c{3}, b{2};
  VectorXd x{3}, lambda{2};
  G << 6., 2., 1.,
       2., 5., 2.,
       1., 2., 4.;
  A << 1., 0., 1.,
       0., 1., 1.;

  c << -8., -3., -3.;
  b << 3., 0.;

  x << 2., -1., 1.;
  lambda << 3., -2.;

  qp::EqQpNullSpace<MatrixXd> eqQpNs(3, 2);
  eqQpNs.compute(G, A);
  eqQpNs.solve(c, b);

  BOOST_CHECK((x - eqQpNs.x()).isZero(1e-8));
  BOOST_CHECK((lambda - eqQpNs.lambda()).isZero(1e-8));
}
