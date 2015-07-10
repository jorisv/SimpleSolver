// includes
// std
#include <iostream>

// boost
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Algo
#include <boost/test/unit_test.hpp>

// Eigen
#define EIGEN_RUNTIME_NO_MALLOC
#include <Eigen/Core>
#include <Eigen/QR>

// SimpleSolver
#include "EqQpNs.h"


using namespace Eigen;

BOOST_AUTO_TEST_CASE(QRTest)
{
  MatrixXd A(2,3);
  A << 1., 0., 1.,
       0., 1., 1.;

  ColPivHouseholderQR<MatrixXd> qrAT(A.transpose());
  MatrixXd Q(qrAT.householderQ());
  MatrixXd Y(Q.topLeftCorner(3, 2));
  MatrixXd Z(Q.topRightCorner(3, 1));
  MatrixXd R(qrAT.matrixR().topRows(2));
  MatrixXd P(qrAT.colsPermutation());

  // check if the nullspace computation is OK (AZ = 0)
  MatrixXd AZ(A*Z);
  BOOST_CHECK(AZ.isZero(1e-8));

  // check if AY is full rank
  MatrixXd AY(A*Y);
  ColPivHouseholderQR<MatrixXd> qrAY(AY);
  BOOST_CHECK_EQUAL(qrAY.rank(), 2);

  // check AY = P R^T
  BOOST_CHECK_SMALL((AY - P*R.transpose()).norm(), 1e-8);

  // check (AY)^{-1} = (P R^T)^{-1} = R^{T^{-1}} P
  MatrixXd RTinvP(R.transpose().triangularView<Lower>()
    .solve(Matrix2d::Identity())*P);
  BOOST_CHECK_SMALL((qrAY.inverse() - RTinvP).norm(), 1e-8);
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

  double lagrangian = -3.5;
  BOOST_CHECK_SMALL(qp::eqQpLagrangian(G, c, A, b, eqQpNs.x(), eqQpNs.lambda())
    - lagrangian, 1e-8);
  BOOST_CHECK(qp::eqQpLagrangianGrad(G, c, A, eqQpNs.x(), eqQpNs.lambda())
    .isZero(1e-8));
}
