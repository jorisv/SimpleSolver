#SimpleSolver

SimpleSolver aim to provide a set of solver writing with the Eigen3 library.

The purpose of this library is to be educational and not about providing efficient solvers.

## Installing

#### Manual

##### Dependencies

 * [Git]()
 * [CMake]() >= 2.8
 * [pkg-config]()
 * [g++]() >= 4.8
 * [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) >= 3.2
 * [Boost](http://www.boost.org/doc/libs/1_58_0/more/getting_started/unix-variants.html) (for unit test only)

##### Building

```sh
git clone --recursive https://github.com/jorisv/SimpleSolver
mkdir SimpleSolver/_build
cd SimpleSolver/_build
cmake [options] ..
make && make intall
```

Where the main options are:

 * `-DCMAKE_BUIlD_TYPE=Release` Build in Release mode
 * `-DCMAKE_INSTALL_PREFIX=some/path/to/install` default is `/usr/local`
 * `-DUNIT_TESTS=ON` Build unit tests.

## Pulling git subtree

To update cmake directory with their upstream git repository:

```sh
git fetch git://github.com/jrl-umi3218/jrl-cmakemodules.git master
git subtree pull --prefix cmake git://github.com/jrl-umi3218/jrl-cmakemodules.git master --squash
```
