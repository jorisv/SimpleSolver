find_package(PkgConfig)

# find eigen3
pkg_check_modules(EIGEN3 REQUIRED eigen3)
find_package(Boost REQUIRED COMPONENTS unit_test_framework)

# -Wno-deprecated-declarations should be remove in a next eigen version
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -std=c++0x -pedantic -Wno-deprecated-declarations")
include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories(${Boost_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})

macro(addUnitTest name)
  if(${UNIT_TESTS})
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} ${Boost_LIBRARIES})
    add_test(${name}Unit ${name})
  endif()
endmacro(addUnitTest)

addUnitTest("eq_qp_ns_test")
addUnitTest("qp_ns_test")
