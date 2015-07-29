set(BOOST_COMPONENTS unit_test_framework)
search_for_eigen()
search_for_boost()

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

addUnitTest("StdConstraintsTest")
addUnitTest("EqQpNsTest")
addUnitTest("QpNsTest")
addUnitTest("LpPrimalTest")
