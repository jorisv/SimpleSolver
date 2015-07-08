cmake_minimum_required(VERSION 2.8.9)
project(EQP CXX)

option(UNIT_TESTS "Generate unit tests." ON)

add_subdirectory(src)

if(${UNIT_TESTS})
  enable_testing()
  add_subdirectory(tests)
endif()
