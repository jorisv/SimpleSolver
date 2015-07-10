cmake_minimum_required(VERSION 2.8.9)

include(cmake/base.cmake)
include(cmake/boost.cmake)
include(cmake/eigen.cmake)

set(PROJECT_NAME SimpleSolver)
set(PROJECT_DESCRIPTION "...")
set(PROJECT_URL "https://github.com/jorisv/SimpleSolver")

# Disable -Werror on Unix for now.
set(CXX_DISABLE_WERROR True)

setup_project()

option(UNIT_TESTS "Generate unit tests." ON)

add_subdirectory(src)

if(${UNIT_TESTS})
  add_subdirectory(tests)
endif()

setup_project_finalize()
