set(SOURCES EqQpNs.cpp QpNs.cpp QpStartTypeI.cpp)
set(HEADERS EqQpNs.h QpNs.h Qp.h QpStartTypeI.h)

search_for_eigen()

# -Wno-deprecated-declarations should be remove in a next eigen version
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -std=c++0x -pedantic -Wno-deprecated-declarations")
include_directories(${EIGEN3_INCLUDE_DIRS})

# !!! SimpleSolver is an empty library !!!
# !!! Its aim is to test header file and have an IDE integration !!!
# !!! Don't link against it !!!
add_library(${CMAKE_PROJECT_NAME} ${SOURCES})

set(INSTALL_PATH include/${CMAKE_PROJECT_NAME})
install(FILES ${HEADERS} DESTINATION ${INSTALL_PATH})
