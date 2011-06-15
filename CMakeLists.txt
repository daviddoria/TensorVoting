cmake_minimum_required(VERSION 2.6)

PROJECT(TensorVoting)

FIND_PACKAGE(ITK REQUIRED)
INCLUDE(${ITK_USE_FILE})

FIND_PACKAGE(VXL REQUIRED)
INCLUDE(${VXL_CMAKE_DIR}/UseVXL.cmake)
INCLUDE_DIRECTORIES(${VXL_RPL_INCLUDE_DIR})
# You must first enable rgtl, then rtvl

#### Executables ####
ADD_EXECUTABLE(TensorVoting TensorVoting.cxx)
TARGET_LINK_LIBRARIES(TensorVoting rtvl rgtl vnl_algo vnl ${ITK_LIBRARIES})
