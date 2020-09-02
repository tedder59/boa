cmake_minimum_required(VERSION 3.12)

project(boa CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR})

FILE(GLOB WINOGRAD_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/winograd/*.cpp)
set(BOA_SRCS ${WINOGRAD_SRCS})

add_library(boa SHARED ${BOA_SRCS})

add_subdirectory(test)