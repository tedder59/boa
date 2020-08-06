cmake_minimum_required(VERSION 3.12)

project(boa CXX)

include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR})

# set(BOA_SRCS ${ARRAY_SRCS})

# add_library(boa SHARED ${BOA_SRCS})

add_subdirectory(test)