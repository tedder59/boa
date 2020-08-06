find_package(GTest REQUIRED)

include_directories(${GTEST_INCLUDE_DIRS})

FILE(GLOB TEST_SRCS "*.cpp")

add_executable(test ${TEST_SRCS})
target_link_libraries(test ${GTEST_BOTH_LIBRARIES} pthread)