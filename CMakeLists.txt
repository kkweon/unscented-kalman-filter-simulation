project(UnscentedKF)

cmake_minimum_required (VERSION 3.5)
add_definitions(-std=c++14)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")
set(sources src/ukf.cpp src/tools.cpp)
set(main src/main.cpp)

include_directories(src)
file(GLOB test_sources "tests/*.cpp")

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin") 

include_directories(/usr/local/include)
include_directories(/usr/local/opt/openssl/include)
link_directories(/usr/local/lib)
link_directories(/usr/local/opt/openssl/lib)
link_directories(/usr/local/Cellar/libuv/1.11.0/lib)

endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin") 

add_executable(UnscentedKF ${main} ${sources})
target_link_libraries(UnscentedKF z ssl uv uWS)

find_package(GTest)
add_executable(test_main ${test_sources} ${sources})
target_link_libraries(test_main GTest::GTest GTest::Main)

enable_testing()
ADD_TEST(ctest_build_test_code "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target test_main)
ADD_TEST(ctest_run_test_code test_main)
SET_TESTS_PROPERTIES(ctest_run_test_code PROPERTIES DEPENDS ctest_build_test_code)

