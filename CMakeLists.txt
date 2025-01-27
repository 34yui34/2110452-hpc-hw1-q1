# Minimum required CMake version
cmake_minimum_required(VERSION 3.10)

# Project name
project(benchmark_ex1)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(benchmark REQUIRED)

# Add source files to the target
add_executable(benchmark_ex1
    ${CMAKE_SOURCE_DIR}/main.cpp
)

target_compile_options(benchmark_ex1 PRIVATE -mavx2)

# Optionally, add compile definitions or flags
# target_compile_definitions(benchmark_ex1 PRIVATE -DENABLE_TRACE_DISPLAY_LOG)

# Optionally, link to other libraries (if needed)
target_link_libraries(benchmark_ex1 benchmark::benchmark)

