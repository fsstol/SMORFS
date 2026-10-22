cmake_minimum_required(VERSION 3.20)
project(smorfs CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

# The library the Python package loads, in the repository root: semi_markov.so, or semi_markov.dll on Windows
# (compile.sh builds the same file on macOS and Linux)
add_library(semi_markov SHARED src/src_lib/library.cpp)
set_target_properties(semi_markov PROPERTIES PREFIX ""
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR} LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
foreach(config ${CMAKE_CONFIGURATION_TYPES})  # multi-configuration generators (Visual Studio)
    string(TOUPPER ${config} CONFIG)
    set_target_properties(semi_markov PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_${CONFIG} ${CMAKE_SOURCE_DIR} LIBRARY_OUTPUT_DIRECTORY_${CONFIG} ${CMAKE_SOURCE_DIR})
endforeach()
if(NOT WIN32)
    set_target_properties(semi_markov PROPERTIES SUFFIX ".so")
endif()
if(MSVC)
    target_compile_options(semi_markov PRIVATE /bigobj)  # one translation unit with Eigen
endif()
if(MINGW)
    target_link_options(semi_markov PRIVATE -static)  # no MinGW runtime DLLs needed next to the library
endif()

# Optimize for the building machine's processor, as compile.sh does (GCC and clang)
option(SMORFS_NATIVE "Compile for this machine's processor (-march=native)" OFF)
if(SMORFS_NATIVE AND NOT MSVC)
    target_compile_options(semi_markov PRIVATE -march=native)
endif()

# The Fourier series' matrix product through a BLAS: Accelerate on macOS by default (run on one thread, fourier.cpp)
option(SMORFS_USE_BLAS "Link a BLAS for the Fourier series' matrix product" ${APPLE})
if(SMORFS_USE_BLAS)
    find_package(BLAS REQUIRED)
    target_compile_definitions(semi_markov PRIVATE SMORFS_USE_BLAS)
    target_link_libraries(semi_markov PRIVATE BLAS::BLAS)
endif()

# The timing test program (src/src_lib/test_main.cpp)
add_executable(semi_markov_calibration src/src_lib/test_main.cpp)
if(MSVC)
    target_compile_options(semi_markov_calibration PRIVATE /bigobj)
endif()
if(MINGW)
    target_link_options(semi_markov_calibration PRIVATE -static)
endif()
