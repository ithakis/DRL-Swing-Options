cmake_minimum_required(VERSION 3.20)
project(grid_dp_pricer CXX)

# ---- Build type / standard ----------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Options ------------------------------------------------------------------
option(GRID_DP_FP64 "Double-precision reference/parity build (OFF = FP32 fast path)" ON)
option(GRID_DP_ACCELERATE "Use Apple Accelerate BLAS" ON)

# ---- Header-only library target ----------------------------------------------
add_library(grid_dp INTERFACE)
target_include_directories(grid_dp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(grid_dp INTERFACE cxx_std_20)

if(GRID_DP_FP64)
  target_compile_definitions(grid_dp INTERFACE GRID_DP_FP64=1)
else()
  target_compile_definitions(grid_dp INTERFACE GRID_DP_FP64=0)
endif()

if(GRID_DP_ACCELERATE AND APPLE)
  # ACCELERATE_NEW_LAPACK selects the modern (non-deprecated) CBLAS headers; default LP64.
  target_compile_definitions(grid_dp INTERFACE GRID_DP_ACCELERATE=1 ACCELERATE_NEW_LAPACK=1)
  target_link_libraries(grid_dp INTERFACE "-framework Accelerate")
else()
  target_compile_definitions(grid_dp INTERFACE GRID_DP_ACCELERATE=0)
endif()

# ---- Optimization flags -------------------------------------------------------
include(CheckCXXCompilerFlag)
set(GDP_OPT_FLAGS -O3)
check_cxx_compiler_flag("-mcpu=apple-m1" GDP_HAS_MCPU_M1)
if(GDP_HAS_MCPU_M1)
  list(APPEND GDP_OPT_FLAGS -mcpu=apple-m1)
endif()
# -ffast-math is only safe for the FP32 fast path; the FP64 reference build must
# keep strict IEEE arithmetic so the 1e-10 parity bar holds.
if(NOT GRID_DP_FP64)
  check_cxx_compiler_flag("-ffast-math" GDP_HAS_FFAST)
  if(GDP_HAS_FFAST)
    list(APPEND GDP_OPT_FLAGS -ffast-math)
  endif()
endif()
target_compile_options(grid_dp INTERFACE ${GDP_OPT_FLAGS})

set(GDP_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data)

# ---- Helpers ------------------------------------------------------------------
function(gdp_add_app name)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/apps/${name}.cpp)
    add_executable(${name} apps/${name}.cpp)
    target_link_libraries(${name} PRIVATE grid_dp)
  endif()
endfunction()

function(gdp_add_test name)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/${name}.cpp)
    add_executable(${name} tests/${name}.cpp)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(${name} PRIVATE grid_dp)
    add_test(NAME ${name} COMMAND ${name} ${GDP_DATA_DIR})
  endif()
endfunction()

# ---- Targets (added as their source files appear) -----------------------------
enable_testing()
gdp_add_test(test_gauss)
gdp_add_test(test_spline)
gdp_add_test(test_kernel_moments)
gdp_add_test(test_kernel_parity)
gdp_add_test(test_inner_control)
gdp_add_test(test_dp_limits)

gdp_add_app(price_dp)
gdp_add_app(sweep_dp)
