cmake_minimum_required(VERSION 3.20)
project(cpp_pricer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---- Options -------------------------------------------------------------
option(CPP_PRICER_FP64 "Build in double precision (for gradient/parity checks)" OFF)
option(CPP_PRICER_ACCELERATE "Link Apple Accelerate BLAS for the layer GEMMs" ON)
option(CPP_PRICER_NATIVE "Tune for the host CPU (-mcpu=apple-m1 / -march=native)" ON)
option(PRICER_PROFILE "Emit the per-block learn-step timing split in bench_train" OFF)

# ---- Optimization flags --------------------------------------------------
add_library(pricer_flags INTERFACE)
target_compile_options(pricer_flags INTERFACE
  $<$<CONFIG:Release>:-O3 -ffast-math -funroll-loops -fno-math-errno>)
if(CPP_PRICER_NATIVE)
  # Apple clang on arm64 understands -mcpu=apple-m1; fall back to -march=native elsewhere.
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    target_compile_options(pricer_flags INTERFACE $<$<CONFIG:Release>:-mcpu=apple-m1>)
  else()
    target_compile_options(pricer_flags INTERFACE $<$<CONFIG:Release>:-march=native>)
  endif()
endif()
if(CPP_PRICER_FP64)
  target_compile_definitions(pricer_flags INTERFACE CPP_PRICER_FP64=1)
endif()
if(PRICER_PROFILE)
  target_compile_definitions(pricer_flags INTERFACE PRICER_PROFILE=1)
endif()

find_package(Threads REQUIRED)

# ---- Core library --------------------------------------------------------
add_library(pricer_core
  src/hhk_sim.cpp
  src/mlp.cpp
  src/kernel.cpp
  src/agent.cpp)
target_include_directories(pricer_core PUBLIC include)
target_link_libraries(pricer_core PUBLIC pricer_flags Threads::Threads)
if(CPP_PRICER_ACCELERATE)
  target_compile_definitions(pricer_core PUBLIC CPP_PRICER_ACCELERATE=1)
  target_link_libraries(pricer_core PUBLIC "-framework Accelerate")
endif()

# ---- Main CLI ------------------------------------------------------------
add_executable(price_swing apps/price_swing.cpp)
target_link_libraries(price_swing PRIVATE pricer_core)

add_executable(hedge_swing apps/hedge_swing.cpp)
target_link_libraries(hedge_swing PRIVATE pricer_core)

# ---- Training micro-benchmark -------------------------------------------
add_executable(bench_train bench/bench_train.cpp)
target_link_libraries(bench_train PRIVATE pricer_core)

# ---- Tests ---------------------------------------------------------------
enable_testing()
set(PRICER_DATA "${CMAKE_CURRENT_SOURCE_DIR}/data")

add_executable(test_parity tests/test_parity.cpp)
target_link_libraries(test_parity PRIVATE pricer_core)
add_test(NAME parity COMMAND test_parity ${PRICER_DATA})

add_executable(test_sim tests/test_sim.cpp)
target_link_libraries(test_sim PRIVATE pricer_core)
add_test(NAME sim COMMAND test_sim)

add_executable(test_grad tests/test_grad.cpp)
target_link_libraries(test_grad PRIVATE pricer_core)
add_test(NAME grad COMMAND test_grad ${PRICER_DATA})

# Fixture-free gradcheck of the runtime-depth nets (O2/H-N1/H-N2). No data needed.
add_executable(test_grad_depth tests/test_grad_depth.cpp)
target_link_libraries(test_grad_depth PRIVATE pricer_core)
add_test(NAME grad_depth COMMAND test_grad_depth)
