cmake_minimum_required(VERSION 3.12)
project(fastpy LANGUAGES CXX CUDA)

set(PYBIND11_CPP_STANDARD -std=c++17)

include(FetchContent)
FetchContent_Declare(pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

find_package(CUDAToolkit REQUIRED)

# Add include directories for the source files
include_directories(${CMAKE_SOURCE_DIR})

add_library(fastpycu STATIC
  src/functions/matmul/matmul.cu
  src/functions/add/add.cu
  src/fastpy/cuda.cu
)

set_target_properties(fastpycu PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_VISIBILITY_PRESET "hidden"
)

add_library(fastpy MODULE
  src/fastpy.pybind.cpp
  src/functions/matmul/utils.cpp
  src/functions/add/utils.cpp
  src/fastpy/array.cpp
)

set_target_properties(fastpy PROPERTIES
  CXX_VISIBILITY_PRESET "hidden"
  INTERPROCEDURAL_OPTIMIZATION TRUE
  PREFIX "${PYTHON_MODULE_PREFIX}"
  SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

# Link cudart explicitly (both targets)
target_link_libraries(fastpycu PRIVATE CUDA::cudart CUDA::cublas CUDA::curand)
target_link_libraries(fastpy   PRIVATE CUDA::cudart CUDA::cublas CUDA::curand pybind11::module fastpycu)

# ---- Robust CUDA DLL copy on Windows (no-op if none found) ----
if (WIN32)
  # Prefer CUDAToolkit variable; fall back to CUDA_PATH env
  set(_cuda_bin "${CUDAToolkit_BIN_DIR}")
  if (NOT _cuda_bin AND DEFINED ENV{CUDA_PATH})
    set(_cuda_bin "$ENV{CUDA_PATH}/bin")
  endif()

  set(_cuda_dlls)
  if (EXISTS "${_cuda_bin}")
    # Match both CUDA 11/12 style names
    file(GLOB _cuda_dlls LIST_DIRECTORIES FALSE
      "${_cuda_bin}/cublas64_*.dll"
      "${_cuda_bin}/cublasLt64_*.dll"
      "${_cuda_bin}/curand64_*.dll"
      "${_cuda_bin}/cudart64_*.dll"
    )
  endif()

  if (_cuda_dlls)
    add_custom_command(TARGET fastpy POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              ${_cuda_dlls} $<TARGET_FILE_DIR:fastpy>
      COMMENT "Copying CUDA runtime DLLs to module dir"
    )
  else()
    message(WARNING
      "Could not find CUDA runtime DLLs in '${_cuda_bin}'. "
      "Skipping copy; you may need to add that directory via os.add_dll_directory at runtime.")
  endif()
endif()
