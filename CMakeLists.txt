# Require CMake 3.15+ (matching scikit-build-core)
cmake_minimum_required(VERSION 3.15...4.0)

# Scikit-build-core sets these values
project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find pybind11 and Python
find_package(pybind11 CONFIG REQUIRED)

# ===== Fetch GGML Library =====
include(FetchContent)
FetchContent_Declare(
  ggml
  GIT_REPOSITORY https://github.com/ggerganov/ggml.git
  GIT_TAG        master
)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_STATIC ON CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(ggml)

# ===== Create Python Module =====
python_add_library(_core MODULE
  src/main.cpp
  src/tokenizer.cpp
  src/crf.cpp
  src/khmer-segmenter.cpp
  WITH_SOABI
)

target_include_directories(_core PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(_core PRIVATE
  pybind11::headers
  ggml
)

# Pass version info
target_compile_definitions(_core PRIVATE VERSION_INFO="${PROJECT_VERSION}")

# Install to the kns package directory
install(TARGETS _core DESTINATION kns)
