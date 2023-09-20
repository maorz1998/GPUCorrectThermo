cmake_minimum_required(VERSION 3.10)

project(dfThermo LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} dfThermo.cpp dfThermo.cu main.cpp)

target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})

add_compile_options(-g)