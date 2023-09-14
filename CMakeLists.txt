project(dfThermo)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(${PROJECT_NAME} dfThermo.cpp main.cpp)

add_compile_options(-g)