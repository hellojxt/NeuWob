cmake_minimum_required(VERSION 3.1)
project(main LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
include(CheckLanguage)
check_language(CUDA)
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
cmake_policy(SET CMP0104 OLD) # for CUDA_ARCHITECTURES
cmake_policy(SET CMP0072 NEW)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr -std=c++17")

# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/deps/eigen/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/deps/lbvh/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/deps/json)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/deps/tinyobjloader)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/deps/fmt)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_subdirectory(deps/imgui)

set(NEUWOB_SRCS
    src/scene.cu
)

add_library(neuwob STATIC ${NEUWOB_SRCS})
target_compile_definitions(neuwob PRIVATE DATA_DIR="${CMAKE_SOURCE_DIR}/data/")

add_library(ui STATIC src/ui.cu)
target_compile_definitions(ui PRIVATE IMGUI_CONFIG_FILE="${CMAKE_SOURCE_DIR}/imgui.ini")
target_link_libraries(ui PUBLIC imgui)


file( GLOB APP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/apps/*.cu )
foreach( SRCFILES ${APP_SOURCES} )
    file(RELATIVE_PATH EXECUTABLE_NAME ${CMAKE_CURRENT_SOURCE_DIR} ${SRCFILES})
    #Remove ".cu" and "apps/" to get the name of the executable.
    string( REPLACE ".cu" "" EXECUTABLE_NAME ${EXECUTABLE_NAME} )
    string( REPLACE "apps/" "" EXECUTABLE_NAME ${EXECUTABLE_NAME} )
    add_executable( ${EXECUTABLE_NAME} ${SRCFILES} )
    target_link_libraries( ${EXECUTABLE_NAME} PUBLIC neuwob ui)
endforeach()