cmake_minimum_required(VERSION 3.16)
project(xfeat_lightglue_trt)

# Set C++ standard
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

set(TENSORRT_ROOT "/usr/src/tensorrt" CACHE PATH
    "Root directory of TensorRT (contains include/ and lib/)")
set(TORCH_PATH "/usr/local/lib/python3.12/dist-packages/torch" CACHE PATH 
    "Root directory of libtorch (contains share/cmake/Torch)")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

if(EXISTS "${TORCH_PATH}/share/cmake/Torch/TorchConfig.cmake")
  find_package(Torch REQUIRED PATHS "${TORCH_PATH}/share/cmake/Torch" NO_DEFAULT_PATH)
else()
  find_package(Torch REQUIRED)
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# 手动给出TensorRT的安装路径
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)

find_library(TENSORRT_LIBRARY_INFER nvinfer
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)

find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)

find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)

set(TENSORRT_LIBRARIES ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN} ${TENSORRT_LIBRARY_ONNXPARSER})

include_directories(
  include
  ${TENSORRT_INCLUDE_DIR}
  ${TORCH_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# xfeat-lightglue 模块编译
add_library(xfeat_libs SHARED
    src/xfeat.cpp
    include/xfeat.h 
    include/InterpolateSparse2D.h   
    include/utils.h
)

add_library(lightglue_libs SHARED
    src/lightglue.cpp
    include/lightglue.h 
    include/InterpolateSparse2D.h   
    include/utils.h
)

target_link_libraries(xfeat_libs 
    CUDA::cudart
    ${OpenCV_LIBS} 
    ${TORCH_LIBRARIES} 
    ${TENSORRT_LIBRARIES} 
    yaml-cpp)

target_link_libraries(lightglue_libs
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
    ${TENSORRT_LIBRARIES}
    CUDA::cudart
    yaml-cpp
)

add_executable(match_test src/main.cpp)
target_link_libraries(match_test xfeat_libs lightglue_libs ${TORCH_LIBRARIES})

add_library(localizer_node SHARED
    src/localizer_node.cpp
)
target_include_directories(localizer_node PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(localizer_node
    xfeat_libs
    lightglue_libs
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
)
ament_target_dependencies(localizer_node
    rclcpp
    rclcpp_components
    sensor_msgs
    geometry_msgs
    std_msgs
    cv_bridge
    image_transport
    tf2
    tf2_geometry_msgs
)

rclcpp_components_register_nodes(localizer_node
    "xfeat_localizer::LocalizerNode"
)

add_executable(localizer_main src/main_ros.cpp)
target_link_libraries(localizer_main localizer_node)
ament_target_dependencies(localizer_main rclcpp)

install(TARGETS
    xfeat_libs
    lightglue_libs
    match_test
    localizer_node
    localizer_main
    DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/
    DESTINATION include
)
install(DIRECTORY config launch reference weights
    DESTINATION share/${PROJECT_NAME}
    OPTIONAL   # weights/ may be empty before engine generation
)

ament_package()