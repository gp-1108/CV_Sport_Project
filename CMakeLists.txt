cmake_minimum_required(VERSION 3.0)

project(YOLO_INFERENCE)

find_package(OpenCV 4 REQUIRED)

add_executable(model_inference src/model_inference.cpp src/yolo.cpp src/postProcessing.cpp src/playerAssignement.cpp)
target_link_libraries(model_inference ${OpenCV_LIBS})