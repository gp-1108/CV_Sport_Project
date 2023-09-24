cmake_minimum_required(VERSION 3.0)

project(YOLO_INFERENCE)

find_package(OpenCV 4 REQUIRED)

add_executable(main src/main.cpp
                src/yolo.cpp
                src/postProcessingYolo.cpp 
                src/playerAssignement.cpp
                src/sceneAnalyzer.cpp
                src/fieldSegmentation.cpp
                src/performances.cpp)

target_link_libraries(main ${OpenCV_LIBS})