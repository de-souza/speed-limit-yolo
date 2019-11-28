cmake_minimum_required(VERSION 3.8..3.16)

project(speed-limit-yolo
    VERSION 0.1
    DESCRIPTION "Real time speed limit sign recognition using YOLOv3"
    LANGUAGES C
)

add_subdirectory(cfg)
add_subdirectory(data)
add_subdirectory(src)
