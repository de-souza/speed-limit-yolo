add_executable(${PROJECT_NAME} main.c image.c ocr.c utils.c video.c)

set_target_properties(${PROJECT_NAME} PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

set(DARKNET_PATH ${PROJECT_SOURCE_DIR}/../darknet/ CACHE PATH "Path to the (compiled) darknet directory")
message(STATUS "Path to the (compiled) darknet directory: ${DARKNET_PATH}")
message(STATUS "Set this path with 'cmake -D DARKNET_PATH:PATH=/home/user/darknet'")

find_library(DARKNET_LIBRARY darknet PATHS ${DARKNET_PATH})
set(OPENCV_LIBS opencv_core;opencv_imgcodecs;opencv_highgui;opencv_videoio)

target_link_libraries(${PROJECT_NAME} PRIVATE ${DARKNET_LIBRARY} ${OPENCV_LIBS} stdc++ m pthread lept tesseract)
target_include_directories(${PROJECT_NAME} PRIVATE ${DARKNET_PATH}/include)
target_compile_definitions(${PROJECT_NAME} PRIVATE DARKNET_PATH="${DARKNET_PATH}")
target_compile_options(${PROJECT_NAME} PRIVATE -O3)

add_custom_target(headers SOURCES image.h ocr.h utils.h video.h)
