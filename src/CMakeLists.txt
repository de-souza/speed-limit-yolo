add_executable(${PROJECT_NAME} main.c image.c ocr.c utils.c video.c)
add_custom_target(headers SOURCES image.h ocr.h utils.h video.h)

set_target_properties(${PROJECT_NAME} PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

list(APPEND LINK_LIBS m pthread stdc++)

set(DARKNET_PATH ${PROJECT_SOURCE_DIR}/../darknet/ CACHE PATH "Path to the darknet library")
message(STATUS "Using: \"-DDARKNET_PATH:PATH=${DARKNET_PATH}\"")
find_library(DARKNET_LIBRARY darknet PATHS ${DARKNET_PATH})
list(APPEND LINK_LIBS ${DARKNET_LIBRARY})
list(APPEND INCLUDE_DIRS ${DARKNET_PATH}/include)
target_compile_definitions(${PROJECT_NAME} PRIVATE DARKNET_PATH="${DARKNET_PATH}")

list(APPEND LINK_LIBS opencv_core opencv_highgui opencv_imgcodecs opencv_videoio)

list(APPEND LINK_LIBS lept tesseract)

option(CLIENTIO "Enable linking to the clientio library")
message(STATUS "Using: \"-DCLIENTIO:BOOL=${CLIENTIO}\"")
if(CLIENTIO)
    set(CLIENTIO_PATH ${PROJECT_SOURCE_DIR}/../car-dashboard/client/clientio/ CACHE PATH "Path to the clientio library")
    message(STATUS "Using: \"-DCLIENTIO_PATH:PATH=${CLIENTIO_PATH}\"")
    find_library(CLIENTIO_LIBRARY clientio PATHS ${CLIENTIO_PATH})
    list(APPEND LINK_LIBS ${CLIENTIO_LIBRARY})
    list(APPEND INCLUDE_DIRS ${CLIENTIO_PATH})
    target_compile_definitions(${PROJECT_NAME} PRIVATE CLIENTIO=1)
else()
    target_compile_definitions(${PROJECT_NAME} PRIVATE CLIENTIO=0)
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE ${LINK_LIBS})
target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIRS})
target_compile_options(${PROJECT_NAME} PRIVATE -O3)
