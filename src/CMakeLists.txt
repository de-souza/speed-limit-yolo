add_executable(${PROJECT_NAME} main.c)

set_target_properties(${PROJECT_NAME} PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

set(DARKNET_PATH ${PROJECT_SOURCE_DIR}/../darknet/ CACHE PATH "Path to the (compiled) darknet directory")

message(STATUS "Path to the (compiled) darknet directory: ${DARKNET_PATH}")
message(STATUS "Set this path with 'cmake -D DARKNET_PATH:PATH=/home/user/darknet'")

find_library(DARKNET_LIBRARY NAMES libdarknet.a PATHS ${DARKNET_PATH})

target_link_libraries(${PROJECT_NAME} PRIVATE ${DARKNET_LIBRARY} lept tesseract)
target_include_directories(${PROJECT_NAME} PRIVATE ${DARKNET_PATH}/include)
