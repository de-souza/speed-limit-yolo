add_executable(${PROJECT_NAME} main.c)

set(DARKNET_PATH ${PROJECT_SOURCE_DIR}/../darknet/ CACHE PATH "Path to the (compiled) darknet directory")

message(STATUS "Path to the (compiled) darknet directory: ${DARKNET_PATH}")
message(STATUS "Set this path with cmake -D DARKNET_PATH:PATH=/home/user/darknet")

target_compile_definitions(${PROJECT_NAME} PRIVATE DARKNET_PATH="${DARKNET_PATH}")

target_compile_features(${PROJECT_NAME} PRIVATE c_std_11)
set_target_properties(${PROJECT_NAME} PROPERTIES C_EXTENSIONS OFF)

find_library(DARKNET_LIBRARY darknet PATHS ${DARKNET_PATH})
target_link_libraries(${PROJECT_NAME} PRIVATE ${DARKNET_LIBRARY})
target_include_directories(${PROJECT_NAME} PRIVATE ${DARKNET_PATH}/include)
