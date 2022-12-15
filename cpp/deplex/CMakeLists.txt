#####################################
# Define Target Information
#####################################
set(TARGET_NAME deplex)

set(TARGET_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(TARGET_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(SRC_FILES
    ${TARGET_SOURCE_DIR}/deplex/config.cpp
    ${TARGET_SOURCE_DIR}/deplex/cell_segment_stat.cpp
    ${TARGET_SOURCE_DIR}/deplex/cell_segment.cpp
    ${TARGET_SOURCE_DIR}/deplex/normals_histogram.cpp
    ${TARGET_SOURCE_DIR}/deplex/plane_extractor.cpp
    ${TARGET_SOURCE_DIR}/deplex/utils/image_to_cloud.cpp
    ${TARGET_SOURCE_DIR}/deplex/utils/eigen_io.cpp
)
#####################################
# Required packages
#####################################
find_package(Eigen3 REQUIRED)

#####################################
# Optional packages
#####################################
# OpenCV is used for IO-operations with images
find_package(OpenCV QUIET)

#####################################
# Define Target
#####################################
add_library(${TARGET_NAME} SHARED ${SRC_FILES})

#####################################
# Define Target Properties
#####################################
target_compile_features(${TARGET_NAME} PUBLIC cxx_std_14)

if (DEBUG_DEPLEX)
    target_compile_definitions(${TARGET_NAME} PRIVATE DEBUG_DEPLEX)
endif()

target_include_directories(
    ${TARGET_NAME}
    SYSTEM
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(${TARGET_NAME} PUBLIC Eigen3::Eigen)

if (OpenCV_FOUND)
    target_link_libraries(${TARGET_NAME} PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(${TARGET_NAME} PUBLIC HAVE_OPENCV)
endif()