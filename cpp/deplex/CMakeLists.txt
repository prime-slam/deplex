#####################################
# Define Target Information
#####################################
set(TARGET_NAME deplex)
set(TARGET_INCLUDE_FOLDER ${CMAKE_CURRENT_SOURCE_DIR})

# TODO: add sources from child cmake scripts
set(TARGET_SOURCES
        algorithm/cell_segment.cpp
        algorithm/config.cpp
        algorithm/histogram.cpp
        algorithm/plane_extractor.cpp
        utils/io/image_to_cloud.cpp)

#####################################
# Required packages
#####################################
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

#####################################
# Define Target
#####################################
add_library(${TARGET_NAME} SHARED ${TARGET_SOURCES})

#####################################
# Define Target Properties
#####################################
target_include_directories(
        ${TARGET_NAME}
        SYSTEM
        PUBLIC
        $<BUILD_INTERFACE:${TARGET_INCLUDE_FOLDER}>
)

target_link_libraries(${TARGET_NAME} PUBLIC ${OpenCV_LIBS})
target_link_libraries(${TARGET_NAME} PUBLIC Eigen3::Eigen)
