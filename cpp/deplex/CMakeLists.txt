#####################################
# Define Target Information
#####################################
set(TARGET_NAME deplex)
set(TARGET_INCLUDE_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/include)


set(TARGET_SOURCES
        include/deplex/plane_extraction.h
        include/deplex/histogram.hpp
        include/deplex/planeseg.hpp
        include/deplex/config.hpp
        include/deplex/image_reader.hpp
        src/plane_extraction.cpp)

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

set_target_properties(${TARGET_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${DEPLEX_LIB_DIR}")