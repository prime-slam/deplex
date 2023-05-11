enable_testing()

#####################################
# Set path to common test data samples
#####################################
set(DATA_DIR ${CMAKE_SOURCE_DIR}/data)

set(ICL_INTRINSICS "${DATA_DIR}/configs/ICL_living_room.K")
set(ICL_CONFIG "${DATA_DIR}/configs/ICL_living_room.ini")
set(ICL_SAMPLE_IMAGE "${DATA_DIR}/icl_nuim/0.png")
set(TUM_INTRINSICS "${DATA_DIR}/configs/TUM_fr3_long_val.K")
set(TUM_CONFIG "${DATA_DIR}/configs/TUM_fr3_long_val.ini")
set(TUM_SAMPLE_IMAGE "${DATA_DIR}/tum/1341848230.910894.png")
set(TUM_SAMPLE_IMAGE_POINTS "${DATA_DIR}/tum/points_1341848230.910894.csv")

#####################################
# Configure globals.hpp
#####################################
configure_file(globals.hpp.in globals.hpp)

#####################################
# Test executable
#####################################
add_executable(unit-tests
        main.cpp
        test_plane_extractor.cpp
        )

target_include_directories(unit-tests SYSTEM PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(unit-tests PRIVATE GTest::gtest_main)
target_link_libraries(unit-tests PRIVATE deplex)

#####################################
# Run tests target
#####################################
add_custom_target(run-unit-tests
        COMMAND unit-tests
        DEPENDS unit-tests
        )

include(GoogleTest)
gtest_discover_tests(unit-tests)