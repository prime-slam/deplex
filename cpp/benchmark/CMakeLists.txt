set(DATA_DIR ${CMAKE_SOURCE_DIR}/data)

set(TUM_INTRINSICS "${DATA_DIR}/configs/TUM_fr3_long_val.K")
set(TUM_CONFIG "${DATA_DIR}/configs/TUM_fr3_long_val.ini")
set(TUM_SAMPLE_IMAGE "${DATA_DIR}/tum/1341848230.910894.png")
set(TUM_SAMPLE_IMAGE_POINTS "${DATA_DIR}/tum/points_1341848230.910894.csv")

configure_file(globals.hpp.in globals.hpp)

add_executable(deplex-benchmark
        benchmark_extractor.cpp
        )

target_include_directories(deplex-benchmark SYSTEM PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(deplex-benchmark PRIVATE deplex)
target_link_libraries(deplex-benchmark PRIVATE benchmark::benchmark)