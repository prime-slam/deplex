add_executable(deplex-benchmark
    main.cpp
    benchmark_extractor.cpp
)

find_package(benchmark REQUIRED)

target_link_libraries(deplex-benchmark PRIVATE deplex)
target_link_libraries(deplex-benchmark PRIVATE benchmark::benchmark_main)