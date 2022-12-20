find_package(benchmark QUIET)

if (NOT benchmark_FOUND)
    include(FetchContent)
    FetchContent_Declare(
            benchmark
            GIT_REPOSITORY https://github.com/google/benchmark.git
            GIT_TAG v1.7.0
    )
    FetchContent_MakeAvailable(benchmark)
endif()