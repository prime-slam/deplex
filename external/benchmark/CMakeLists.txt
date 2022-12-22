find_package(benchmark QUIET)

include(FetchContent)
FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.7.1
)
FetchContent_MakeAvailable(benchmark)
