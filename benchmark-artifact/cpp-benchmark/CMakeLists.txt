 #####################################
 # process_cloud.cpp
 #####################################
 add_executable(benchmark_process_cloud process_cloud.cpp)
 target_compile_features(benchmark_process_cloud PRIVATE cxx_std_17)
 target_link_libraries(benchmark_process_cloud PRIVATE ${PROJECT_NAME})

 target_compile_options(benchmark_process_cloud PRIVATE -std=c++14 -Ofast)

 #####################################
 # process_sequence.cpp
 #####################################
 add_executable(benchmark_process_sequence process_sequence.cpp)
 target_compile_features(benchmark_process_sequence PRIVATE cxx_std_17)
 target_link_libraries(benchmark_process_sequence PRIVATE ${PROJECT_NAME})

 target_compile_options(benchmark_process_sequence PRIVATE -std=c++14 -Ofast)
