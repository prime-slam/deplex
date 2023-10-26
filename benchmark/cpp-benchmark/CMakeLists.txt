 target_compile_options(deplex PRIVATE -O3)

 #####################################
 # process_cloud.cpp
 #####################################
 add_executable(benchmark_process_cloud benchmark_process_cloud.cpp)
 target_compile_features(benchmark_process_cloud PRIVATE cxx_std_17)
 target_link_libraries(benchmark_process_cloud PRIVATE ${PROJECT_NAME})

 #####################################
 # process_sequence.cpp
 #####################################
 add_executable(benchmark_process_sequence benchmark_process_sequence.cpp)
 target_compile_features(benchmark_process_sequence PRIVATE cxx_std_17)
 target_link_libraries(benchmark_process_sequence PRIVATE ${PROJECT_NAME})