 #####################################
 # process_cloud.cpp
 #####################################
 add_executable(process_cloud process_cloud.cpp)
 target_compile_features(process_cloud PRIVATE cxx_std_17)
 target_link_libraries(process_cloud PRIVATE ${PROJECT_NAME})

 find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

 target_link_libraries(process_cloud PRIVATE Python3::Python)

 target_compile_options(process_cloud PRIVATE -std=c++14 -Ofast)

 #####################################
 # process_sequence.cpp
 #####################################
 add_executable(process_sequence process_sequence.cpp)
 target_compile_features(process_sequence PRIVATE cxx_std_17)
 target_link_libraries(process_sequence PRIVATE ${PROJECT_NAME})

 target_compile_options(process_cloud PRIVATE -std=c++14 -Ofast)
