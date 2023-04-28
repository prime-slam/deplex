#####################################
# process_cloud.cpp
#####################################
add_executable(process_cloud process_cloud.cpp)
target_compile_features(process_cloud PRIVATE cxx_std_17)
target_link_libraries(process_cloud PRIVATE ${PROJECT_NAME})

#####################################
# process_sequence.cpp
#####################################
add_executable(process_sequence process_sequence.cpp)
target_compile_features(process_sequence PRIVATE cxx_std_17)
target_link_libraries(process_sequence PRIVATE ${PROJECT_NAME})