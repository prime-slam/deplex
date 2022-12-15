#####################################
# process_cloud.cpp
#####################################
add_executable(process_cloud process_cloud.cpp)
target_compile_features(process_cloud PRIVATE cxx_std_17)
target_link_libraries(process_cloud PRIVATE ${PROJECT_NAME})
