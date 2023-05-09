set(DSYEV_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dsyevh3.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dsyevc3.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dsyevq3.c
)

add_library(dsyev SHARED ${DSYEV_SRC})

target_include_directories(
    dsyev
    SYSTEM
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set_target_properties(dsyev PROPERTIES LINK_FLAGS
            "-undefined dynamic_lookup")
endif()

target_compile_options(dsyev PRIVATE -fPIC -Wall -O3)
target_compile_features(dsyev PRIVATE c_std_99)