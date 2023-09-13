include(ExternalProject)

ExternalProject_Add(
        gperftools
        PREFIX ${CMAKE_BINARY_DIR}/gperftools
        GIT_REPOSITORY https://github.com/gperftools/gperftools.git
        GIT_TAG gperftools-2.13

        SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/gperftools

        BUILD_IN_SOURCE 1
        CONFIGURE_COMMAND ${CMAKE_BINARY_DIR}/_deps/gperftools/autogen.sh && ${CMAKE_BINARY_DIR}/_deps/gperftools/configure
        BUILD_COMMAND make
        INSTALL_COMMAND make install

        UPDATE_COMMAND ""
        LOG_DOWNLOAD ON
)