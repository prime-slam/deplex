include(FetchContent)

FetchContent_Declare(
        ctpl
        GIT_REPOSITORY https://github.com/vit-vit/CTPL.git
        GIT_TAG ctpl_v.0.0.2
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
)

FetchContent_GetProperties(ctpl)
if(NOT ctpl_POPULATED)
    FetchContent_Populate(ctpl)
endif()


add_library(ctpl INTERFACE)
target_include_directories(ctpl INTERFACE ${ctpl_SOURCE_DIR})

FetchContent_MakeAvailable(ctpl)