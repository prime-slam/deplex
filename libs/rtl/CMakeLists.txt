add_library(rtl INTERFACE)

target_include_directories(
  rtl
  SYSTEM
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)